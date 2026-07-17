"""GCN-head + LoRA training with InfoNCE on ESCI query-product pairs."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable

import torch

from app.common import ensure_dir, get_logger
from app.constants import esci as C
from app.constants import train as T
from app.helpers.training import LocaleData, encode_texts, info_nce, locale_weights
from app.models import GCNHead, TextEncoder

log = get_logger(__name__)


def encode_products(encoder, head, tokenizer, batch, num_neighbors, device):
    h_p = encode_texts(
        encoder, tokenizer, batch["product_texts"], T.MAX_PRODUCT_TOKENS, device
    )
    P, D = h_p.shape

    uniq: dict[str, int] = {}
    for texts in batch["neighbors"]:
        for t in texts:
            uniq.setdefault(t, len(uniq))
    h_n = torch.zeros(P, num_neighbors, D, device=device, dtype=h_p.dtype)
    mask = torch.zeros(P, num_neighbors, device=device, dtype=h_p.dtype)
    if uniq:
        h_uniq = encode_texts(
            encoder, tokenizer, list(uniq), T.MAX_QUERY_TOKENS, device
        )
        for i, texts in enumerate(batch["neighbors"]):
            for j, t in enumerate(texts[:num_neighbors]):
                h_n[i, j] = h_uniq[uniq[t]]
                mask[i, j] = 1.0
    return head(h_p, h_n, mask)


def save_model(encoder, head, out_dir: Path, config: dict) -> None:
    ensure_dir(out_dir)
    torch.save(head.state_dict(), out_dir / T.GCN_HEAD_FILE)
    encoder.backbone.save_pretrained(out_dir / T.LORA_DIR)
    (out_dir / T.CONFIG_FILE).write_text(json.dumps(config, indent=2))
    log.info("saved model -> %s", out_dir)


def train(
    locales: tuple[str, ...] | None = None,
    base_model: str = T.BASE_MODEL,
    epochs: int = T.EPOCHS,
    batch_size: int = T.BATCH_SIZE,
    max_steps: int | None = None,
    limit: int | None = None,
    lr: float = T.LEARNING_RATE,
    num_neighbors: int = T.NUM_NEIGHBORS,
    max_negs: int = T.MAX_HARD_NEGATIVES,
    temperature: float = T.TEMPERATURE,
    seed: int = C.SEED,
    out_dir: Path | str | None = None,
    eval_queries: int = T.EVAL_QUERIES_PER_LOCALE,
    require_negs: bool = False,
    max_per_query: int | None = None,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict:
    from transformers import AutoTokenizer

    locales = tuple(locales) if locales else C.LOCALES
    out_dir = Path(out_dir) if out_dir else T.MODELS_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)

    data = {
        loc: LocaleData(
            loc,
            limit=limit,
            seed=seed,
            require_negs=require_negs,
            max_per_query=max_per_query,
        )
        for loc in locales
    }
    weights = locale_weights(
        {loc: len(d) for loc, d in data.items()}, T.LOCALE_SMOOTHING
    )
    log.info("locale weights: %s", {k: round(v, 3) for k, v in weights.items()})

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    encoder = TextEncoder(base_model).to(device)
    head = GCNHead(encoder.hidden_size).to(device)
    if device == "cuda":
        encoder.backbone.gradient_checkpointing_enable()

    params = [p for p in encoder.parameters() if p.requires_grad] + list(
        head.parameters()
    )
    n_trainable = sum(p.numel() for p in params)
    log.info("trainable params: %s (device=%s)", f"{n_trainable:,}", device)

    total_rows = sum(len(d) for d in data.values())
    steps = max_steps or math.ceil(total_rows / batch_size) * epochs
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=T.WEIGHT_DECAY)
    warmup = min(T.WARMUP_STEPS, steps // 10 + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / steps),
    )

    config = {
        "base_model": base_model,
        "locales": list(locales),
        "batch_size": batch_size,
        "steps": steps,
        "lr": lr,
        "num_neighbors": num_neighbors,
        "max_negs": max_negs,
        "temperature": temperature,
        "hidden_size": encoder.hidden_size,
        "lora_r": T.LORA_R,
    }

    encoder.train()
    head.train()
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    running = 0.0
    for step in range(1, steps + 1):
        loc = rng.choices(list(weights), weights=list(weights.values()))[0]
        batch = data[loc].next_batch(batch_size, max_negs, num_neighbors)
        with autocast:
            x_q = encode_texts(
                encoder, tokenizer, batch["queries"], T.MAX_QUERY_TOKENS, device
            )
            x_p = encode_products(
                encoder, head, tokenizer, batch, num_neighbors, device
            )
            loss = info_nce(
                x_q, x_p, batch["pos_idx"], batch["product_ids"], temperature
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, T.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        running += loss.item()
        if step % T.LOG_EVERY == 0:
            log.info(
                "step %s/%s | loss %.4f | lr %.2e | locale %s",
                step,
                steps,
                running / T.LOG_EVERY,
                scheduler.get_last_lr()[0],
                loc,
            )
            running = 0.0
        if step % T.CHECKPOINT_EVERY == 0:
            save_model(encoder, head, out_dir, config | {"step": step})
            if on_checkpoint:
                on_checkpoint()

    save_model(encoder, head, out_dir, config | {"step": steps})
    if on_checkpoint:
        on_checkpoint()  # persist the model before eval can fail

    from app.eval import evaluate

    encoder.eval()
    head.eval()
    metrics = {
        loc: evaluate(
            encoder, head, tokenizer, loc, eval_queries, num_neighbors, device
        )
        for loc in locales
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics: %s", metrics)
    if on_checkpoint:
        on_checkpoint()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train GCN head + LoRA adapters")
    parser.add_argument("--locales", nargs="+", choices=C.LOCALES, default=None)
    parser.add_argument("--base-model", default=T.BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=T.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=T.BATCH_SIZE)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=T.LEARNING_RATE)
    parser.add_argument("--num-neighbors", type=int, default=T.NUM_NEIGHBORS)
    parser.add_argument("--max-negs", type=int, default=T.MAX_HARD_NEGATIVES)
    parser.add_argument("--eval-queries", type=int, default=T.EVAL_QUERIES_PER_LOCALE)
    parser.add_argument("--require-negs", action="store_true")
    parser.add_argument("--max-per-query", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        locales=args.locales,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        limit=args.limit,
        lr=args.lr,
        num_neighbors=args.num_neighbors,
        max_negs=args.max_negs,
        eval_queries=args.eval_queries,
        require_negs=args.require_negs,
        max_per_query=args.max_per_query,
        out_dir=args.out_dir,
    )
