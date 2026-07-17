"""SPLADE training: InfoNCE ranking loss + FLOPS sparsity regularizers on ESCI pairs."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable

import torch

from app.common import ensure_dir, get_logger
from app.constants import esci as C
from app.constants import splade as S
from app.helpers.training import LocaleData, encode_texts, info_nce, locale_weights
from app.models.splade import SpladeEncoder, flops_loss

log = get_logger(__name__)


def lambda_ramp(step: int, total: int, ramp_frac: float) -> float:
    ramp_steps = max(1, int(total * ramp_frac))
    return min(1.0, (step / ramp_steps) ** 2)


def save_model(model: SpladeEncoder, out_dir: Path, config: dict) -> None:
    ensure_dir(out_dir)
    model.backbone.save_pretrained(out_dir)
    (out_dir / "splade_config.json").write_text(json.dumps(config, indent=2))
    log.info("saved splade -> %s", out_dir)


def train(
    locales: tuple[str, ...] | None = None,
    base_model: str = S.BASE_MODEL,
    epochs: int = S.EPOCHS,
    batch_size: int = S.BATCH_SIZE,
    max_steps: int | None = None,
    limit: int | None = None,
    lr: float = S.LEARNING_RATE,
    max_negs: int = S.MAX_HARD_NEGATIVES,
    lambda_query: float = S.LAMBDA_QUERY,
    lambda_doc: float = S.LAMBDA_DOC,
    temperature: float = S.TEMPERATURE,
    seed: int = C.SEED,
    out_dir: Path | str | None = None,
    eval_queries: int = S.EVAL_QUERIES_PER_LOCALE,
    require_negs: bool = False,
    max_per_query: int | None = None,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict:
    from transformers import AutoTokenizer

    locales = tuple(locales) if locales else C.LOCALES
    out_dir = Path(out_dir) if out_dir else S.SPLADE_DIR
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
            with_graph=False,
        )
        for loc in locales
    }
    weights = locale_weights(
        {loc: len(d) for loc, d in data.items()}, S.LOCALE_SMOOTHING
    )
    log.info("locale weights: %s", {k: round(v, 3) for k, v in weights.items()})

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = SpladeEncoder(base_model).to(device)
    if device == "cuda":
        model.backbone.gradient_checkpointing_enable()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("trainable params: %s (device=%s)", f"{n_trainable:,}", device)

    total_rows = sum(len(d) for d in data.values())
    steps = max_steps or math.ceil(total_rows / batch_size) * epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=S.WEIGHT_DECAY
    )
    warmup = max(1, int(steps * S.WARMUP_FRAC))
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
        "max_negs": max_negs,
        "lambda_query": lambda_query,
        "lambda_doc": lambda_doc,
        "temperature": temperature,
        "vocab_size": model.vocab_size,
    }

    model.train()
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    running = {"loss": 0.0, "rank": 0.0, "nnz_q": 0.0, "nnz_d": 0.0}
    for step in range(1, steps + 1):
        loc = rng.choices(list(weights), weights=list(weights.values()))[0]
        batch = data[loc].next_batch(batch_size, max_negs)
        ramp = lambda_ramp(step, steps, S.LAMBDA_RAMP_FRAC)
        with autocast:
            x_q = encode_texts(
                model, tokenizer, batch["queries"], S.MAX_QUERY_TOKENS, device
            )
            x_d = encode_texts(
                model, tokenizer, batch["product_texts"], S.MAX_PRODUCT_TOKENS, device
            )
            rank = info_nce(
                x_q,
                x_d,
                batch["pos_idx"],
                batch["product_ids"],
                temperature,
                normalize=False,
            )
            loss = (
                rank
                + ramp * lambda_query * flops_loss(x_q)
                + ramp * lambda_doc * flops_loss(x_d)
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), S.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        running["loss"] += loss.item()
        running["rank"] += rank.item()
        running["nnz_q"] += (x_q > 0).float().sum(dim=-1).mean().item()
        running["nnz_d"] += (x_d > 0).float().sum(dim=-1).mean().item()
        if step % S.LOG_EVERY == 0:
            n = S.LOG_EVERY
            log.info(
                "step %s/%s | loss %.4f | rank %.4f | nnz q/d %.0f/%.0f | lr %.2e | %s",
                step,
                steps,
                running["loss"] / n,
                running["rank"] / n,
                running["nnz_q"] / n,
                running["nnz_d"] / n,
                scheduler.get_last_lr()[0],
                loc,
            )
            running = {k: 0.0 for k in running}
        if step % S.CHECKPOINT_EVERY == 0:
            save_model(model, out_dir, config | {"step": step})
            if on_checkpoint:
                on_checkpoint()

    save_model(model, out_dir, config | {"step": steps})
    if on_checkpoint:
        on_checkpoint()  # persist the model before eval can fail

    del optimizer, scheduler
    if device == "cuda":
        torch.cuda.empty_cache()

    from app.eval import evaluate_splade

    model.eval()
    metrics = {
        loc: evaluate_splade(model, tokenizer, loc, eval_queries, device)
        for loc in locales
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics: %s", metrics)
    if on_checkpoint:
        on_checkpoint()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train SPLADE sparse encoder")
    parser.add_argument("--locales", nargs="+", choices=C.LOCALES, default=None)
    parser.add_argument("--base-model", default=S.BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=S.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=S.BATCH_SIZE)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=S.LEARNING_RATE)
    parser.add_argument("--max-negs", type=int, default=S.MAX_HARD_NEGATIVES)
    parser.add_argument("--lambda-query", type=float, default=S.LAMBDA_QUERY)
    parser.add_argument("--lambda-doc", type=float, default=S.LAMBDA_DOC)
    parser.add_argument("--eval-queries", type=int, default=S.EVAL_QUERIES_PER_LOCALE)
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
        max_negs=args.max_negs,
        lambda_query=args.lambda_query,
        lambda_doc=args.lambda_doc,
        eval_queries=args.eval_queries,
        require_negs=args.require_negs,
        max_per_query=args.max_per_query,
        out_dir=args.out_dir,
    )
