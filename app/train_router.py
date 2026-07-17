"""Train the query router on tower-outcome labels; report routed vs single-tower NDCG."""

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F

from app.common import ensure_dir, get_logger, save_parquet
from app.constants import esci as C
from app.constants import router as R
from app.constants import splade as S
from app.constants import train as T
from app.helpers.routing import (
    LABEL_GCN,
    LABEL_SPLADE,
    LABEL_TIE,
    base_query_embeddings,
    build_labels,
)
from app.models.gcn import load_gcn
from app.models.router import RouterHead
from app.models.splade import load_splade

log = get_logger(__name__)


def labels_path(split: str, locale: str) -> Path:
    return R.ROUTER_DIR / R.LABELS_FILE.format(split=split, locale=locale)


def get_labels(
    locales, split, n_queries, force, gcn, head, gcn_tok, splade, splade_tok, device
) -> pd.DataFrame:
    frames = []
    for loc in locales:
        path = labels_path(split, loc)
        if path.exists() and not force:
            frames.append(pd.read_parquet(path))
            continue
        df = build_labels(
            loc, split, n_queries, gcn, head, gcn_tok, splade, splade_tok, device
        )
        save_parquet(df, path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def routed_metrics(df: pd.DataFrame, choice: torch.Tensor) -> dict:
    ndcg_gcn = torch.as_tensor(df["ndcg_gcn"].to_numpy())
    ndcg_splade = torch.as_tensor(df["ndcg_splade"].to_numpy())
    routed = torch.where(choice == LABEL_GCN, ndcg_gcn, ndcg_splade)
    decided = df["label"] != LABEL_TIE
    acc = (
        (choice[decided.to_numpy()] == torch.as_tensor(df.loc[decided, "label"].to_numpy()))
        .float()
        .mean()
        .item()
        if decided.any()
        else 0.0
    )
    return {
        "queries": len(df),
        "routed_ndcg": round(routed.mean().item(), 4),
        "always_gcn_ndcg": round(ndcg_gcn.mean().item(), 4),
        "always_splade_ndcg": round(ndcg_splade.mean().item(), 4),
        "oracle_ndcg": round(torch.maximum(ndcg_gcn, ndcg_splade).mean().item(), 4),
        "routing_accuracy": round(acc, 4),
        "gcn_share": round((choice == LABEL_GCN).float().mean().item(), 4),
    }


def train(
    locales: tuple[str, ...] | None = None,
    train_queries: int = R.TRAIN_QUERIES_PER_LOCALE,
    test_queries: int = R.TEST_QUERIES_PER_LOCALE,
    epochs: int = R.EPOCHS,
    lr: float = R.LEARNING_RATE,
    force_labels: bool = False,
    seed: int = C.SEED,
    out_dir: Path | str | None = None,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict:
    from transformers import AutoTokenizer

    locales = tuple(locales) if locales else C.LOCALES
    out_dir = Path(out_dir) if out_dir else R.ROUTER_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    gcn, head, gcn_cfg = load_gcn(T.MODELS_DIR, device)
    gcn_tok = AutoTokenizer.from_pretrained(gcn_cfg["base_model"])
    splade, splade_cfg = load_splade(S.SPLADE_DIR, device)
    splade_tok = AutoTokenizer.from_pretrained(splade_cfg["base_model"])
    log.info(
        "towers loaded: gcn step=%s | splade step=%s",
        gcn_cfg.get("step"),
        splade_cfg.get("step"),
    )

    train_df = get_labels(
        locales, C.SPLIT_TRAIN, train_queries, force_labels,
        gcn, head, gcn_tok, splade, splade_tok, device,
    )
    test_df = get_labels(
        locales, C.SPLIT_TEST, test_queries, force_labels,
        gcn, head, gcn_tok, splade, splade_tok, device,
    )

    decided = train_df[train_df["label"] != LABEL_TIE].reset_index(drop=True)
    log.info("router training rows: %s (ties dropped)", len(decided))
    x_train = base_query_embeddings(
        gcn, gcn_tok, list(decided[C.COL_QUERY]), device
    ).float()
    y_train = torch.as_tensor(decided["label"].to_numpy(), device=device)
    x_test = base_query_embeddings(
        gcn, gcn_tok, list(test_df[C.COL_QUERY]), device
    ).float()

    router = RouterHead(x_train.shape[1], R.HIDDEN).to(device)
    counts = torch.bincount(y_train, minlength=2).float()
    class_weights = counts.sum() / (2 * counts.clamp(min=1))
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=lr, weight_decay=R.WEIGHT_DECAY
    )

    router.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(x_train), device=device)
        total = 0.0
        for i in range(0, len(perm), R.BATCH_SIZE):
            idx = perm[i : i + R.BATCH_SIZE]
            logits = router(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx], weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        log.info("epoch %s/%s | loss %.4f", epoch, epochs, total / len(perm))

    router.eval()
    with torch.no_grad():
        choice = router(x_test).argmax(dim=-1).cpu()
    metrics = routed_metrics(test_df, choice)
    log.info("router metrics: %s", metrics)

    ensure_dir(out_dir)
    torch.save(router.state_dict(), out_dir / R.ROUTER_HEAD_FILE)
    config = {
        "dim": x_train.shape[1],
        "hidden": R.HIDDEN,
        "gcn_step": gcn_cfg.get("step"),
        "splade_step": splade_cfg.get("step"),
        "train_rows": len(decided),
        "tie_epsilon": R.TIE_EPSILON,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("saved router -> %s", out_dir)
    if on_checkpoint:
        on_checkpoint()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the two-tower query router")
    parser.add_argument("--locales", nargs="+", choices=C.LOCALES, default=None)
    parser.add_argument("--train-queries", type=int, default=R.TRAIN_QUERIES_PER_LOCALE)
    parser.add_argument("--test-queries", type=int, default=R.TEST_QUERIES_PER_LOCALE)
    parser.add_argument("--epochs", type=int, default=R.EPOCHS)
    parser.add_argument("--lr", type=float, default=R.LEARNING_RATE)
    parser.add_argument("--force-labels", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        locales=args.locales,
        train_queries=args.train_queries,
        test_queries=args.test_queries,
        epochs=args.epochs,
        lr=args.lr,
        force_labels=args.force_labels,
        out_dir=args.out_dir,
    )
