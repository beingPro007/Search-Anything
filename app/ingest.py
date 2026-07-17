"""Product ingestion: encode vector shards (resumable), upsert to Qdrant, search."""

import argparse
import uuid
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

from app.common import ensure_dir, get_logger, load_parquet, save_parquet
from app.constants import esci as C
from app.constants import ingest as I
from app.constants import splade as S
from app.constants import train as T
from app.databuilder import p2q_path, products_path, queries_path
from app.helpers.encoding import (
    dense_product_vectors,
    sparse_product_vectors,
    sparse_query_vector,
)
from app.helpers.training import encode_texts

log = get_logger(__name__)


def shard_path(locale: str, idx: int) -> Path:
    return I.VECTORS_DIR / I.SHARD_FILE.format(locale=locale, idx=idx)


def shard_plan(
    locales: tuple[str, ...], shard_size: int = I.SHARD_SIZE
) -> list[tuple[str, int]]:
    plan = []
    for loc in locales:
        rows = pq.ParquetFile(products_path(loc)).metadata.num_rows
        for idx in range((rows + shard_size - 1) // shard_size):
            plan.append((loc, idx))
    return plan


def _graph_maps(locale: str) -> tuple[dict, dict]:
    p2q = load_parquet(p2q_path(locale))
    neighbors_of = dict(
        zip(p2q[C.COL_PRODUCT_ID], (list(n) for n in p2q["neighbor_query_ids"]))
    )
    queries = load_parquet(queries_path(locale))
    query_text_of = dict(zip(queries[C.COL_QUERY_ID], queries[C.COL_QUERY]))
    return neighbors_of, query_text_of


def encode_shard(
    locale: str,
    idx: int,
    shard_size: int = I.SHARD_SIZE,
    limit: int | None = None,
    force: bool = False,
) -> Path:
    out = shard_path(locale, idx)
    if out.exists() and not force:
        log.info("cached shard: %s", out)
        return out

    from transformers import AutoTokenizer

    from app.models.gcn import load_gcn
    from app.models.splade import load_splade

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcn, head, gcn_cfg = load_gcn(T.MODELS_DIR, device)
    gcn_tok = AutoTokenizer.from_pretrained(gcn_cfg["base_model"])
    splade, splade_cfg = load_splade(S.SPLADE_DIR, device)
    splade_tok = AutoTokenizer.from_pretrained(splade_cfg["base_model"])
    neighbors_of, query_text_of = _graph_maps(locale)

    products = load_parquet(products_path(locale))
    start = idx * shard_size
    chunk = products.iloc[start : start + shard_size]
    if limit:
        chunk = chunk.head(limit)
    log.info("encoding shard %s/%s: %s products", locale, idx, f"{len(chunk):,}")

    rows = []
    pids = list(chunk[C.COL_PRODUCT_ID])
    titles = list(chunk[C.COL_TITLE])
    texts = list(chunk[C.COL_PRODUCT_TEXT])
    for i in range(0, len(pids), I.DENSE_BATCH):
        b_pids = pids[i : i + I.DENSE_BATCH]
        b_texts = texts[i : i + I.DENSE_BATCH]
        b_neigh = [
            [
                query_text_of[q]
                for q in neighbors_of.get(p, [])[: T.NUM_NEIGHBORS]
                if q in query_text_of
            ]
            for p in b_pids
        ]
        dense = dense_product_vectors(gcn, head, gcn_tok, b_texts, b_neigh, device)
        sparse: list = []
        for j in range(0, len(b_texts), I.SPARSE_BATCH):
            sparse.extend(
                sparse_product_vectors(
                    splade, splade_tok, b_texts[j : j + I.SPARSE_BATCH], device
                )
            )
        for pid, title, vec, (s_idx, s_val) in zip(
            b_pids, titles[i : i + I.DENSE_BATCH], dense.cpu(), sparse
        ):
            rows.append((pid, locale, title, vec.tolist(), s_idx, s_val))

    df = pd.DataFrame(
        rows,
        columns=[
            C.COL_PRODUCT_ID,
            C.COL_LOCALE,
            C.COL_TITLE,
            "dense",
            "sparse_indices",
            "sparse_values",
        ],
    )
    return save_parquet(df, out)


def make_client(url: str | None = None, api_key: str | None = None):
    from qdrant_client import QdrantClient

    if not url:
        log.info("no qdrant url -> in-memory client")
        return QdrantClient(":memory:")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client, dim: int) -> None:
    from qdrant_client import models as qm

    if client.collection_exists(I.COLLECTION):
        return
    client.create_collection(
        collection_name=I.COLLECTION,
        vectors_config={
            I.DENSE_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
        },
        sparse_vectors_config={I.SPARSE_VECTOR_NAME: qm.SparseVectorParams()},
    )
    log.info("created collection %s (dense dim=%s)", I.COLLECTION, dim)


def point_id(locale: str, product_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{locale}:{product_id}"))


def upsert_shard(client, path: Path, batch: int = I.UPSERT_BATCH) -> int:
    from qdrant_client import models as qm

    df = pd.read_parquet(path)
    ensure_collection(client, len(df["dense"].iloc[0]))
    points = [
        qm.PointStruct(
            id=point_id(r[C.COL_LOCALE], r[C.COL_PRODUCT_ID]),
            vector={
                I.DENSE_VECTOR_NAME: list(r["dense"]),
                I.SPARSE_VECTOR_NAME: qm.SparseVector(
                    indices=list(r["sparse_indices"]), values=list(r["sparse_values"])
                ),
            },
            payload={
                C.COL_PRODUCT_ID: r[C.COL_PRODUCT_ID],
                C.COL_LOCALE: r[C.COL_LOCALE],
                C.COL_TITLE: r[C.COL_TITLE],
            },
        )
        for _, r in df.iterrows()
    ]
    for i in range(0, len(points), batch):
        client.upsert(collection_name=I.COLLECTION, points=points[i : i + batch])
    log.info("upserted %s points <- %s", f"{len(points):,}", path.name)
    return len(points)


def upsert_all(client, locales: tuple[str, ...]) -> int:
    total = 0
    for loc in locales:
        idx = 0
        while shard_path(loc, idx).exists():
            total += upsert_shard(client, shard_path(loc, idx))
            idx += 1
    log.info("total upserted: %s", f"{total:,}")
    return total


@torch.no_grad()
def search_dense(client, gcn_encoder, tokenizer, query: str, device, limit=5):
    import torch.nn.functional as F

    x = encode_texts(gcn_encoder, tokenizer, [query], T.MAX_QUERY_TOKENS, device)
    vec = F.normalize(x, dim=-1)[0].cpu().tolist()
    return client.query_points(
        I.COLLECTION, query=vec, using=I.DENSE_VECTOR_NAME, limit=limit
    ).points


def search_sparse(client, splade_model, tokenizer, query: str, device, limit=5):
    from qdrant_client import models as qm

    idx, val = sparse_query_vector(splade_model, tokenizer, query, device)
    return client.query_points(
        I.COLLECTION,
        query=qm.SparseVector(indices=idx, values=val),
        using=I.SPARSE_VECTOR_NAME,
        limit=limit,
    ).points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="encode + ingest product vectors")
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("encode", help="encode one shard to parquet")
    e.add_argument("--locale", choices=C.LOCALES, required=True)
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--force", action="store_true")

    u = sub.add_parser("upsert", help="upsert all shards to qdrant")
    u.add_argument("--locales", nargs="+", choices=C.LOCALES, default=None)
    u.add_argument("--url", default=None)
    u.add_argument("--api-key", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dir(I.VECTORS_DIR)
    if args.command == "encode":
        encode_shard(args.locale, args.shard, limit=args.limit, force=args.force)
    else:
        client = make_client(args.url, args.api_key)
        upsert_all(client, tuple(args.locales) if args.locales else C.LOCALES)
