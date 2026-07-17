"""Routing labels: per query, which tower (GCN dense vs SPLADE) ranks its judged products better."""

import pandas as pd
import torch
import torch.nn.functional as F

from app.common import get_logger, load_parquet
from app.constants import esci as C
from app.constants import router as R
from app.constants import splade as S
from app.constants import train as T
from app.databuilder import p2q_path, queries_path
from app.helpers.training import encode_texts, encode_texts_chunked

log = get_logger(__name__)

LABEL_GCN = 0
LABEL_SPLADE = 1
LABEL_TIE = -1


def _graph_maps(locale: str) -> tuple[dict, dict]:
    p2q = load_parquet(p2q_path(locale))
    neighbors_of = dict(
        zip(p2q[C.COL_PRODUCT_ID], (list(n) for n in p2q["neighbor_query_ids"]))
    )
    queries = load_parquet(queries_path(locale))
    query_text_of = dict(zip(queries[C.COL_QUERY_ID], queries[C.COL_QUERY]))
    return neighbors_of, query_text_of


@torch.no_grad()
def build_labels(
    locale: str,
    split: str,
    n_queries: int,
    gcn_encoder,
    gcn_head,
    gcn_tokenizer,
    splade_model,
    splade_tokenizer,
    device: str,
    num_neighbors: int = T.NUM_NEIGHBORS,
    tie_epsilon: float = R.TIE_EPSILON,
    seed: int = C.SEED,
) -> pd.DataFrame:
    from app.eval import _metrics_from_scores, eval_queries
    from app.train_gcn_head import encode_products

    neighbors_of, query_text_of = _graph_maps(locale)

    rows = []
    for query, pids, labels, texts in eval_queries(locale, n_queries, seed, split):
        batch = {
            "product_texts": texts,
            "neighbors": [
                [
                    query_text_of[q]
                    for q in neighbors_of.get(p, [])[:num_neighbors]
                    if q in query_text_of
                ]
                for p in pids
            ],
        }
        x_q = encode_texts(
            gcn_encoder, gcn_tokenizer, [query], T.MAX_QUERY_TOKENS, device
        )
        x_p = encode_products(
            gcn_encoder, gcn_head, gcn_tokenizer, batch, num_neighbors, device
        )
        dense = (F.normalize(x_q, dim=-1) @ F.normalize(x_p, dim=-1).T).squeeze(0)
        ndcg_gcn, _ = _metrics_from_scores(dense, labels)

        s_q = encode_texts(
            splade_model, splade_tokenizer, [query], S.MAX_QUERY_TOKENS, device
        )
        s_d = encode_texts_chunked(
            splade_model, splade_tokenizer, texts, S.MAX_PRODUCT_TOKENS, device
        )
        sparse = (s_q @ s_d.T).squeeze(0)
        ndcg_splade, _ = _metrics_from_scores(sparse, labels)

        delta = ndcg_gcn - ndcg_splade
        if abs(delta) < tie_epsilon:
            label = LABEL_TIE
        else:
            label = LABEL_GCN if delta > 0 else LABEL_SPLADE
        rows.append((locale, query, ndcg_gcn, ndcg_splade, label))

    df = pd.DataFrame(
        rows, columns=[C.COL_LOCALE, C.COL_QUERY, "ndcg_gcn", "ndcg_splade", "label"]
    )
    counts = df["label"].value_counts().to_dict()
    log.info(
        "labels[%s/%s]: %s rows | gcn_wins=%s splade_wins=%s ties=%s",
        locale,
        split,
        len(df),
        counts.get(LABEL_GCN, 0),
        counts.get(LABEL_SPLADE, 0),
        counts.get(LABEL_TIE, 0),
    )
    return df


@torch.no_grad()
def base_query_embeddings(
    gcn_encoder, tokenizer, queries: list[str], device: str, chunk: int = 64
) -> torch.Tensor:
    """Frozen base-model CLS features for the router (LoRA adapters disabled)."""
    parts = []
    with gcn_encoder.backbone.disable_adapter():
        for i in range(0, len(queries), chunk):
            parts.append(
                encode_texts(
                    gcn_encoder,
                    tokenizer,
                    queries[i : i + chunk],
                    T.MAX_QUERY_TOKENS,
                    device,
                )
            )
    return torch.cat(parts, dim=0)
