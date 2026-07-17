"""Reranking eval on ESCI test judgments: graded NDCG + Recall@10 over E products."""

import math
import random

import torch
import torch.nn.functional as F

from app.common import get_logger, load_parquet
from app.constants import esci as C
from app.constants import splade as S
from app.constants import train as T
from app.databuilder import examples_path, p2q_path, products_path, queries_path
from app.helpers.training import encode_texts, encode_texts_chunked

log = get_logger(__name__)


def ndcg(ranked_gains: list[float]) -> float:
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ranked_gains))
    ideal = sorted(ranked_gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def eval_queries(locale: str, n_queries: int, seed: int, split: str = C.SPLIT_TEST):
    """Yield (query_text, product_ids, labels, product_texts) for judged queries."""
    examples = load_parquet(examples_path(locale))
    test = examples[examples[C.COL_SPLIT] == split]
    prods = load_parquet(
        products_path(locale), columns=[C.COL_PRODUCT_ID, C.COL_PRODUCT_TEXT]
    )
    text_of = dict(zip(prods[C.COL_PRODUCT_ID], prods[C.COL_PRODUCT_TEXT]))

    grouped = list(test.groupby(C.COL_QUERY_ID))
    random.Random(seed).shuffle(grouped)
    served = 0
    for _, group in grouped:
        if served >= n_queries:
            return
        labels = list(group[C.COL_LABEL])
        if len(labels) < 2 or C.LABEL_EXACT not in labels:
            continue
        pids = list(group[C.COL_PRODUCT_ID])
        served += 1
        yield (
            group[C.COL_QUERY].iloc[0],
            pids,
            labels,
            [text_of[p] for p in pids],
        )


def _metrics_from_scores(scores: torch.Tensor, labels: list[str]) -> tuple[float, float]:
    order = scores.argsort(descending=True).tolist()
    gains = [T.NDCG_GAINS[labels[i]] for i in order]
    n_exact = sum(1 for lb in labels if lb == C.LABEL_EXACT)
    top10_exact = sum(1 for i in order[:10] if labels[i] == C.LABEL_EXACT)
    return ndcg(gains), top10_exact / min(n_exact, 10)


@torch.no_grad()
def evaluate(
    encoder,
    head,
    tokenizer,
    locale: str,
    n_queries: int = T.EVAL_QUERIES_PER_LOCALE,
    num_neighbors: int = T.NUM_NEIGHBORS,
    device: str = "cpu",
    seed: int = C.SEED,
) -> dict:
    from app.train_gcn_head import encode_products

    p2q = load_parquet(p2q_path(locale))
    neighbors_of = dict(
        zip(p2q[C.COL_PRODUCT_ID], (list(n) for n in p2q["neighbor_query_ids"]))
    )
    queries = load_parquet(queries_path(locale))
    query_text_of = dict(zip(queries[C.COL_QUERY_ID], queries[C.COL_QUERY]))

    ndcgs, recalls = [], []
    for query, pids, labels, texts in eval_queries(locale, n_queries, seed):
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
        x_q = encode_texts(encoder, tokenizer, [query], T.MAX_QUERY_TOKENS, device)
        x_p = encode_products(encoder, head, tokenizer, batch, num_neighbors, device)
        scores = (F.normalize(x_q, dim=-1) @ F.normalize(x_p, dim=-1).T).squeeze(0)
        n, r = _metrics_from_scores(scores, labels)
        ndcgs.append(n)
        recalls.append(r)

    result = {
        "queries": len(ndcgs),
        "ndcg": round(sum(ndcgs) / max(len(ndcgs), 1), 4),
        "recall@10": round(sum(recalls) / max(len(recalls), 1), 4),
    }
    log.info("eval[%s]: %s", locale, result)
    return result


@torch.no_grad()
def evaluate_dense_base(
    encoder,
    tokenizer,
    locale: str,
    n_queries: int = T.EVAL_QUERIES_PER_LOCALE,
    device: str = "cpu",
    seed: int = C.SEED,
) -> dict:
    """Plain CLS-embedding cosine reranking: no GCN head, no neighbors."""
    ndcgs, recalls = [], []
    for query, _, labels, texts in eval_queries(locale, n_queries, seed):
        x_q = encode_texts(encoder, tokenizer, [query], T.MAX_QUERY_TOKENS, device)
        x_p = encode_texts_chunked(
            encoder, tokenizer, texts, T.MAX_PRODUCT_TOKENS, device
        )
        scores = (F.normalize(x_q, dim=-1) @ F.normalize(x_p, dim=-1).T).squeeze(0)
        n, r = _metrics_from_scores(scores, labels)
        ndcgs.append(n)
        recalls.append(r)

    result = {
        "queries": len(ndcgs),
        "ndcg": round(sum(ndcgs) / max(len(ndcgs), 1), 4),
        "recall@10": round(sum(recalls) / max(len(recalls), 1), 4),
    }
    log.info("eval-base[%s]: %s", locale, result)
    return result


@torch.no_grad()
def evaluate_splade(
    model,
    tokenizer,
    locale: str,
    n_queries: int = S.EVAL_QUERIES_PER_LOCALE,
    device: str = "cpu",
    seed: int = C.SEED,
) -> dict:
    ndcgs, recalls, nnz_q, nnz_d = [], [], [], []
    for query, _, labels, texts in eval_queries(locale, n_queries, seed):
        x_q = encode_texts(model, tokenizer, [query], S.MAX_QUERY_TOKENS, device)
        x_d = encode_texts_chunked(
            model, tokenizer, texts, S.MAX_PRODUCT_TOKENS, device
        )
        scores = (x_q @ x_d.T).squeeze(0)
        n, r = _metrics_from_scores(scores, labels)
        ndcgs.append(n)
        recalls.append(r)
        nnz_q.append((x_q > 0).float().sum().item())
        nnz_d.append((x_d > 0).float().sum(dim=-1).mean().item())

    result = {
        "queries": len(ndcgs),
        "ndcg": round(sum(ndcgs) / max(len(ndcgs), 1), 4),
        "recall@10": round(sum(recalls) / max(len(recalls), 1), 4),
        "nnz_query": round(sum(nnz_q) / max(len(nnz_q), 1), 1),
        "nnz_doc": round(sum(nnz_d) / max(len(nnz_d), 1), 1),
    }
    log.info("eval-splade[%s]: %s", locale, result)
    return result
