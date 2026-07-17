"""Batch product encoding for ingestion: dense GCN vectors + SPLADE sparse terms."""

import torch
import torch.nn.functional as F

from app.constants import ingest as I
from app.constants import splade as S
from app.constants import train as T
from app.helpers.training import encode_texts


def _topk_sparse(rep: torch.Tensor, top_k: int) -> tuple[list[int], list[float]]:
    values, indices = rep.topk(min(top_k, rep.shape[-1]))
    keep = values > 0
    return indices[keep].cpu().tolist(), values[keep].cpu().tolist()


@torch.no_grad()
def dense_product_vectors(
    gcn_encoder, gcn_head, tokenizer, texts, neighbors, device
) -> torch.Tensor:
    from app.train_gcn_head import encode_products

    batch = {"product_texts": texts, "neighbors": neighbors}
    x = encode_products(
        gcn_encoder, gcn_head, tokenizer, batch, T.NUM_NEIGHBORS, device
    )
    return F.normalize(x, dim=-1)


@torch.no_grad()
def sparse_product_vectors(
    splade_model, tokenizer, texts, device, top_k: int = I.SPARSE_TOP_K
) -> list[tuple[list[int], list[float]]]:
    reps = encode_texts(splade_model, tokenizer, texts, S.MAX_PRODUCT_TOKENS, device)
    return [_topk_sparse(rep, top_k) for rep in reps]


@torch.no_grad()
def sparse_query_vector(
    splade_model, tokenizer, query: str, device, top_k: int = I.SPARSE_TOP_K
) -> tuple[list[int], list[float]]:
    reps = encode_texts(splade_model, tokenizer, [query], S.MAX_QUERY_TOKENS, device)
    return _topk_sparse(reps[0], top_k)
