"""Shared training pieces: locale data, batch InfoNCE, locale mixing, tokenized encoding."""

import random

import pandas as pd
import torch
import torch.nn.functional as F

from app.common import get_logger, load_parquet
from app.constants import esci as C
from app.databuilder import p2q_path, products_path, qp_path, queries_path

log = get_logger(__name__)


class LocaleData:
    def __init__(
        self,
        locale: str,
        limit: int | None = None,
        seed: int = C.SEED,
        require_negs: bool = False,
        max_per_query: int | None = None,
        with_graph: bool = True,
    ):
        qp = load_parquet(qp_path(C.SPLIT_TRAIN, locale))
        if require_negs:
            qp = qp[qp["neg_product_ids"].map(len) > 0]
        if max_per_query:
            qp = qp.groupby(C.COL_QUERY_ID, sort=False).head(max_per_query)
        if limit:
            qp = qp.head(limit)
        log.info("rows[%s]: %s after signal filters", locale, f"{len(qp):,}")
        self.queries = list(qp[C.COL_QUERY])
        self.pos_ids = list(qp[C.COL_PRODUCT_ID])
        self.neg_ids = [list(n) for n in qp["neg_product_ids"]]

        prods = load_parquet(
            products_path(locale), columns=[C.COL_PRODUCT_ID, C.COL_PRODUCT_TEXT]
        )
        self.text_of = dict(zip(prods[C.COL_PRODUCT_ID], prods[C.COL_PRODUCT_TEXT]))

        self.neighbors_of: dict = {}
        self.query_text_of: dict = {}
        if with_graph:
            p2q = load_parquet(p2q_path(locale))
            self.neighbors_of = dict(
                zip(
                    p2q[C.COL_PRODUCT_ID],
                    (list(n) for n in p2q["neighbor_query_ids"]),
                )
            )
            queries = load_parquet(queries_path(locale))
            self.query_text_of = dict(
                zip(queries[C.COL_QUERY_ID], queries[C.COL_QUERY])
            )

        self._order = list(range(len(self.queries)))
        self._cursor = 0
        self._rng = random.Random(seed)
        self._rng.shuffle(self._order)

    def __len__(self) -> int:
        return len(self.queries)

    def neighbor_texts(self, product_id: str, k: int) -> list[str]:
        qids = self.neighbors_of.get(product_id, [])[:k]
        return [self.query_text_of[q] for q in qids if q in self.query_text_of]

    def next_batch(self, size: int, max_negs: int, num_neighbors: int = 0) -> dict:
        idxs = []
        while len(idxs) < size:
            if self._cursor >= len(self._order):
                self._rng.shuffle(self._order)
                self._cursor = 0
            idxs.append(self._order[self._cursor])
            self._cursor += 1

        queries, product_ids, product_texts, neighbors = [], [], [], []
        for i in idxs:
            query = self.queries[i]
            queries.append(query)
            pid = self.pos_ids[i]
            product_ids.append(pid)
            product_texts.append(self.text_of[pid])
            # the anchor query must not leak into its positive's neighbors (paper Fig 1)
            texts = self.neighbor_texts(pid, num_neighbors + 1)
            neighbors.append([t for t in texts if t != query][:num_neighbors])
        for i in idxs:
            for pid in self.neg_ids[i][:max_negs]:
                if pid not in self.text_of:
                    continue
                product_ids.append(pid)
                product_texts.append(self.text_of[pid])
                neighbors.append(self.neighbor_texts(pid, num_neighbors))

        return {
            "queries": queries,
            "product_ids": product_ids,
            "product_texts": product_texts,
            "neighbors": neighbors,
            "pos_idx": list(range(size)),
        }


def encode_texts(encoder, tokenizer, texts, max_tokens, device):
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    ).to(device)
    return encoder(tok["input_ids"], tok["attention_mask"])


def info_nce(x_q, x_p, pos_idx, product_ids, temperature, normalize=True):
    if normalize:
        x_q = F.normalize(x_q, dim=-1)
        x_p = F.normalize(x_p, dim=-1)
    logits = x_q @ x_p.T / temperature  # (B, P)

    # same product id elsewhere in the batch is not a real negative
    pid_codes = pd.factorize(pd.Series(product_ids))[0]
    pid_t = torch.as_tensor(pid_codes, device=logits.device)
    pos_t = torch.as_tensor(pos_idx, device=logits.device)
    dup = (pid_t.unsqueeze(0) == pid_t[pos_t].unsqueeze(1)) & (
        torch.arange(len(product_ids), device=logits.device).unsqueeze(0)
        != pos_t.unsqueeze(1)
    )
    logits = logits.masked_fill(dup, float("-inf"))
    return F.cross_entropy(logits, pos_t)


def locale_weights(sizes: dict[str, int], smoothing: float) -> dict[str, float]:
    raw = {loc: n**smoothing for loc, n in sizes.items()}
    total = sum(raw.values())
    return {loc: w / total for loc, w in raw.items()}
