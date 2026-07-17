"""Train-split graph: product->query edges (E labels) and query->query co-relevance."""

import random
from itertools import combinations

import pandas as pd

from app.common import get_logger
from app.constants import esci as C

log = get_logger(__name__)


def build_graph(
    examples: pd.DataFrame,
    max_p2q: int = C.MAX_NEIGHBOR_QUERIES,
    max_q2q: int = C.MAX_Q2Q_PER_PRODUCT,
    seed: int = C.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train = examples[
        (examples[C.COL_SPLIT] == C.SPLIT_TRAIN)
        & examples[C.COL_LABEL].isin(C.POSITIVE_LABELS)
    ]

    queries = (
        train[[C.COL_QUERY_ID, C.COL_QUERY]]
        .drop_duplicates(C.COL_QUERY_ID)
        .reset_index(drop=True)
    )

    by_product: dict = {}
    for qid, pid in zip(train[C.COL_QUERY_ID], train[C.COL_PRODUCT_ID]):
        by_product.setdefault(pid, set()).add(qid)

    p2q_rows, q2q_rows, seen = [], [], set()
    for pid, qid_set in by_product.items():
        qids = sorted(qid_set)
        neigh = qids if len(qids) <= max_p2q else rng.sample(qids, max_p2q)
        p2q_rows.append((pid, neigh))
        if len(qids) < 2:
            continue
        pairs = list(combinations(qids, 2))
        if len(pairs) > max_q2q:
            pairs = rng.sample(pairs, max_q2q)
        for a, b in pairs:
            if (a, b) not in seen:
                seen.add((a, b))
                q2q_rows.append((pid, a, b))

    p2q = pd.DataFrame(p2q_rows, columns=[C.COL_PRODUCT_ID, "neighbor_query_ids"])
    q2q = pd.DataFrame(
        q2q_rows, columns=[C.COL_PRODUCT_ID, "query_id_a", "query_id_b"]
    )
    log.info(
        "graph: %s queries, %s p2q products, %s q2q pairs",
        f"{len(queries):,}",
        f"{len(p2q):,}",
        f"{len(q2q):,}",
    )
    return queries, p2q, q2q
