"""Contrastive pair construction. Pairs store product ids; texts live in products_clean."""

import random
from itertools import combinations

import pandas as pd

from app.common import get_logger
from app.constants import esci as C

log = get_logger(__name__)


def product_text_lookup(products: pd.DataFrame) -> dict[tuple[str, str], str]:
    return dict(
        zip(
            zip(products[C.COL_LOCALE], products[C.COL_PRODUCT_ID]),
            products[C.COL_PRODUCT_TEXT],
        )
    )


def _negatives_by_query(examples: pd.DataFrame) -> dict:
    tier = {label: i for i, label in enumerate(C.NEGATIVE_LABEL_PRIORITY)}
    negs = examples[examples[C.COL_LABEL].isin(tier)]
    out: dict = {}
    cols = (C.COL_QUERY_ID, C.COL_LABEL, C.COL_PRODUCT_ID, C.COL_LOCALE)
    for qid, label, pid, locale in zip(*(negs[c] for c in cols)):
        out.setdefault(qid, []).append((tier[label], pid, locale))
    return out


def build_query_product(
    examples: pd.DataFrame,
    products: pd.DataFrame,
    max_negatives: int = C.MAX_NEGATIVES_PER_ROW,
    seed: int = C.SEED,
) -> pd.DataFrame:
    rng = random.Random(seed)
    text_of = product_text_lookup(products)
    negs_of = _negatives_by_query(examples)

    pos = examples[examples[C.COL_LABEL].isin(C.POSITIVE_LABELS)]
    rows = []
    cols = (C.COL_QUERY_ID, C.COL_QUERY, C.COL_PRODUCT_ID, C.COL_LOCALE, C.COL_SPLIT)
    for qid, query, pid, locale, split in zip(*(pos[c] for c in cols)):
        pos_text = text_of.get((locale, pid), "")
        if not pos_text:
            continue
        cands = list(negs_of.get(qid, ()))
        if cands:
            # shuffle within tiers, keep I before C
            rng.shuffle(cands)
            cands.sort(key=lambda item: item[0])
        neg_ids: list[str] = []
        seen = {pos_text}
        for _, npid, nloc in cands:
            text = text_of.get((nloc, npid), "")
            if text and text not in seen:
                neg_ids.append(npid)
                seen.add(text)
            if len(neg_ids) == max_negatives:
                break
        rows.append((qid, query, pid, locale, split, neg_ids))

    df = pd.DataFrame(
        rows,
        columns=[
            C.COL_QUERY_ID,
            C.COL_QUERY,
            C.COL_PRODUCT_ID,
            C.COL_LOCALE,
            C.COL_SPLIT,
            "neg_product_ids",
        ],
    )
    with_negs = int(df["neg_product_ids"].map(len).gt(0).sum()) if len(df) else 0
    log.info(
        "query-product: %s rows (%s with hard negatives)",
        f"{len(df):,}",
        f"{with_negs:,}",
    )
    return df


def build_product_product(
    examples: pd.DataFrame,
    products: pd.DataFrame,
    cap_per_query: int = C.MAX_PP_PAIRS_PER_QUERY,
    seed: int = C.SEED,
) -> pd.DataFrame:
    rng = random.Random(seed)
    text_of = product_text_lookup(products)
    pos = examples[examples[C.COL_LABEL].isin(C.POSITIVE_LABELS)]

    by_query: dict = {}
    cols = (C.COL_QUERY_ID, C.COL_PRODUCT_ID, C.COL_LOCALE)
    for qid, pid, locale in zip(*(pos[c] for c in cols)):
        by_query.setdefault(qid, set()).add((pid, locale))

    seen: set = set()
    rows = []
    for qid, items in by_query.items():
        if len(items) < 2:
            continue
        pairs = list(combinations(sorted(items), 2))
        if len(pairs) > cap_per_query:
            pairs = rng.sample(pairs, cap_per_query)
        for (pid_a, loc_a), (pid_b, _), in pairs:
            key = (loc_a, *sorted((pid_a, pid_b)))
            if key in seen:
                continue
            seen.add(key)
            text_a = text_of.get((loc_a, pid_a), "")
            text_b = text_of.get((loc_a, pid_b), "")
            if not text_a or not text_b or text_a == text_b:
                continue
            rows.append((qid, loc_a, pid_a, pid_b))

    df = pd.DataFrame(
        rows,
        columns=[C.COL_QUERY_ID, C.COL_LOCALE, "product_id_a", "product_id_b"],
    )
    log.info("product-product: %s pairs", f"{len(df):,}")
    return df
