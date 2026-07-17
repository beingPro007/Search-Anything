"""Cleaning stage: normalize text, drop noisy rows, build product_text."""

import pandas as pd

from app.common import get_logger
from app.constants import esci as C
from app.utils import join_nonempty, normalize_series, truncate

log = get_logger(__name__)


def _log_drop(stage: str, before: int, after: int) -> None:
    log.info(
        "%s: dropped %s (%s -> %s)",
        stage,
        f"{before - after:,}",
        f"{before:,}",
        f"{after:,}",
    )


def build_product_text(products: pd.DataFrame) -> pd.Series:
    bullets = products[C.COL_BULLETS].map(lambda t: truncate(t, C.MAX_BULLET_CHARS))
    fields = zip(
        products[C.COL_TITLE], products[C.COL_BRAND], products[C.COL_COLOR], bullets
    )
    texts = [
        truncate(join_nonempty(list(parts), C.PRODUCT_TEXT_SEP), C.MAX_PRODUCT_CHARS)
        for parts in fields
    ]
    return pd.Series(texts, index=products.index)


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    for col in (C.COL_TITLE, C.COL_BULLETS, C.COL_BRAND, C.COL_COLOR):
        df[col] = normalize_series(df[col])

    n = len(df)
    df = df[df[C.COL_TITLE].str.len() >= C.MIN_TITLE_CHARS]
    _log_drop("products/short-title", n, len(df))

    n = len(df)
    df = df.drop_duplicates(subset=[C.COL_LOCALE, C.COL_PRODUCT_ID], keep="first")
    _log_drop("products/dup-id", n, len(df))

    df[C.COL_PRODUCT_TEXT] = build_product_text(df)
    keep = [C.COL_PRODUCT_ID, C.COL_LOCALE, C.COL_TITLE, C.COL_PRODUCT_TEXT]
    return df[keep].reset_index(drop=True)


def clean_examples(
    examples: pd.DataFrame, products_clean: pd.DataFrame
) -> pd.DataFrame:
    df = examples.copy()
    df[C.COL_QUERY] = normalize_series(df[C.COL_QUERY])

    n = len(df)
    min_len = df[C.COL_LOCALE].map(C.MIN_QUERY_CHARS).fillna(2)
    df = df[df[C.COL_QUERY].str.len() >= min_len]
    _log_drop("examples/short-query", n, len(df))

    n = len(df)
    df = df.drop_duplicates(
        subset=[C.COL_QUERY_ID, C.COL_PRODUCT_ID, C.COL_LOCALE], keep="first"
    )
    _log_drop("examples/dup-judgment", n, len(df))

    n = len(df)
    valid = pd.MultiIndex.from_frame(products_clean[[C.COL_LOCALE, C.COL_PRODUCT_ID]])
    rows = pd.MultiIndex.from_frame(df[[C.COL_LOCALE, C.COL_PRODUCT_ID]])
    df = df[rows.isin(valid)]
    _log_drop("examples/unknown-product", n, len(df))

    n = len(df)
    with_pos = df.loc[df[C.COL_LABEL].isin(C.POSITIVE_LABELS), C.COL_QUERY_ID]
    df = df[df[C.COL_QUERY_ID].isin(set(with_pos))]
    _log_drop("examples/no-positive", n, len(df))

    return df.reset_index(drop=True)
