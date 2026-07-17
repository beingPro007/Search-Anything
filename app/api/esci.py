"""Access layer for the raw ESCI dataset files (locale-filtered via parquet pushdown)."""

from pathlib import Path

import pandas as pd

from app.common import download_file, get_logger, load_parquet
from app.constants import esci as C

log = get_logger(__name__)


def raw_path(filename: str) -> Path:
    return C.RAW_DIR / filename


def download_raw(force: bool = False) -> dict[str, Path]:
    return {
        name: download_file(f"{C.ESCI_BASE_URL}/{name}", raw_path(name), force=force)
        for name in C.RAW_FILES
    }


def _ensure_raw(filename: str) -> Path:
    path = raw_path(filename)
    if not path.exists():
        download_file(f"{C.ESCI_BASE_URL}/{filename}", path)
    return path


def load_examples(locale: str, small: bool = False) -> pd.DataFrame:
    version_col = C.COL_SMALL_VERSION if small else C.COL_LARGE_VERSION
    df = load_parquet(
        _ensure_raw(C.EXAMPLES_FILE),
        columns=list(C.EXAMPLES_COLUMNS),
        filters=[(version_col, "==", 1), (C.COL_LOCALE, "==", locale)],
    )
    log.info(
        "examples[%s]: %s rows (version=%s)",
        locale,
        f"{len(df):,}",
        "small" if small else "large",
    )
    return df.reset_index(drop=True)


def load_products(locale: str) -> pd.DataFrame:
    df = load_parquet(
        _ensure_raw(C.PRODUCTS_FILE),
        columns=list(C.PRODUCTS_COLUMNS),
        filters=[(C.COL_LOCALE, "==", locale)],
    )
    log.info("products[%s]: %s rows", locale, f"{len(df):,}")
    return df.reset_index(drop=True)


def load_sources() -> pd.DataFrame:
    return pd.read_csv(_ensure_raw(C.SOURCES_FILE))
