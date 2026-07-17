from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from app.common.logger import get_logger

log = get_logger(__name__)

_CHUNK_BYTES = 1 << 20


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        log.info("cached: %s", dest)
        return dest

    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, disable=None
        ) as bar:
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
                    fh.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    log.info("wrote %s rows -> %s", f"{len(df):,}", path)
    return path


def load_parquet(
    path: Path,
    columns: list[str] | None = None,
    filters: list[tuple] | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns, filters=filters)
    log.info("read %s rows <- %s", f"{len(df):,}", path)
    return df
