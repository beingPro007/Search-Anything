"""Language-agnostic text normalization (safe for us/es/jp locales)."""

import html
import re
import unicodedata

import pandas as pd

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_text)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    # CJK text may have no spaces; keep the hard cut then.
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip()


def join_nonempty(parts: list[str], sep: str) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        if part and part not in seen:
            kept.append(part)
            seen.add(part)
    return sep.join(kept)
