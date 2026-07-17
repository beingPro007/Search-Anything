"""Constants for the Amazon ESCI dataset build (github.com/amazon-science/esci-data)."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ESCI_DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ESCI_BASE_URL = (
    "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"
)
EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"
SOURCES_FILE = "shopping_queries_dataset_sources.csv"
RAW_FILES = (EXAMPLES_FILE, PRODUCTS_FILE, SOURCES_FILE)

COL_QUERY = "query"
COL_QUERY_ID = "query_id"
COL_PRODUCT_ID = "product_id"
COL_LOCALE = "product_locale"
COL_LABEL = "esci_label"
COL_SPLIT = "split"
COL_SMALL_VERSION = "small_version"
COL_LARGE_VERSION = "large_version"

COL_TITLE = "product_title"
COL_DESCRIPTION = "product_description"
COL_BULLETS = "product_bullet_point"
COL_BRAND = "product_brand"
COL_COLOR = "product_color"
COL_PRODUCT_TEXT = "product_text"

EXAMPLES_COLUMNS = (
    COL_QUERY_ID,
    COL_QUERY,
    COL_PRODUCT_ID,
    COL_LOCALE,
    COL_LABEL,
    COL_SPLIT,
)
PRODUCTS_COLUMNS = (
    COL_PRODUCT_ID,
    COL_LOCALE,
    COL_TITLE,
    COL_BULLETS,
    COL_BRAND,
    COL_COLOR,
)

LABEL_EXACT = "E"
LABEL_SUBSTITUTE = "S"
LABEL_COMPLEMENT = "C"
LABEL_IRRELEVANT = "I"

POSITIVE_LABELS = frozenset({LABEL_EXACT})
# S is excluded: substitutes are near-positives and act as false negatives in InfoNCE.
NEGATIVE_LABEL_PRIORITY = (LABEL_IRRELEVANT, LABEL_COMPLEMENT)

LOCALES = ("us", "es", "jp")
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"

MIN_QUERY_CHARS = {"us": 2, "es": 2, "jp": 1}
MIN_TITLE_CHARS = 3
MAX_BULLET_CHARS = 500
MAX_PRODUCT_CHARS = 1500
PRODUCT_TEXT_SEP = " | "

MAX_NEGATIVES_PER_ROW = 4
MAX_PP_PAIRS_PER_QUERY = 5
MAX_NEIGHBOR_QUERIES = 5
MAX_Q2Q_PER_PRODUCT = 5
SEED = 42

OUT_PRODUCTS = "products_clean_{locale}.parquet"
OUT_EXAMPLES_CLEAN = "examples_clean_{locale}.parquet"
OUT_QUERY_PRODUCT = "query_product_{split}_{locale}.parquet"
OUT_PRODUCT_PRODUCT = "product_product_{split}_{locale}.parquet"
OUT_QUERIES = "queries_{locale}.parquet"
OUT_GRAPH_P2Q = "graph_p2q_{locale}.parquet"
OUT_GRAPH_Q2Q = "graph_q2q_{locale}.parquet"
