"""Constants for product ingestion (encode -> vector shards -> Qdrant)."""

from app.constants.esci import DATA_DIR

DENSE_BATCH = 64
SPARSE_BATCH = 32
SHARD_SIZE = 50_000
# cap sparse terms per product so ingestion stays sane even for dense-ish checkpoints
SPARSE_TOP_K = 512

VECTORS_DIR = DATA_DIR / "vectors"
SHARD_FILE = "vectors_{locale}_{idx:04d}.parquet"

COLLECTION = "products"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "splade"
UPSERT_BATCH = 256
