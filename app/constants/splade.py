"""Hyperparameters for SPLADE training (qdrant.tech sparse-embeddings-ecommerce series)."""

from app.constants.esci import DATA_DIR

BASE_MODEL = "FacebookAI/xlm-roberta-base"

MAX_QUERY_TOKENS = 32
MAX_PRODUCT_TOKENS = 128
MAX_HARD_NEGATIVES = 2

# FLOPS regularizers; doc weight lower so products keep more attribute terms
LAMBDA_QUERY = 5e-5
LAMBDA_DOC = 3e-5
LAMBDA_RAMP_FRAC = 0.33
TEMPERATURE = 1.0

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
EPOCHS = 1
WARMUP_FRAC = 0.1
MAX_GRAD_NORM = 1.0
LOCALE_SMOOTHING = 0.7
LOG_EVERY = 50
CHECKPOINT_EVERY = 2000

EVAL_QUERIES_PER_LOCALE = 300
SPLADE_DIR = DATA_DIR / "models" / "splade"
