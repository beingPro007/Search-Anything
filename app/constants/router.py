"""Hyperparameters for the query router (GCN dense tower vs SPLADE sparse tower)."""

from app.constants.esci import DATA_DIR

HIDDEN = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 5
BATCH_SIZE = 256

# |ndcg_gcn - ndcg_splade| below this is a tie -> dropped from training
TIE_EPSILON = 0.02
TRAIN_QUERIES_PER_LOCALE = 2000
TEST_QUERIES_PER_LOCALE = 400

ROUTER_DIR = DATA_DIR / "models" / "router"
ROUTER_HEAD_FILE = "router_head.pt"
LABELS_FILE = "labels_{split}_{locale}.parquet"
