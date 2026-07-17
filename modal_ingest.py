"""Product ingestion on Modal.

  modal run modal_ingest.py::smoke                 # encode 200 products -> in-memory qdrant -> search
  modal run --detach modal_ingest.py               # encode all shards (INGEST_GPUS containers)
  modal run modal_ingest.py --qdrant-url ... --qdrant-api-key ...   # + upsert to Qdrant Cloud
"""

import os

import modal

DATA_PATH = "/data"
GPU = os.environ.get("TRAIN_GPU", "L4")
INGEST_GPUS = int(os.environ.get("INGEST_GPUS", "3"))

app = modal.App("esci-ingest")
volume = modal.Volume.from_name("esci-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.44",
        "peft>=0.12",
        "sentencepiece",
        "protobuf",
        "pandas>=2.2",
        "pyarrow>=15",
        "requests>=2.31",
        "tqdm>=4.66",
        "qdrant-client>=1.10",
    )
    .env(
        {
            "ESCI_DATA_DIR": DATA_PATH,
            "HF_HOME": f"{DATA_PATH}/hf-cache",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_python_source("app")
)


@app.function(
    image=image,
    volumes={DATA_PATH: volume},
    gpu=GPU,
    memory=32768,
    cpu=8,
    timeout=60 * 60 * 8,
    max_containers=INGEST_GPUS,
)
def encode_shard_remote(locale: str, idx: int, limit: int | None = None):
    from app.ingest import encode_shard

    volume.reload()
    path = encode_shard(locale, idx, limit=limit)
    volume.commit()
    return str(path)


@app.function(
    image=image, volumes={DATA_PATH: volume}, memory=16384, cpu=4, timeout=60 * 60 * 8
)
def upsert_remote(
    locales: str = "us,es,jp",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
):
    from app.ingest import make_client, upsert_all

    volume.reload()
    wanted = tuple(loc.strip() for loc in locales.split(",") if loc.strip())
    client = make_client(qdrant_url, qdrant_api_key)
    return upsert_all(client, wanted)


@app.function(
    image=image, volumes={DATA_PATH: volume}, gpu=GPU, memory=32768, cpu=8, timeout=3600
)
def smoke(locale: str = "us", n_products: int = 200):
    import torch
    from transformers import AutoTokenizer

    from app.common import load_parquet
    from app.constants import esci as C
    from app.constants import splade as S
    from app.constants import train as T
    from app.databuilder import qp_path
    from app.ingest import (
        encode_shard,
        make_client,
        search_dense,
        search_sparse,
        shard_path,
        upsert_shard,
    )
    from app.models.gcn import load_gcn
    from app.models.splade import load_splade

    volume.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shard_path(locale, 0).unlink(missing_ok=True)
    path = encode_shard(locale, 0, limit=n_products, force=True)

    client = make_client()
    upsert_shard(client, path)

    gcn, head, gcn_cfg = load_gcn(T.MODELS_DIR, device)
    gcn_tok = AutoTokenizer.from_pretrained(gcn_cfg["base_model"])
    splade, splade_cfg = load_splade(S.SPLADE_DIR, device)
    splade_tok = AutoTokenizer.from_pretrained(splade_cfg["base_model"])

    queries = list(
        load_parquet(qp_path(C.SPLIT_TEST, locale), columns=[C.COL_QUERY])[
            C.COL_QUERY
        ].head(3)
    )
    report = {}
    for query in queries:
        dense_hits = search_dense(client, gcn, gcn_tok, query, device)
        sparse_hits = search_sparse(client, splade, splade_tok, query, device)
        report[query] = {
            "dense": [h.payload[C.COL_TITLE][:60] for h in dense_hits[:3]],
            "sparse": [h.payload[C.COL_TITLE][:60] for h in sparse_hits[:3]],
        }
    for q, hits in report.items():
        print(f"\nQUERY: {q}")
        print("  dense :", *hits["dense"], sep="\n    ")
        print("  sparse:", *hits["sparse"], sep="\n    ")
    return report


@app.local_entrypoint()
def main(
    locales: str = "us,es,jp",
    limit: int | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
):
    from app.constants import esci as C  # local import for the shard plan
    from app.ingest import shard_plan

    wanted = tuple(loc.strip() for loc in locales.split(",") if loc.strip())
    plan = shard_plan(wanted)
    print(f"{len(plan)} shards across {len(wanted)} locales, {INGEST_GPUS} GPUs")
    for path in encode_shard_remote.starmap([(loc, idx, limit) for loc, idx in plan]):
        print("done:", path)
    if qdrant_url:
        print(upsert_remote.remote(locales, qdrant_url, qdrant_api_key))
