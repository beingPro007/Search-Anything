"""SPLADE training on Modal: `modal run modal_splade.py [--max-steps N]` or deploy + spawn."""

import os

import modal

DATA_PATH = "/data"
GPU = os.environ.get("TRAIN_GPU", "L4")

app = modal.App("esci-splade-train")
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
    timeout=60 * 60 * 20,
)
def train_remote(
    locales: str = "us,es,jp",
    epochs: int = 1,
    batch_size: int = 8,
    max_steps: int | None = None,
    limit: int | None = None,
    max_negs: int = 2,
    eval_queries: int = 200,
    require_negs: bool = False,
    max_per_query: int | None = None,
):
    from app.train_splade import train

    volume.reload()
    wanted = tuple(loc.strip() for loc in locales.split(",") if loc.strip())
    metrics = train(
        locales=wanted,
        epochs=epochs,
        batch_size=batch_size,
        max_steps=max_steps,
        limit=limit,
        max_negs=max_negs,
        eval_queries=eval_queries,
        require_negs=require_negs,
        max_per_query=max_per_query,
        on_checkpoint=volume.commit,
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    locales: str = "us,es,jp",
    epochs: int = 1,
    batch_size: int = 8,
    max_steps: int | None = None,
    limit: int | None = None,
    max_negs: int = 2,
    eval_queries: int = 200,
    require_negs: bool = False,
    max_per_query: int | None = None,
):
    metrics = train_remote.remote(
        locales=locales,
        epochs=epochs,
        batch_size=batch_size,
        max_steps=max_steps,
        limit=limit,
        max_negs=max_negs,
        eval_queries=eval_queries,
        require_negs=require_negs,
        max_per_query=max_per_query,
    )
    print(metrics)
