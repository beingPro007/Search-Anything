"""Router training on Modal: `modal run modal_router.py [--force-labels]`."""

import os

import modal

DATA_PATH = "/data"
GPU = os.environ.get("TRAIN_GPU", "L4")

app = modal.App("esci-router-train")
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
    timeout=60 * 60 * 6,
)
def train_remote(
    locales: str = "us,es,jp",
    train_queries: int = 2000,
    test_queries: int = 400,
    force_labels: bool = False,
):
    from app.train_router import train

    volume.reload()
    wanted = tuple(loc.strip() for loc in locales.split(",") if loc.strip())
    metrics = train(
        locales=wanted,
        train_queries=train_queries,
        test_queries=test_queries,
        force_labels=force_labels,
        on_checkpoint=volume.commit,
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    locales: str = "us,es,jp",
    train_queries: int = 2000,
    test_queries: int = 400,
    force_labels: bool = False,
):
    print(
        train_remote.remote(
            locales=locales,
            train_queries=train_queries,
            test_queries=test_queries,
            force_labels=force_labels,
        )
    )
