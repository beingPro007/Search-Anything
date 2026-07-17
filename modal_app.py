"""Run the ESCI data build on Modal: `modal run modal_app.py [--small] [--locales us,es,jp]`."""

import modal

DATA_PATH = "/data"

app = modal.App("esci-data-build")
volume = modal.Volume.from_name("esci-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas>=2.2", "pyarrow>=15", "requests>=2.31", "tqdm>=4.66")
    .env({"ESCI_DATA_DIR": DATA_PATH})
    .add_local_python_source("app")
)


@app.function(image=image, volumes={DATA_PATH: volume}, memory=4096, cpu=2, timeout=3600)
def download():
    from app.api import download_raw

    download_raw()
    volume.commit()


@app.function(
    image=image, volumes={DATA_PATH: volume}, memory=16384, cpu=4, timeout=7200
)
def build_locale(locale: str, small: bool = False, force: bool = False):
    from app.databuilder import build

    volume.reload()
    build(locales=(locale,), small=small, force=force)
    volume.commit()


@app.function(image=image, volumes={DATA_PATH: volume}, memory=1024, timeout=600)
def show_status(small: bool = False):
    from app.databuilder import status

    volume.reload()
    status(small=small)


@app.local_entrypoint()
def main(
    locales: str = "us,es,jp",
    small: bool = False,
    force: bool = False,
    status_only: bool = False,
):
    if status_only:
        show_status.remote(small=small)
        return
    download.remote()
    wanted = [loc.strip() for loc in locales.split(",") if loc.strip()]
    for _ in build_locale.map(wanted, kwargs={"small": small, "force": force}):
        pass
    show_status.remote(small=small)
