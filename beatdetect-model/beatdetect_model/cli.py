from pathlib import Path

import typer

from . import config_loader

app = typer.Typer(help="Beat detection pipeline CLI", pretty_exceptions_enable=False)


@app.command()
def download(config: str = "../configs/dev.toml"):
    """Download raw data (annotations, spectrograms)."""
    from .scripts import download_annotations, download_spectrograms

    cfg = config_loader.load_config(config)
    download_annotations.main(cfg)
    download_spectrograms.main(cfg)


@app.command()
def preprocess(config: str = "../configs/dev.toml"):
    """Clean and process annotations and features."""
    from .scripts import (
        combine_dataset_info,
        create_spectral_flux,
        encode_annotations,
        normalize_annotations,
    )

    cfg = config_loader.load_config(config)
    normalize_annotations.main(cfg)
    combine_dataset_info.main(cfg)
    encode_annotations.main(cfg)
    create_spectral_flux.main(cfg)


@app.command()
def split(config: str = "../configs/dev.toml"):
    """Split processed data into train/validation/test sets and save the splits."""
    from .scripts import make_splits

    cfg = config_loader.load_config(config)
    make_splits.main(cfg)


@app.command()
def prepare_data(config: str = "../configs/dev.toml"):
    """Download and preprocess data"""
    download(config)
    preprocess(config)
    split(config)


@app.command()
def train(
    config: str = "../configs/dev.toml",
    no_log: bool = typer.Option(
        False,
        "--no-log",
        help="Disable logging",
    ),
):
    """Run training with given config."""
    from .scripts import train

    cfg = config_loader.load_config(config)
    train.main(cfg, log=not no_log)


@app.command()
def create_transition_matrix(config: str = "../configs/dev.toml"):
    """Precompute state transition matrix used in postprocessing."""
    from .model.postprocessing import create_transitions

    cfg = config_loader.load_config(config)
    create_transitions.main(cfg)


@app.command()
def create_init_dist(config: str = "../configs/dev.toml"):
    """Initial log probability distributions from data for postprocessing"""
    from .model.postprocessing import create_init_dist

    cfg = config_loader.load_config(config)
    create_init_dist.main(cfg)


@app.command()
def run_all(config: str = "../configs/dev.toml"):
    """Run the full pipeline end-to-end."""
    prepare_data(config)
    train(config)


_BEAT_MODEL_PATH = typer.Argument(
    None,
    help="Path to beat detect tcn model, if omitted defaults to latest in models/",
)


@app.command()
def save_onnx(
    config: str = "../configs/dev.toml", beat_model_path: str = _BEAT_MODEL_PATH
):
    from .scripts import save_onnx

    cfg = config_loader.load_config(config)
    if beat_model_path is not None:
        beat_model_path = Path(beat_model_path)
    save_onnx.main(cfg, beat_model_path)


_ARTIFACTS_DIR = typer.Argument(
    help="Path to directory containing model artifacts."
    "Directory name is used as the version identifier.",
)


@app.command()
def upload_artifacts(artifacts_dir: str = _ARTIFACTS_DIR):
    from .scripts import upload_artifacts

    upload_artifacts.main(Path(artifacts_dir))


if __name__ == "__main__":
    app()
