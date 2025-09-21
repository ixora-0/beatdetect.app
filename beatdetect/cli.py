import typer

from beatdetect import config_loader

app = typer.Typer(help="Beat detection pipeline CLI")


@app.command()
def download(config: str = "configs/dev.toml"):
    """Download raw data (annotations, spectrograms)."""
    from .scripts import download_annotations, download_spectrograms

    cfg = config_loader.load_config(config)
    download_annotations.main(cfg)
    download_spectrograms.main(cfg)


@app.command()
def preprocess(config: str = "configs/dev.toml"):
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
def split(config: str = "configs/dev.toml"):
    """Split processed data into train/validation/test sets and save the splits."""
    from .scripts import make_splits

    cfg = config_loader.load_config(config)
    make_splits.main(cfg)


@app.command()
def prepare_data(config: str = "configs/dev.toml"):
    """Download and preprocess data"""
    download(config)
    preprocess(config)
    split(config)


@app.command()
def train(config: str = "configs/dev.toml"):
    """Run training with given config."""
    from .scripts import train

    cfg = config_loader.load_config(config)
    train.main(cfg)


@app.command()
def run_all(config: str = "configs/dev.toml"):
    """Run the full pipeline end-to-end."""
    prepare_data(config)
    train(config)


if __name__ == "__main__":
    app()
