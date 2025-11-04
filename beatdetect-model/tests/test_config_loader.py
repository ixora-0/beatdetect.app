from pathlib import Path

from beatdetect_model.config_loader import Config, load_config


def test_load_config_minimal():
    config = load_config()

    assert isinstance(config, Config)

    assert isinstance(config.paths.models, Path)
    assert isinstance(config.paths.data.raw.annotations, Path)
