import json
from pathlib import Path

from ..config_loader import Config


class PathResolver:
    def __init__(self, config: Config, dataset: str):
        self.paths = config.paths
        self.dataset = dataset

    def resolve_annotations_dir(self, cleaned=True) -> Path:
        p = (
            self.paths.data.interim.cleaned_annotations
            if cleaned
            else self.paths.data.raw.annotations
        )
        return p / self.dataset / "annotations" / "beats"

    @property
    def spectrograms_file(self) -> Path:
        return self.paths.data.raw.spectrograms / self.dataset / f"{self.dataset}.npz"

    @property
    def encoded_beats_dir(self) -> Path:
        return self.paths.data.processed.encoded_beats / self.dataset

    @property
    def spectral_flux_dir(self) -> Path:
        return self.paths.data.processed.spectral_flux / self.dataset


def iterate_beat_files(config: Config, processed=True, has_downbeats=True):
    path = (
        config.paths.data.processed.annotations
        if processed
        else config.paths.data.raw.annotations
    )
    if not path.exists():
        raise FileNotFoundError(f"{path} doesn't exist.")

    if processed:
        with (path / "info.json").open("r") as f:
            info = json.load(f)
    else:
        info = {}
        for dataset in config.downloads.datasets:
            info_file_path = path / dataset / "info.json"
            with info_file_path.open("r") as f:
                info[dataset] = json.load(f)

    for dataset in config.downloads.datasets:
        if has_downbeats and not info[dataset]["has_downbeats"]:
            continue

        dataset_path = path / dataset
        for beats_file in dataset_path.rglob("*.beats"):
            with open(beats_file) as f:
                if not f.read(1):
                    print(f"\033[K{beats_file} is empty, skipping.")
                    continue
            yield beats_file
