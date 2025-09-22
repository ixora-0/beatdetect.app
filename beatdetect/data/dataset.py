import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config_loader import Config
from ..utils.paths import PathResolver


class BeatDataset(Dataset):
    def __init__(
        self,
        config: Config,
        split: str,
        datasets: list[str] | None = None,
    ):
        """
        Initialize BeatDataset.

        Args:
            config: Configuration object
            datasets: List of datasets to include
            split: Split to load ('train', 'val', 'test').
        """
        self.config = config
        self.datasets = datasets if datasets is not None else config.downloads.datasets
        self.spectrograms_path = config.paths.data.raw.spectrograms
        self.spectral_flux_path = config.paths.data.processed.spectral_flux

        self.splits_file = config.paths.data.processed.splits_info
        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {self.splits_file}. "
                "Please run the dataset splitting script first."
            )
        # Load only samples for this split from CSV
        dataset_set = set(self.datasets)
        self.samples = []
        with open(self.splits_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset"] in dataset_set and row["split"] == split:
                    self.samples.append((row["dataset"], row["name"]))

        print(f"Loaded {split} split: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # determine which dataset and track this index corresponds to
        dataset, name = self.samples[idx]
        paths = PathResolver(self.config, dataset)

        # Load spectrogram on-demand to avoid memory issues with large NPZ files
        with np.load(paths.spectrograms_file) as spec_archive:
            mel = torch.from_numpy(spec_archive.get(f"{name}/track").T).to(
                torch.float32
            )
        flux = torch.load(self.spectral_flux_path / dataset / f"{name}.pt")

        # Load beats and downbeats
        target = torch.load(paths.encoded_annotations_dir / f"{name}.pt")

        return mel, flux, target
