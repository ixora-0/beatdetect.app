import csv
import random

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
        use_pitch_shift: bool = True,
        pitch_shift_prob: float = 0.5,
    ):
        """
        Initialize BeatDataset.

        Args:
            config: Configuration object
            datasets: List of datasets to include
            split: Split to load ('train', 'val', 'test').
            use_pitch_shift: Whether to use pitch shift augmentation
            pitch_shift_prob: Probability of using pitch shifted version
        """
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.datasets = datasets if datasets is not None else config.downloads.datasets
        self.spectrograms_path = config.paths.data.raw.spectrograms
        self.spectral_flux_path = config.paths.data.processed.spectral_flux

        self.splits_file = config.paths.data.processed.splits_info
        self.use_pitch_shift = use_pitch_shift and split == "train"  # Only for training
        self.pitch_shift_prob = pitch_shift_prob
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

        self.spec_archives = {}
        for dataset in config.downloads.datasets:
            paths = PathResolver(self.config, dataset)
            self.spec_archives[dataset] = np.load(paths.spectrograms_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # determine which dataset and track this index corresponds to
        dataset, name = self.samples[idx]
        paths = PathResolver(self.config, dataset)

        flux = torch.load(self.spectral_flux_path / dataset / f"{name}.pt")
        # Choose pitch shift variant
        if (
            self.use_pitch_shift
            and self.rng.random() < self.pitch_shift_prob
            and dataset != "gtzan"
        ):
            shift = self.rng.randint(-5, 6)
            if shift == 0:
                key = f"{name}/track"
            else:
                key = f"{name}/track_ps{shift}"
        else:
            key = f"{name}/track"
        mel = torch.from_numpy(self.spec_archives[dataset].get(key).T).to(torch.float32)

        # Load beats and downbeats
        target = torch.load(paths.encoded_annotations_dir / f"{name}.pt")

        return f"{dataset}/{name}", mel, flux, target
