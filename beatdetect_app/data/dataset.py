import csv
import json
import random

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from ..config_loader import Config
from ..utils.paths import PathResolver


class BeatDataset(Dataset):
    def __init__(
        self,
        config: Config,
        split: str,
        device: torch.device,
        require_downbeat: bool = False,
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
            require_downbeat: If True, skip samples that don't have downbeat annotations
            use_pitch_shift: Whether to use pitch shift augmentation
            pitch_shift_prob: Probability of using pitch shifted version
        """
        self.config = config
        self.device = device
        self.rng = random.Random(config.random_seed)
        self.datasets = datasets if datasets is not None else config.downloads.datasets
        self.spectrograms_path = config.paths.data.raw.spectrograms
        self.spectral_flux_path = config.paths.data.processed.spectral_flux

        self.splits_file = config.paths.data.processed.splits_info
        self.use_pitch_shift = use_pitch_shift and split == "train"  # Only for training
        self.pitch_shift_prob = pitch_shift_prob
        self.require_downbeat = require_downbeat

        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {self.splits_file}. "
                "Please run the dataset splitting script first."
            )

        with config.paths.data.processed.datasets_info.open("r", encoding="utf-8") as f:
            info = json.load(f)
        self.has_downbeat = {
            dataset: info[dataset]["has_downbeats"] for dataset in info.keys()
        }

        # Load only samples for this split from CSV
        dataset_set = set(self.datasets)
        self.samples = []
        with open(self.splits_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset"] in dataset_set and row["split"] == split:
                    if require_downbeat and not self.has_downbeat.get(
                        row["dataset"], False
                    ):
                        continue
                    self.samples.append((row["dataset"], row["name"]))

        print(f"Loaded {split} split: {len(self.samples)} samples")

        self.spec_archives = {}
        for dataset in self.datasets:
            paths = PathResolver(self.config, dataset)
            self.spec_archives[dataset] = np.load(paths.spectrograms_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # determine which dataset and track this index corresponds to
        dataset, name = self.samples[idx]
        paths = PathResolver(self.config, dataset)

        flux = torch.load(
            self.spectral_flux_path / dataset / f"{name}.pt", map_location=self.device
        )
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
        mel = torch.from_numpy(self.spec_archives[dataset].get(key).T).to(
            device=self.device, dtype=torch.float32
        )

        # Load beats and downbeats
        target = torch.load(
            paths.encoded_annotations_dir / f"{name}.pt", map_location=self.device
        )

        has_downbeat = self.has_downbeat[dataset]

        return f"{dataset}/{name}", mel, flux, target, has_downbeat

    def get_annotation(self, id: str) -> np.ndarray:
        dataset, name = id.split("/")
        paths = PathResolver(self.config, dataset)
        annotation_file_name = name + ".beats"
        beat_df = pl.read_csv(
            paths.resolve_annotations_dir(cleaned=True) / annotation_file_name,
            separator="\t",
            has_header=False,
        )
        return beat_df.to_numpy()
