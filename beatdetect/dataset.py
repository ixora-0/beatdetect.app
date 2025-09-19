import numpy as np
import torch
from torch.utils.data import Dataset

from .config_loader import Config
from .utils.paths import PathResolver


class BeatDataset(Dataset):
    def __init__(self, config: Config, datasets: list[str] | None = None):
        self.config = config
        self.datasets = datasets if datasets is not None else config.downloads.datasets
        self.spectrograms_path = config.paths.data.raw.spectrograms
        self.spectral_flux_path = config.paths.data.processed.spectral_flux

        # build an index of all (dataset, name) pairs
        self.samples = []
        self.spectrograms = {}

        for dataset in self.datasets:
            paths = PathResolver(config, dataset)

            # load available track names for this dataset
            # using encoded beats files to determine names
            names = sorted(
                [
                    p.name.removesuffix(".pt")
                    for p in paths.encoded_beats_dir.glob("*.pt")
                ]
            )

            # load spectrogram archive for this dataset once
            self.spectrograms[dataset] = np.load(paths.spectrograms_file)

            for name in names:
                beats_file = paths.encoded_beats_dir / f"{name}.pt"
                downbeats_file = paths.encoded_downbeats_dir / f"{name}.pt"

                if beats_file.exists() and downbeats_file.exists():
                    self.samples.append((dataset, name))
                else:
                    print(f"Warning: Missing annotations for {dataset}/{name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # determine which dataset and track this index corresponds to
        dataset, name = self.samples[idx]

        paths = PathResolver(self.config, dataset)

        spec_archive = self.spectrograms[dataset]
        mel = torch.from_numpy(spec_archive.get(f"{name}/track").T).to(torch.float32)

        flux = torch.load(self.spectral_flux_path / dataset / f"{name}.pt")

        # Load beats and downbeats
        beats = torch.load(paths.encoded_beats_dir / f"{name}.pt")
        downbeats = torch.load(paths.encoded_downbeats_dir / f"{name}.pt")

        # Stack beats and downbeats: [0, :] = beats, [1, :] = downbeats
        target = torch.stack([beats, downbeats], dim=0)

        return mel, flux, target
