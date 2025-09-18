import numpy as np
import torch
from torch.utils.data import Dataset

from .config_loader import Config


class BeatDataset(Dataset):
    def __init__(self, config: Config, datasets: list[str] | None = None):
        self.datasets = datasets if datasets is not None else config.downloads.datasets
        self.paths = config.paths
        self.encoded_beats_path = config.paths.data.processed.encoded_beats
        self.spectral_flux_path = config.paths.data.processed.spectral_flux
        self.spectrograms_path = config.paths.data.raw.spectrograms

        # build an index of all (dataset, name) pairs
        self.samples = []
        self.spectrograms = {}

        for dataset in self.datasets:
            # load available track names for this dataset
            # using encoded beats files to determine names
            names = sorted(
                [
                    p.name.removesuffix(".pt")
                    for p in (self.encoded_beats_path / dataset).glob("*.pt")
                ]
            )
            # load spectrogram archive for this dataset once
            spectrogram = np.load(self.spectrograms_path / dataset / f"{dataset}.npz")
            self.spectrograms[dataset] = spectrogram

            for name in names:
                self.samples.append((dataset, name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # determine which dataset and track this index corresponds to
        dataset, name = self.samples[idx]

        spec_archive = self.spectrograms[dataset]
        mel = torch.from_numpy(spec_archive.get(f"{name}/track").T).to(torch.float32)

        flux = torch.load(self.spectral_flux_path / dataset / f"{name}.pt")
        target = torch.load(self.encoded_beats_path / dataset / f"{name}.pt")

        return mel, flux, target
