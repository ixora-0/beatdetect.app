import numpy as np
import torch
from torch.utils.data import Dataset

from beatdetect import (
    ANNOTATIONS_PROCESSED_PATH,
    ENCODED_BEATS_PATH,
    SPECTRAL_FLUX_PATH,
    SPECTROGRAMS_RAW_PATH,
)


class BeatDataset(Dataset):
    def __init__(self, datasets: list[str]):
        self.datasets = datasets

        # build an index of all (dataset, name) pairs
        self.samples = []
        self.spectrograms = {}

        for dataset in self.datasets:
            # load available track names for this dataset
            # using encoded beats files to determine names
            names = sorted(
                [
                    p.name.removesuffix(".pt")
                    for p in (ENCODED_BEATS_PATH / dataset).glob("*.pt")
                ]
            )
            # load spectrogram archive for this dataset once
            spectrogram = np.load(SPECTROGRAMS_RAW_PATH / dataset / f"{dataset}.npz")
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

        flux = torch.load(SPECTRAL_FLUX_PATH / dataset / f"{name}.pt")
        target = torch.load(ENCODED_BEATS_PATH / dataset / f"{name}.pt")

        return mel, flux, target
