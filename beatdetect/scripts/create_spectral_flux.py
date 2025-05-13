import argparse

import librosa
import numpy as np
import torch
import torchaudio

from beatdetect import (
    DATASETS,
    SPECTRAL_FLUX_PATH,
    SPECTROGRAMS_RAW_PATH,
)
from beatdetect.histogram_params import HOP_LENGTH, SAMPLE_RATE


def main(specified_dataset=None):
    print(f"Saving spectral flux files into {SPECTRAL_FLUX_PATH}")
    count = 0
    for dataset in DATASETS if specified_dataset is None else [specified_dataset]:
        spectrograms = np.load(SPECTROGRAMS_RAW_PATH / dataset / f"{dataset}.npz")
        spectral_flux_dir = SPECTRAL_FLUX_PATH / dataset
        spectral_flux_dir.mkdir(parents=True, exist_ok=True)
        for file_name in spectrograms:
            if not file_name.endswith("/track"):
                continue
            print(f"\033[KProcessing file {file_name}", end="\r")
            name = file_name.removesuffix("/track")

            melspect = torch.from_numpy(spectrograms[file_name]).to(torch.float32)
            # ibrosa.onset.onset_strength requires log-power spectrogram
            log_melspect = torchaudio.transforms.AmplitudeToDB(stype="power")(melspect)
            spectral_flux = librosa.onset.onset_strength(
                S=log_melspect.T,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                lag=2,
                max_size=3,
            )
            spectral_flux = np.clip(spectral_flux, None, 4)  # clip values above 4

            torch.save(
                torch.from_numpy(spectral_flux), spectral_flux_dir / f"{name}.pt"
            )
            count += 1

    print()
    print(f"Finished processing {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Generate spectral flux from spectrograms, and save them to {SPECTRAL_FLUX_PATH}."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset. If omitted, create spectral flux for all datasets.",
    )
    args = parser.parse_args()

    main(args.dataset)
