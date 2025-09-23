import argparse
import textwrap

import librosa
import numpy as np
import torch
import torchaudio

from beatdetect.utils.paths import PathResolver

from ..config_loader import Config, load_config


def main(config: Config, specified_dataset=None):
    print(
        f"Saving spectral flux files into {config.paths.data.processed.spectral_flux}"
    )
    datasets = [specified_dataset] if specified_dataset else config.downloads.datasets
    count = 0
    for dataset in datasets:
        paths = PathResolver(config, dataset)
        spectrograms = np.load(paths.spectrograms_file)
        spectral_flux_dir = config.paths.data.processed.spectral_flux / dataset
        spectral_flux_dir.mkdir(parents=True, exist_ok=True)

        for file_name in spectrograms:
            if not file_name.endswith("/track"):
                continue
            print(f"\033[KProcessing file {file_name}", end="\r")
            name = file_name.removesuffix("/track")
            output_path = spectral_flux_dir / f"{name}.pt"
            if output_path.exists():
                continue

            melspect = torch.from_numpy(spectrograms[file_name]).to(torch.float32)
            # ibrosa.onset.onset_strength requires log-power spectrogram
            log_melspect = torchaudio.transforms.AmplitudeToDB(stype="power")(melspect)
            spectral_flux = librosa.onset.onset_strength(
                S=log_melspect.T,
                sr=config.spectrogram.sample_rate,
                hop_length=config.spectrogram.hop_length,
                lag=2,
                max_size=3,
            )
            spectral_flux = np.clip(spectral_flux, None, 4)  # clip values above 4

            torch.save(torch.from_numpy(spectral_flux), output_path)
            count += 1

    print()
    print(f"Finished processing {count} files")


if __name__ == "__main__":
    config = load_config()
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f"""
            Generate spectral flux from spectrograms,
            and save them to {config.paths.data.processed.spectral_flux}.
        """)
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset. If omitted, create spectral flux for all datasets.",
    )
    args = parser.parse_args()

    main(config, args.dataset)
