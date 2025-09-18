import argparse
import textwrap

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from ..config_loader import Config, load_config
from ..utils.paths import PathResolver


def main(config: Config, specified_dataset: str | None = None):
    print(
        "Saving smoothed one hot-encoded annotations into"
        f"{config.paths.data.processed.encoded_beats}"
    )

    count = 0
    datasets = [specified_dataset] if specified_dataset else config.downloads.datasets
    for dataset in datasets:
        paths = PathResolver(config, dataset)
        annotations_dir = paths.resolve_annotations_dir()

        paths.encoded_beats_dir.mkdir(parents=True, exist_ok=True)

        spectrograms = np.load(paths.spectrograms_file)

        for path in annotations_dir.glob("*.beats"):
            print(f"\033[KProcessing file {path}", end="\r")
            name = path.name.removesuffix(".beats")
            spectrogram = spectrograms.get(f"{name}/track")
            num_frames = spectrogram.shape[0]

            beat_df = pl.read_csv(
                path,
                separator="\t",
                has_header=False,
            )
            beat_times = beat_df.to_numpy()[:, 0]

            # one hot encode
            indices = np.round(beat_times * config.spectrogram.fps).astype(int)
            indices = np.clip(indices, 0, num_frames - 1)
            onehot = torch.zeros(num_frames)
            onehot[indices] = 1

            # smooth using gaussian window
            kernel_size = 11  # should depend on period, but keep it simple for now
            sigma = 1

            half = (kernel_size - 1) // 2
            t = torch.arange(-half, half + 1, dtype=torch.float32)
            gauss = torch.exp(-0.5 * (t / sigma).pow(2))
            gauss = gauss / gauss.sum()  # normalize to sum=1

            # F.pad works on 1-D just fine when you supply a 2-tuple
            padded = F.pad(onehot, (half, half), mode="constant", value=0)
            windows = padded.unfold(0, kernel_size, 1)  # (T, kernel_size)

            # windows: (T, K), gauss: (K,) → result: (T,)
            smoothed = (windows * gauss).sum(dim=1)

            # renormalize
            center_weight = gauss[half]
            smoothed = smoothed / center_weight

            torch.save(smoothed, paths.encoded_beats_dir / f"{name}.pt")
            count += 1

    print()
    print(f"Finished processing {count} files")


if __name__ == "__main__":
    config = load_config()
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f"""
            One hot-encode beat annotations, smoothen it,
            and save them to {config.paths.data.processed.encoded_beats}.
        """)
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to encode. If omitted, encodes all datasets.",
    )
    args = parser.parse_args()

    main(config, args.dataset)
