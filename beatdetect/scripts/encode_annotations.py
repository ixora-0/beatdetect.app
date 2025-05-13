import argparse

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from beatdetect import (
    ANNOTATIONS_PROCESSED_PATH,
    DATASETS,
    ENCODED_BEATS_PATH,
    SPECTROGRAMS_RAW_PATH,
)
from beatdetect.histogram_params import HOP_LENGTH, SAMPLE_RATE


def main(specified_dataset=None):
    print(f"Saving smoothed one hot-encoded annotations into {ENCODED_BEATS_PATH}")
    fps = SAMPLE_RATE / HOP_LENGTH
    count = 0
    for dataset in DATASETS if specified_dataset is None else [specified_dataset]:
        spectrograms = np.load(SPECTROGRAMS_RAW_PATH / dataset / f"{dataset}.npz")
        annotations_dir = ANNOTATIONS_PROCESSED_PATH / dataset / "annotations" / "beats"
        encoded_dir = ENCODED_BEATS_PATH / dataset
        encoded_dir.mkdir(parents=True, exist_ok=True)

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
            indices = np.round(beat_times * fps).astype(int)
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

            torch.save(smoothed, encoded_dir / f"{name}.pt")
            count += 1

    print()
    print(f"Finished processing {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"One hot-encode beat annotations, smoothen it, and save them to {ENCODED_BEATS_PATH}."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to encode. If omitted, encodes all datasets.",
    )
    args = parser.parse_args()

    main(args.dataset)
