import argparse

import numpy as np
import polars as pl
import torch

from beatdetect import (
    ANNOTATIONS_PROCESSED_PATH,
    DATASETS,
    ONE_HOT_BEATS_PATH,
    SPECTROGRAMS_RAW_PATH,
)
from beatdetect.histogram_params import HOP_LENGTH, SAMPLE_RATE


def main(specified_dataset=None):
    print(f"Saving one hot-encoded annotations into {ONE_HOT_BEATS_PATH}")
    fps = SAMPLE_RATE / HOP_LENGTH
    count = 0
    for dataset in DATASETS if specified_dataset is None else [specified_dataset]:
        spectrograms = np.load(SPECTROGRAMS_RAW_PATH / dataset / f"{dataset}.npz")
        annotations_dir = ANNOTATIONS_PROCESSED_PATH / dataset / "annotations" / "beats"
        onehot_dir = ONE_HOT_BEATS_PATH / dataset
        onehot_dir.mkdir(parents=True, exist_ok=True)

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
            indices = np.round(beat_times * fps).astype(int)
            indices = np.clip(indices, 0, num_frames - 1)
            onehot = torch.zeros(num_frames)
            onehot[indices] = 1

            torch.save(onehot, onehot_dir / f"{name}.pt")
            count += 1

    print()
    print(f"Finished processing {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"One hot-encode beat annotations, and save them to {ONE_HOT_BEATS_PATH}."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to encode. If omitted, encodes all datasets.",
    )
    args = parser.parse_args()

    main(args.dataset)
