import argparse
import json
import textwrap
from collections import defaultdict

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from ..config_loader import Config, load_config
from ..utils.paths import PathResolver


def main(config: Config, specified_dataset: str | None = None):
    print("Saving smoothed one hot-encoded annotations")

    # Load dataset info
    info_path = config.paths.data.interim.annotations / "info.json"
    with info_path.open("r") as f:
        dataset_info = json.load(f)

    count, skipped = 0, defaultdict(list)
    datasets = [specified_dataset] if specified_dataset else config.downloads.datasets
    for dataset in datasets:
        has_downbeats = dataset_info.get(dataset, {}).get("has_downbeats", False)
        paths = PathResolver(config, dataset)
        annotations_dir = paths.resolve_annotations_dir(cleaned=True)

        paths.encoded_beats_dir.mkdir(parents=True, exist_ok=True)
        if has_downbeats:
            paths.encoded_downbeats_dir.mkdir(parents=True, exist_ok=True)

        spectrograms = np.load(paths.spectrograms_file)

        for path in annotations_dir.glob("*.beats"):
            print(f"\033[KProcessing file {path}", end="\r")
            name = path.stem
            spectrogram = spectrograms.get(f"{name}/track")
            num_frames = spectrogram.shape[0]

            beat_df = pl.read_csv(path, separator="\t", has_header=False)
            if has_downbeats and beat_df.shape[1] != 2:
                print(
                    f"Skipping {path} because it doesn't have beat indices"
                    " (supposed to be there)"
                )
                skipped[dataset].append(name)
                continue

            beat_times = beat_df[:, 0].to_numpy()
            beat_indices = beat_df[:, 1].to_numpy().astype(int)

            frame_indices = np.round(beat_times * config.spectrogram.fps).astype(int)
            frame_indices = np.clip(frame_indices, 0, num_frames - 1)

            # Beats: mark all beat frames
            onehot_beats = torch.zeros(num_frames)
            onehot_beats[frame_indices] = 1

            # Downbeats: mark only those with beat index == 1
            if has_downbeats:
                downbeat_mask = beat_indices == 1
                onehot_downbeats = torch.zeros(num_frames)
                onehot_downbeats[frame_indices[downbeat_mask]] = 1

            # smooth using gaussian window
            def smooth(onehot: torch.Tensor) -> torch.Tensor:
                kernel_size = 11  # should depend on period, but keep it simple for now
                sigma = 1
                half = (kernel_size - 1) // 2
                t = torch.arange(-half, half + 1, dtype=torch.float32)
                gauss = torch.exp(-0.5 * (t / sigma).pow(2))
                gauss /= gauss.sum()  # normalize to sum=1
                # F.pad works on 1-D just fine when you supply a 2-tuple
                padded = F.pad(onehot, (half, half), mode="constant", value=0)
                windows = padded.unfold(0, kernel_size, 1)
                # windows: (T, K), gauss: (K,) → result: (T,)
                smoothed = (windows * gauss).sum(dim=1)
                # renormalize
                center_weight = gauss[half]
                return smoothed / center_weight

            smoothed_beats = smooth(onehot_beats)
            torch.save(smoothed_beats, paths.encoded_beats_dir / f"{name}.pt")

            if has_downbeats:
                smoothed_downbeats = smooth(onehot_downbeats)
                torch.save(
                    smoothed_downbeats, paths.encoded_downbeats_dir / f"{name}.pt"
                )
            count += 1

    skipped_path = config.paths.data.processed.encoded_beats / "skipped.json"
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    with skipped_path.open("w") as f:
        json.dump(skipped, f, indent=2)

    print()
    print(f"Finished processing {count} files")
    print(f"Skipped files listed in {skipped_path}")


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
