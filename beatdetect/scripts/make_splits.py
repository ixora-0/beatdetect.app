import csv
import json
import random
from pathlib import Path

from ..config_loader import Config, load_config
from ..utils.paths import PathResolver


def load_splits(splits_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load splits from a JSON file."""
    with open(splits_path) as f:
        splits_data = json.load(f)

    # Convert lists back to tuples
    splits = {
        split_name: [tuple(sample) for sample in samples]
        for split_name, samples in splits_data.items()
    }

    return splits


def main(config: Config):
    print("Creating splits.")
    hypers = config.training

    # Set seed for reproducibility
    random.seed(config.random_seed)

    # List of [dataset, name] per data point
    all_samples = []

    for dataset in config.downloads.datasets:
        paths = PathResolver(config, dataset)

        # load available track names for this dataset
        # using encoded beats files to determine names
        names = sorted(
            [p.name.removesuffix(".pt") for p in paths.encoded_beats_dir.glob("*.pt")]
        )

        for name in names:
            beats_file = paths.encoded_beats_dir / f"{name}.pt"
            downbeats_file = paths.encoded_downbeats_dir / f"{name}.pt"
            if beats_file.exists() and downbeats_file.exists():
                all_samples.append([dataset, name])
            else:
                print(f"Warning: Missing annotations for {dataset}/{name}")

    # Calculate split sizes
    total_samples = len(all_samples)
    train_size = int(total_samples * hypers.train_ratio)
    val_size = int(total_samples * hypers.val_ratio)

    # Shuffle samples
    random.shuffle(all_samples)

    splits = {
        "train": all_samples[:train_size],
        "val": all_samples[train_size : train_size + val_size],
        "test": all_samples[train_size + val_size :],
    }

    print(f"Dataset split created with seed {config.random_seed}:")
    print(f"  Total samples: {total_samples}")
    print(f"  Train: {train_size} samples")
    print(f"  Validation: {val_size} samples")
    print(f"  Test: {len(splits['test'])} samples")

    # Save splits to CSV file
    output_path = config.paths.data.processed.splits_info
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["dataset", "name", "split"])
        # Rows
        for split_name, samples in splits.items():
            for dataset, name in samples:
                writer.writerow([dataset, name, split_name])

    print(f"Splits saved to: {output_path}")
    print("Finished splitting dataset.")


if __name__ == "__main__":
    config = load_config()
    main(config)
