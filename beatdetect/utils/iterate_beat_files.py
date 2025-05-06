import json

from beatdetect import ANNOTATIONS_PROCESSED_PATH, ANNOTATIONS_RAW_PATH


def iterate_beat_files(processed=True, has_downbeats=True):
    path = ANNOTATIONS_PROCESSED_PATH if processed else ANNOTATIONS_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(f"{ANNOTATIONS_RAW_PATH} doesn't exist.")

    annotation_raw_dataset_paths = [p for p in path.iterdir() if p.is_dir()]
    for dataset_path in annotation_raw_dataset_paths:
        info_file_path = dataset_path / "info.json"
        if has_downbeats:
            with info_file_path.open("r") as f:
                info = json.load(f)
            if not info["has_downbeats"]:
                continue

        for beats_file in dataset_path.rglob("*.beats"):
            with open(beats_file) as f:
                if not f.read(1):
                    print(f"\033[K{beats_file} is empty, skipping.")
                    continue
            yield beats_file
