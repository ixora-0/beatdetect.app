import json

from beatdetect import ANNOTATIONS_PROCESSED_PATH, ANNOTATIONS_RAW_PATH, DATASETS
from beatdetect.utils import JSON


def main():
    combined_info: dict[str, JSON] = {}
    for dataset in DATASETS:
        info_file_path = ANNOTATIONS_RAW_PATH / dataset / "info.json"
        with open(info_file_path, "r") as f:
            combined_info[dataset] = json.load(f)

    ANNOTATIONS_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    combined_info_path = ANNOTATIONS_PROCESSED_PATH / "info.json"
    with open(combined_info_path, "w") as f:
        json.dump(combined_info, f, indent=2)

    print(f"Combined dataset information has been saved to {combined_info_path}")


if __name__ == "__main__":
    main()
