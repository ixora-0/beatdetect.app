import json

from ..config_loader import Config, load_config
from ..utils import JSON


def main(config: Config):
    inp = config.paths.data.raw.annotations
    combined_info: dict[str, JSON] = {}
    for dataset in config.downloads.datasets:
        info_file_path = inp / dataset / "info.json"
        with open(info_file_path) as f:
            combined_info[dataset] = json.load(f)

    combined_info_path = config.paths.data.processed.datasets_info
    combined_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_info_path, "w") as f:
        json.dump(combined_info, f, indent=2)

    print(f"Combined dataset information has been saved to {combined_info_path}")


if __name__ == "__main__":
    main(load_config())
