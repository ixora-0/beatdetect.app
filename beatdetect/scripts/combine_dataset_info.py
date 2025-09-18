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

    out = config.paths.data.interim.annotations
    out.mkdir(parents=True, exist_ok=True)
    combined_info_path = out / "info.json"
    with open(combined_info_path, "w") as f:
        json.dump(combined_info, f, indent=2)

    print(f"Combined dataset information has been saved to {combined_info_path}")


if __name__ == "__main__":
    main(load_config())
