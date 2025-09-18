import re
from pathlib import Path

from ..config_loader import Config, load_config
from ..utils.paths import iterate_beat_files


def normalize_annotation(input_path: Path, output_path: Path) -> bool:
    """
    Clean a beat file by replacing inconsistent separators with a single tab
    and save it to the processed path.

    Args:
        input_path (pathlib.Path): The path to the original beat file.
        output_path (pathlib.Path): Destination of clean beat file to save to.

    Returns:
        bool: True if the file was cleaned and saved, False otherwise.
    """
    try:
        with open(input_path) as file:
            original_content = file.read()

        # replace inconsistent separators with a single tab
        new_content = re.sub(r"[\t ]+", "\t", original_content)

        # write cleaned content to new location
        with open(output_path, "w") as file:
            file.write(new_content)

        return True

    except Exception as e:
        print(f"Error processing {input_path}, skipping. Error: {e}")
        return False


def main(config: Config):
    beat_files = iterate_beat_files(config, cleaned=False)
    total_files = 0
    cleaned_files = 0
    for idx, input_path in enumerate(beat_files, 1):
        total_files += 1
        print(f"\033[KProcessing file #{idx}: {input_path}", end="\r")

        # determine the output path
        relative_path = input_path.relative_to(config.paths.data.raw.annotations)
        output_path = config.paths.data.interim.annotations / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if normalize_annotation(input_path, output_path):
            cleaned_files += 1

    print(f"Total files processed: {total_files}")
    print(f"Files saved to {config.paths.data.interim.annotations}: {cleaned_files}")


if __name__ == "__main__":
    config = load_config()
    main(config)
