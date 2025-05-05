import pathlib
import re

from beatdetect import ANNOTATIONS_PROCESSED_PATH, ANNOTATIONS_RAW_PATH
from beatdetect.utils import iterate_beat_files


def normalize_annotation(path: pathlib.Path) -> bool:
    """
    Clean a beat file by replacing inconsistent separators with a single tab and save it to the processed path.

    Args:
        input_path (pathlib.Path): The path to the original beat file.

    Returns:
        bool: True if the file was cleaned and saved, False otherwise.
    """
    try:
        with open(path, "r") as file:
            original_content = file.read()

        # replace inconsistent separators with a single tab
        new_content = re.sub(r"[\t ]+", "\t", original_content)

        # determine the output path in ANNOTATIONS_PROCESSED_PATH
        relative_path = path.relative_to(ANNOTATIONS_RAW_PATH)
        output_path = ANNOTATIONS_PROCESSED_PATH / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # write cleaned content to new location
        with open(output_path, "w") as file:
            file.write(new_content)

        return True

    except Exception as e:
        print(f"Error processing {path}, skipping. Error: {e}")
        return False


def main():
    total_files = 0
    processed_files = 0

    for idx, input_path in enumerate(iterate_beat_files(), 1):
        total_files += 1
        print(f"\033[KProcessing file #{idx}: {input_path}", end="\r")
        if normalize_annotation(input_path):
            processed_files += 1

    print(f"Total files processed: {total_files}")
    print(f"Files saved to {ANNOTATIONS_PROCESSED_PATH}: {processed_files}")


if __name__ == "__main__":
    main()
