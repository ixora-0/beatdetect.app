import time
import datetime
import zipfile
from pypdl import Pypdl
from beatdetect import (
    DOWNLOADS_PATH,
    ANNOTATIONS_GITHUB_USER,
    ANNOTATIONS_GITHUB_REPO,
    ANNOTATIONS_GITHUB_BRANCH,
    ANNOTATIONS_RAW_PATH,
)


def main():
    DOWNLOADS_PATH.mkdir(parents=True, exist_ok=True)
    file_name = f"{ANNOTATIONS_GITHUB_BRANCH}.zip"
    url = "https://github.com/{}/{}/archive/refs/heads/{}".format(
        ANNOTATIONS_GITHUB_USER, ANNOTATIONS_GITHUB_REPO, file_name
    )

    # downloading
    target_dir = str(DOWNLOADS_PATH)  # pypdl doesn't take pathlib.Path
    print(f"Fetching from {url} into {target_dir}")
    dl = Pypdl()
    dl.start(url, target_dir, retries=3, clear_terminal=False)
    print("Finished downloading.")

    # extracting
    print(f"Extracting to {ANNOTATIONS_RAW_PATH}")
    start_time = time.perf_counter()

    with zipfile.ZipFile(DOWNLOADS_PATH / file_name, "r") as zf:
        files = [f for f in zf.namelist() if not f.endswith("/")]  # skips directories
        total = len(files)

        for i, member in enumerate(files, 1):
            parts = member.split("/", 1)
            out_path = ANNOTATIONS_RAW_PATH / parts[1 if len(parts) == 2 else 0]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                f.write(zf.read(member))
            print(f"Total extracted {i}/{total}", end="\r")
        print()
    time_taken = time.perf_counter() - start_time
    print(f"Time elapsed: {datetime.timedelta(seconds=round(time_taken))}")

    print("Done.")


if __name__ == "__main__":
    main()
