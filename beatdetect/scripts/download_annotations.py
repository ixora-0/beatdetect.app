import datetime
import time
import urllib.request
import zipfile

from tqdm import tqdm

from beatdetect.config_loader import Config, load_config


class TqdmHook:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            # If total_size is unknown (-1), create indeterminate progress bar
            if total_size <= 0:
                self.pbar = tqdm(unit="B", unit_scale=True, desc="Downloading")
            else:
                self.pbar = tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                )

        downloaded = block_num * block_size

        if total_size <= 0:
            # For unknown size, just update with block size
            if block_num > 0:
                self.pbar.update(block_size)
        else:
            # For known size, update normally
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.update(total_size - self.pbar.n)

        # Close when download seems complete (no more data)
        if block_num > 0 and block_size == 0:
            self.pbar.close()


def main(config: Config):
    config.paths.downloads.mkdir(parents=True, exist_ok=True)

    info = config.downloads.annotations
    file_name = f"{info.github_branch}.zip"
    url = f"https://github.com/{info.github_user}/{info.github_repo}/archive/refs/heads/{file_name}"

    # downloading
    output_file = config.paths.downloads / file_name
    if output_file.exists():
        print(f"File {output_file} exists, skipping download")
    else:
        print(f"Fetching from {url} into {output_file}")
        urllib.request.urlretrieve(url, output_file, reporthook=TqdmHook())
        print("Finished downloading.")

    # extracting
    out_dir = config.paths.data.raw.annotations
    print(f"Extracting to {out_dir}")
    start_time = time.perf_counter()

    with zipfile.ZipFile(output_file, "r") as zf:
        files = [f for f in zf.namelist() if not f.endswith("/")]  # skips directories

        for member in tqdm(files, desc="Extracting", unit="file"):
            parts = member.split("/", 1)
            out_path = out_dir / parts[1 if len(parts) == 2 else 0]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                f.write(zf.read(member))
    time_taken = time.perf_counter() - start_time
    print(f"Time elapsed: {datetime.timedelta(seconds=round(time_taken))}")
    print("Done.")


if __name__ == "__main__":
    main(load_config())
