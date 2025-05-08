import argparse
import datetime
import math
import pathlib
import shutil
import time
import zipfile
from typing import NotRequired, TypedDict
from urllib.parse import urljoin

import requests
from pypdl import Pypdl
from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper

from beatdetect import (
    DATASETS,
    DOWNLOADS_PATH,
    RC_API_URL,
    SPECTROGRAMS_RAW_PATH,
    SPECTROGRAMS_URL_TEMPLATE,
)
from beatdetect.utils import JSON


class Stats(TypedDict):
    name: str
    size: float
    bytes: NotRequired[float]
    percentage: NotRequired[float]
    eta: NotRequired[float]
    speed: NotRequired[float]


SEGMENTS = 5
# wait time (s) before each poll to rc server (only used applicable if remote)
INTERVAL = 1


def format_bytes(n: float) -> str:
    """Format bytes in a human-readable way."""
    if n == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = min(int(math.log2(n) / 10), len(units) - 1)
    return f"{n / (2 ** (10 * idx)):.2f} {units[idx]}"


def get_transfer_stats(path: pathlib.Path) -> Stats | None:
    """Get transfer stats from rclone remote control server for a given path."""
    endpoint = urljoin(RC_API_URL, "core/stats")
    resp = requests.post(endpoint)
    stats: JSON = resp.json()
    transferring: list[Stats] = stats.get("transferring", [])
    for transfer in transferring:
        subpath = pathlib.Path(transfer["name"])
        path_parts = path.parts
        subpath_parts = subpath.parts

        if len(path_parts) >= len(subpath_parts):
            if path_parts[-len(subpath_parts) :] == subpath_parts:
                return transfer
    return None


def wait_for_transfer(download_path: pathlib.Path) -> None:
    """Wait for the file to be transferred."""
    print("Waiting for transfer to potentially start.")
    # HACK: need some delay before rclone to start transferring
    # waiting INTERVAL * 5
    for _ in range(5):
        if get_transfer_stats(download_path) is not None:
            pbar = tqdm(
                desc=f"Transferring {download_path}",
                unit="iB",
                unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}{postfix}",
            )
            while True:
                if (stats := get_transfer_stats(download_path)) is None:
                    break
                pbar.total = stats["size"]
                pbar.n = stats.get("bytes", 0)
                pbar.set_postfix(
                    eta=datetime.timedelta(seconds=stats.get("eta") or 0),
                    speed=format_bytes(stats.get("speed", 0)) + "/s",
                )
                time.sleep(INTERVAL)
            pbar.close()
            break
        time.sleep(INTERVAL)


def main(remote: bool):
    """Download datasets"""
    DOWNLOADS_PATH.mkdir(parents=True, exist_ok=True)

    dl = Pypdl(allow_reuse=True)
    for dataset in DATASETS[12:]:
        filename = f"{dataset}.zip"
        download_path = DOWNLOADS_PATH / filename

        # downloading
        if download_path.is_file():
            print(f"Skipping downloading {dataset}, {download_path} exists")
        else:
            url = SPECTROGRAMS_URL_TEMPLATE.substitute(dataset=dataset)
            print(f"Downloading {dataset}.")
            print(f"Fetching from {url} into {download_path}")
            dl.start(
                url,
                str(download_path),
                retries=3,
                clear_terminal=False,
                segments=SEGMENTS,
                overwrite=False,
            )
            print("Finished downloading.")

        # wait until file is no longer being uploaded
        if remote:
            wait_for_transfer(download_path)

        # now safe to unzip
        unzip_path = SPECTROGRAMS_RAW_PATH / dataset
        unzip_path.mkdir(parents=True, exist_ok=True)
        with (
            zipfile.ZipFile(download_path) as zf,
            tqdm(
                desc=f"Extracting {dataset} to {unzip_path}",
                unit="iB",
                unit_scale=True,
                total=sum(i.file_size for i in zf.infolist()),
            ) as pbar,
        ):
            for i in zf.infolist():
                with zf.open(i) as fi, open(unzip_path / i.filename, "wb") as fo:
                    shutil.copyfileobj(CallbackIOWrapper(pbar.update, fi), fo)

    dl.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets from {}".format(
            SPECTROGRAMS_URL_TEMPLATE.substitute(dataset="<dataset>")
        )
    )

    parser.add_argument(
        "--remote",
        action="store_true",
        default=False,
        help=f"{DOWNLOADS_PATH} is remote and mounted using rclone with remote control server at {RC_API_URL}.",
    )

    args = parser.parse_args()

    main(args.remote)
