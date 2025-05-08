

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pathlib
    import re
    from collections import Counter

    import marimo as mo
    import plotly.express as px
    import polars as pl

    from beatdetect import ANNOTATIONS_PROCESSED_PATH, ANNOTATIONS_RAW_PATH, DATASETS

    return (
        ANNOTATIONS_PROCESSED_PATH,
        ANNOTATIONS_RAW_PATH,
        Counter,
        DATASETS,
        json,
        mo,
        pathlib,
        pl,
        px,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Processing data

        Having some trouble reading annotations file because of inconsistent separator
        """
    )
    return


@app.cell
def _(ANNOTATIONS_PROCESSED_PATH, ANNOTATIONS_RAW_PATH, DATASETS, json):
    # helper
    def iterate_beat_files(processed=True, has_downbeats=True):
        path = ANNOTATIONS_PROCESSED_PATH if processed else ANNOTATIONS_RAW_PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} doesn't exist.")

        if processed:
            with (path / "info.json").open("r") as f:        
                info = json.load(f)
        else:
            info = {}
            for dataset in DATASETS:
                info_file_path = path / dataset / "info.json"
                with info_file_path.open("r") as f:
                    info[dataset] = json.load(f)

        for dataset in DATASETS:
            if has_downbeats and not info[dataset]["has_downbeats"]:
                continue

            dataset_path = path / dataset
            for beats_file in dataset_path.rglob("*.beats"):
                with open(beats_file) as f:
                    if not f.read(1):
                        print(f"\033[K{beats_file} is empty, skipping.")
                        continue
                yield beats_file

    return (iterate_beat_files,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Getting an idea of separator frequencies in data""")
    return


@app.cell
def _(iterate_beat_files):
    _sep_freq = {" ": 0, "\t": 0, ",": 0}
    for _path in iterate_beat_files(processed=False, has_downbeats=False):
        with open(_path, "r") as _f:
            _l = _f.readline()
            for sep in (" ", "\t", ","):
                if sep in _l:
                    _sep_freq[sep] += 1

    for _sep, _count in _sep_freq.items():
        print(f"  {repr(_sep)}: {_count}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Converting all separators to one tab""")
    return


@app.cell
def _(ANNOTATIONS_PROCESSED_PATH, ANNOTATIONS_RAW_PATH, pathlib, re):
    separator = "\t"
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
            new_content = re.sub(r"[\t ]+", separator, original_content)

            # determine the output path in ANNOTATIONS_PROCESSED_PATH
            relative_path = path.relative_to(ANNOTATIONS_RAW_PATH)
            output_path = ANNOTATIONS_PROCESSED_PATH / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # write cleaned content to new location
            with open(output_path, "w") as file:
                file.write(new_content)

            return True

        except Exception as e:
            print(f"\r\033[KError processing {path}, skipping. Error: {e}")
            return False

    return normalize_annotation, separator


@app.cell
def _(ANNOTATIONS_PROCESSED_PATH, iterate_beat_files, normalize_annotation):
    _total_files = 0
    _processed_files = 0
    import time
    for idx, input_path in enumerate(iterate_beat_files(processed=False), 1):
        _total_files += 1
        print(f"\r\033[KProcessing file #{idx}: {input_path}", end="")
        if normalize_annotation(input_path):
            _processed_files += 1

    print()
    print(f"Total files processed: {_total_files}")
    print(f"Files saved to {ANNOTATIONS_PROCESSED_PATH}: {_processed_files}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Mergin all info.json from dataset into one""")
    return


@app.cell
def _(ANNOTATIONS_PROCESSED_PATH, ANNOTATIONS_RAW_PATH, DATASETS, JSON, json):
    def combine_info():
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
    combine_info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exploring time signatures""")
    return


@app.cell
def _(Counter, iterate_beat_files, pl, separator):
    beat_time_signatures = []
    for _beats_file in iterate_beat_files():
        try:
            _beats = (
                pl.read_csv(
                    _beats_file,
                    separator=separator,
                    has_header=False,
                )
                .get_columns()[1]
                .to_list()
            )
        except IndexError:
            print(f"{_beats_file} doesn't have beat info, skipping.")
        _ending_beats = Counter(
            _beats[i] for i in range(len(_beats) - 1) if _beats[i + 1] == 1
        )
        _diff_signatures = set(_ending_beats.keys())
        _time_signature = Counter(
            _beats[i] for i in range(len(_beats) - 1) if _beats[i + 1] == 1
        ).most_common(1)[0][0]

        beat_time_signatures.append(
            {
                "Name": _beats_file.name,
                "Time Signature": _time_signature,
                "Others": list(_diff_signatures - {_time_signature}),
            }
        )
    beat_time_signatures = pl.DataFrame(beat_time_signatures)

    return (beat_time_signatures,)


@app.cell
def _(beat_time_signatures, px):
    px.histogram(beat_time_signatures, x="Time Signature")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Mostly 4/4 as expected.

        Finding songs that have time signature change
        """
    )
    return


@app.cell
def _(beat_time_signatures, pl):
    beat_time_signatures.filter(pl.col("Others").list.len() != 0)
    return


if __name__ == "__main__":
    app.run()
