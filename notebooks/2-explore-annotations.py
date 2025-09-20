import marimo

__generated_with = "0.15.3"
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

    from beatdetect.config_loader import load_config
    from beatdetect.utils.paths import iterate_beat_files, PathResolver

    import torch
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    return (
        Counter,
        PathResolver,
        go,
        iterate_beat_files,
        load_config,
        make_subplots,
        mo,
        np,
        pl,
        px,
        torch,
    )


@app.cell
def _(load_config):
    config = load_config()
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exploring time signatures""")
    return


@app.cell
def _(Counter, config, iterate_beat_files, pl):
    beat_time_signatures = []
    for _beats_file in iterate_beat_files(config):
        try:
            _beats = (
                pl.read_csv(
                    _beats_file,
                    separator="\t",
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
    beat_time_signatures
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exploring encoded beats""")
    return


@app.cell
def _(PathResolver, config, torch):
    paths = PathResolver(config, "tapcorrect")
    _sample = "001_youtube_fV4DiAyExN0.pt"
    beats = torch.load(paths.encoded_beats_dir / _sample).numpy()
    downbeats = torch.load(paths.encoded_downbeats_dir / _sample).numpy()

    num_frames = beats.shape[0]
    num_frames
    return beats, downbeats


@app.cell
def _(beats, downbeats, go, make_subplots, np):
    _start, _end = 2000, 3000
    _frames = np.arange(_start, _end)
    _beats = beats[_start:_end]
    _downbeats = downbeats[_start:_end]

    _fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("Beats", "Downbeats")
    )

    # Beats
    _fig.add_trace(
        go.Scatter(
            x=_frames,
            y=_beats,
            mode="lines",
            name="Beats",
            line=dict(color="royalblue"),
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Scatter(
            x=_frames,
            y=_downbeats,
            mode="lines",
            name="Downbeats",
            line=dict(color="firebrick"),
        ),
        row=2,
        col=1,
    )

    _fig.update_layout(
        height=500,
        title="Encoded Beats / Downbeats",
        xaxis_title="Frame index",
        yaxis_title="Activation",
        template="plotly_white",
        showlegend=False,
    )

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Class imbalance""")
    return


@app.cell
def _(PathResolver, config, torch):
    ratios = {"beats": [], "downbeats": []}

    for dataset in config.downloads.datasets:
        _paths = PathResolver(config, dataset)
        for _beats_file in _paths.encoded_beats_dir.glob("*.pt"):
            _beats = torch.load(_beats_file).numpy()
            ratios["beats"].append(_beats.shape[0] / sum(_beats))
        for _downbeats_file in _paths.encoded_downbeats_dir.glob("*.pt"):
            _downbeats = torch.load(_downbeats_file).numpy()
            ratios["downbeats"].append(_downbeats.shape[0] / sum(_downbeats))
    return (ratios,)


@app.cell
def _(px, ratios):
    px.histogram(ratios, marginal="box")
    return


if __name__ == "__main__":
    app.run()
