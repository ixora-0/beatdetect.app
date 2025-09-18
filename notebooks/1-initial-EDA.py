import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import librosa
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl
    import polars.selectors as cs
    import requests
    import torch
    import torchaudio

    from beatdetect.config_loader import load_config
    from beatdetect.utils.paths import PathResolver
    return (
        PathResolver,
        cs,
        librosa,
        load_config,
        mo,
        np,
        pl,
        px,
        torch,
        torchaudio,
    )


@app.cell
def _(load_config):
    config = load_config()
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load spectrogram dataset""")
    return


@app.cell
def _(PathResolver, config, np):
    dataset = "tapcorrect"
    paths = PathResolver(config, dataset)
    d = np.load(paths.spectrograms_file)
    return d, paths


@app.cell
def _(d):
    d.files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exploring a single sample""")
    return


@app.cell
def _(d, torch):
    fn = d.files[0]
    melspect = torch.from_numpy(d[fn])
    melspect.shape
    return fn, melspect


@app.cell
def _(melspect, px):
    _start = 0
    _len = 500
    px.imshow(melspect[_start : _start + _len, :].T, origin="lower")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Reconstructing audio from spectrogram""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="https://download.pytorch.org/torchaudio/tutorial-assets/torchaudio_feature_extractions.png"
    )
    return


@app.cell
def _(config, melspect, torch, torchaudio):
    # inverse the ln(1+1000x) scaling
    _melspect_scaled = (torch.exp(melspect) - 1) / 1000

    # torchaudio works on float32 but data is float16
    _melspect_float = _melspect_scaled.to(torch.float32)

    # reshape to [batch, n_mels, time]
    _melspect_reshaped = _melspect_float.unsqueeze(0).permute(0, 2, 1)

    # inverse mel-scale spectrogram to power spectrogram
    _inversed_waveform = torchaudio.transforms.InverseMelScale(
        n_mels=config.spectrogram.n_mels,
        sample_rate=config.spectrogram.sample_rate,
        n_stft=config.spectrogram.n_stft,
        f_min=config.spectrogram.f_min,
        f_max=config.spectrogram.f_max,
        mel_scale=config.spectrogram.mel_scale,
    )(_melspect_reshaped)
    # inverse power spectrogram to waveform
    waveform = torchaudio.transforms.GriffinLim(
        n_fft=config.spectrogram.n_fft,
        hop_length=config.spectrogram.hop_length,
        power=1,
    )(_inversed_waveform)
    # output is shape [n_channels, time], assuming we have only 1 channel
    waveform = waveform[0]
    return (waveform,)


@app.cell
def _(config, mo, waveform):
    mo.audio(waveform.numpy(), rate=config.spectrogram.sample_rate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Low quality audio because the power spectrograms doesn't have phase information for component frequencies."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load corresponding annotation""")
    return


@app.cell
def _(fn, paths, pl):
    _annotation_file_name = fn.removesuffix("/track") + ".beats"
    beat_df = pl.read_csv(
        paths.resolve_annotations_dir(cleaned=True) / _annotation_file_name,
        separator="\t",
        has_header=False,
    )
    beat_df
    return (beat_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exploring waveform with beats""")
    return


@app.cell
def _(beat_df, config, px, torch, waveform):
    _start = 10
    _len = 30
    _waveform_slice = waveform[
        _start * config.spectrogram.sample_rate : (_start + _len)
        * config.spectrogram.sample_rate
    ]

    _fig = px.line(
        x=_start
        + torch.arange(len(_waveform_slice)) / config.spectrogram.sample_rate,
        y=_waveform_slice,
        title="Waveform with beats",
        width=1000,
        height=400,
    )
    _fig.update_traces(line=dict(width=1))
    _fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")

    _beat_colors = px.colors.qualitative.Plotly[1:3]
    _high, _low = _waveform_slice.max(), _waveform_slice.min()
    _showed_legend = {"Downbeat": False, "Beat": False}
    for _time, _beat in beat_df.rows():
        if _start > _time or _time > _start + _len:
            continue
        _is_downbeat = _beat == 1
        _name = "Downbeat" if _is_downbeat else "Beat"
        _fig.add_shape(
            showlegend=not _showed_legend[_name],
            legendgroup=_name,
            name=_name,
            type="line",
            x0=_time,
            x1=_time,
            y0=_low,
            y1=_high,
            line_width=1,
            line_color=_beat_colors[0] if _is_downbeat else _beat_colors[1],
            line_dash="solid" if _is_downbeat else "5px",
        )
        _showed_legend[_name] = True

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create audio with click track""")
    return


@app.cell
def _(beat_df, config, cs, librosa, mo, waveform):
    _beats = librosa.clicks(
        times=beat_df.select(cs.by_index(0)),
        sr=config.spectrogram.sample_rate,
        click_freq=1000.0,
        click_duration=0.1,
        length=len(waveform),
    )
    _downbeats = librosa.clicks(
        times=beat_df.filter(cs.by_index(1) == 1).select(cs.by_index(0)),
        sr=config.spectrogram.sample_rate,
        click_freq=1500.0,
        click_duration=0.15,
        length=len(waveform),
    )
    _click_volume = 0.05
    mo.audio(
        _click_volume * _beats + _click_volume * _downbeats + waveform.numpy(),
        rate=config.spectrogram.sample_rate,
    )
    return


if __name__ == "__main__":
    app.run()
