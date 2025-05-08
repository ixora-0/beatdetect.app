

import marimo

__generated_with = "0.13.2"
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

    from beatdetect import (
        ANNOTATIONS_PROCESSED_PATH,
        ANNOTATIONS_RAW_PATH,
        DATASETS,
        SPECTROGRAMS_RAW_PATH,
    )
    from beatdetect.histogram_params import (
        F_MAX,
        F_MIN,
        HOP_LENGTH,
        MEL_SCALE,
        N_FFT,
        N_MELS,
        N_STFT,
        SAMPLE_RATE,
    )
    return (
        ANNOTATIONS_PROCESSED_PATH,
        F_MAX,
        F_MIN,
        HOP_LENGTH,
        MEL_SCALE,
        N_FFT,
        N_MELS,
        N_STFT,
        SAMPLE_RATE,
        SPECTROGRAMS_RAW_PATH,
        cs,
        librosa,
        mo,
        np,
        pl,
        px,
        torch,
        torchaudio,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load spectrogram dataset""")
    return


@app.cell
def _(SPECTROGRAMS_RAW_PATH, np):
    dataset = "tapcorrect"
    d = np.load(SPECTROGRAMS_RAW_PATH / dataset / f"{dataset}.npz")
    return d, dataset


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
    px.imshow(melspect[_start:_start+_len, :].T, origin="lower")
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
def _(
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    MEL_SCALE,
    N_FFT,
    N_MELS,
    N_STFT,
    SAMPLE_RATE,
    melspect,
    torch,
    torchaudio,
):
    # inverse the ln(1+1000x) scaling
    _melspect_scaled = (torch.exp(melspect) - 1) / 1000

    # torchaudio works on float32 but data is float16
    _melspect_float = _melspect_scaled.to(torch.float32)

    # reshape to [batch, n_mels, time]
    _melspect_reshaped = _melspect_float.unsqueeze(0).permute(0, 2, 1)

    # inverse mel-scale spectrogram to power spectrogram
    _inversed_waveform = torchaudio.transforms.InverseMelScale(
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        n_stft=N_STFT,
        f_min=F_MIN,
        f_max=F_MAX,
        mel_scale=MEL_SCALE
    )(_melspect_reshaped)
    # inverse power spectrogram to waveform 
    waveform = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=1,
    )(_inversed_waveform)
    # output is shape [n_channels, time], assuming we have only 1 channel
    waveform = waveform[0]

    return (waveform,)


@app.cell
def _(SAMPLE_RATE, mo, waveform):
    mo.audio(waveform.numpy(), rate=SAMPLE_RATE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Low quality audio because the power spectrograms doesn't have phase information for component frequencies.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load corresponding annotation""")
    return


@app.cell
def _(ANNOTATIONS_PROCESSED_PATH, dataset):
    annotation_dataset_paths = [p for p in ANNOTATIONS_PROCESSED_PATH.iterdir() if p.is_dir()]
    annotation_dataset_path = [p for p in annotation_dataset_paths if p.name == dataset][0]
    return (annotation_dataset_path,)


@app.cell
def _(annotation_dataset_path, fn, pl):
    _annotation_file_name = fn.removesuffix("/track") + ".beats"
    beat_df = pl.read_csv(
        annotation_dataset_path / "annotations/beats" / _annotation_file_name,
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
def _(SAMPLE_RATE, beat_df, px, torch, waveform):
    _start = 10
    _len = 30
    _waveform_slice = waveform[_start*SAMPLE_RATE:(_start+_len)*SAMPLE_RATE]

    _fig = px.line(
        x=_start + torch.arange(len(_waveform_slice)) / SAMPLE_RATE,
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
def _(SAMPLE_RATE, beat_df, cs, librosa, mo, waveform):
    _beats = librosa.clicks(
        times=beat_df.select(cs.by_index(0)),
        sr=SAMPLE_RATE,
        click_freq=1000.0,
        click_duration=0.1,
        length=len(waveform)
    )
    _downbeats = librosa.clicks(
        times=beat_df.filter(cs.by_index(1) == 1).select(cs.by_index(0)),
        sr=SAMPLE_RATE,
        click_freq=1500.0,
        click_duration=0.15,
        length=len(waveform)
    )
    _click_volume = 0.05
    mo.audio(_click_volume * _beats + _click_volume * _downbeats + waveform.numpy(), rate=SAMPLE_RATE)
    return


if __name__ == "__main__":
    app.run()
