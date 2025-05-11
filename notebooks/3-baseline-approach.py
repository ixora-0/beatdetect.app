

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pprint import pprint

    import librosa
    import marimo as mo
    import mir_eval
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import polars.selectors as cs
    import torch
    import torchaudio
    from plotly.subplots import make_subplots

    from beatdetect import ANNOTATIONS_PROCESSED_PATH, SPECTROGRAMS_RAW_PATH
    from beatdetect.histogram_params import (
        HOP_LENGTH,
        N_FFT,
        N_MELS,
        N_STFT,
        SAMPLE_RATE,
    )
    return (
        ANNOTATIONS_PROCESSED_PATH,
        HOP_LENGTH,
        SAMPLE_RATE,
        SPECTROGRAMS_RAW_PATH,
        go,
        librosa,
        make_subplots,
        mir_eval,
        mo,
        np,
        pl,
        px,
        torch,
        torchaudio,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Purpose

        The goal of this notebook is to estimate the tempo and beat from musical audio signals using traditional signal processing techniques.
        The methodology we follow is described in:

        Matthew E. P. Davies, M. (2021). _Tempo, Beat and Downbeat Estimation_. https://tempobeatdownbeat.github.io/tutorial/intro.html.

        See in Chapter 2, "Baseline approach", and "How do we evaluate?" sections for more details and justifications for the approach.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Loading a sample""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load spectrogram""")
    return


@app.cell
def _(SPECTROGRAMS_RAW_PATH, np, torch):
    dataset = "tapcorrect"
    d = np.load(SPECTROGRAMS_RAW_PATH / dataset / f"{dataset}.npz")
    fn = d.files[0]
    melspect_raw = torch.from_numpy(d[fn]).to(torch.float32)
    melspect_raw.shape

    return dataset, fn, melspect_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Slice spectrogram into a smaller section for clearer visualization.""")
    return


@app.cell
def _(HOP_LENGTH, SAMPLE_RATE, librosa, melspect_raw, np, px):
    start = 15
    length = 15

    _start_frame, _end_frame = librosa.time_to_frames([start, start + length], sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    melspect = melspect_raw[_start_frame:_end_frame, :]

    _xs_frames = np.arange(_start_frame, _end_frame)
    xs = librosa.frames_to_time(_xs_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    px.imshow(
        melspect.T,
        origin="lower",
        aspect="auto",
        x=xs,
        labels={"x": "Time (s)", "y": "Mel Frequency Bin"},
    )
    return length, melspect, start, xs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load beat annotations""")
    return


@app.cell
def _(ANNOTATIONS_PROCESSED_PATH, dataset, fn, pl):
    annotation_dataset_paths = [p for p in ANNOTATIONS_PROCESSED_PATH.iterdir() if p.is_dir()]
    annotation_dataset_path = [p for p in annotation_dataset_paths if p.name == dataset][0]
    annotation_file_name = fn.removesuffix("/track") + ".beats"
    beat_df = pl.read_csv(
        annotation_dataset_path / "annotations/beats" / annotation_file_name,
        separator="\t",
        has_header=False,
    )
    beat_df
    return (beat_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Ignoring downbeats for our purpose. Also trimming it down to our audio slice.""")
    return


@app.cell
def _(beat_df, length, start):
    annotated_beats = beat_df.to_numpy()[:, 0]
    annotated_beats = annotated_beats[(annotated_beats >= start) & (annotated_beats <= start + length)]
    annotated_beats.shape
    return (annotated_beats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Processsing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tempo""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Annotated (correct) tempo

        Get tempo from median inter beat interval from annotations.
        """
    )
    return


@app.cell
def _(annotated_beats, np):
    inferred_tempo = 60 / np.median(np.diff(annotated_beats))
    inferred_tempo
    return (inferred_tempo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Spectral flux

        Get spectral flux. Measures how quickly the power spectrum of a signal changes over time, often used to detect audio onsets (ie. notes of instruments being hit). It is the difference in spectral content between the each frame and the its previous frame.
        """
    )
    return


@app.cell
def _(HOP_LENGTH, SAMPLE_RATE, librosa, melspect, torchaudio):
    # ibrosa.onset.onset_strength requires log-power spectrogram
    log_melspect = torchaudio.transforms.AmplitudeToDB(stype="power")(melspect)

    spectral_flux = librosa.onset.onset_strength(
        S=log_melspect.T,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        lag=2,
        max_size=3,
    )
    return (spectral_flux,)


@app.cell
def _(annotated_beats, go, make_subplots, melspect, np, px, spectral_flux, xs):
    def create_beat_lines(beats):
        beat_lines = []
        showed_legend = False
        for time in beats:
            beat_lines.append(
                dict(
                    showlegend=not showed_legend,
                    legendgroup=0,
                    type="line",
                    x=time,
                    line_width=1,
                    opacity=0.5,
                )
            )
            showed_legend = True

        return beat_lines

    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Log Mel Spectrogram", "Spectral Flux"],
    )

    # Add spectrogram
    _fig.add_trace(
        go.Heatmap(
            z=melspect.T,
            colorbar=dict(
                title="",
                len=0.6,
                yanchor="top",
                y=1.1,
                x=1.02,
            ),
            x=xs,
            y=np.arange(melspect.shape[0]),
            name="Mel Spectrogram",
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Scatter(
            x=xs,
            y=spectral_flux,
            mode="lines",
            name="Spectral Flux",
            line_color=px.colors.qualitative.Plotly[0],
            line_width=1,
        ),
        row=2,
        col=1,
    )

    # Add vertical beat lines
    annotated_beats_lines = create_beat_lines(annotated_beats)
    for _line in annotated_beats_lines:
        _fig.add_vline(
            name="Annotated beat",
            line_color=px.colors.qualitative.Plotly[1],
            row=2,
            col=1,
            **_line,
        )

    # Style
    _fig.update_layout(
        yaxis_title="Mel Frequency Bin",
        showlegend=True,
        legend=dict(yanchor="top", y=0.5, xanchor="left", x=1.02),
    )
    _fig.update_xaxes(title_text="Time (s)", showgrid=False, row=2, col=1)
    return annotated_beats_lines, create_beat_lines


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Estimate tempo""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Getting tempogram (local autocorrelation of spectral flux).""")
    return


@app.cell
def _(HOP_LENGTH, SAMPLE_RATE, go, inferred_tempo, librosa, spectral_flux, xs):
    tempogram = librosa.feature.tempogram(
        onset_envelope=spectral_flux, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )

    # we just use librosa's implementation to get estimated tempo
    # but we still calcaulate tempogram and autocorrelation to plot and understand the approach
    estimated_tempo = librosa.feature.rhythm.tempo(
        onset_envelope=spectral_flux, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )[0]


    _fig = go.Figure()

    _fig.add_trace(
        go.Heatmap(
            z=tempogram,
            x=xs,
        )
    )

    for _label, _tempo in [
        ("Inferred", inferred_tempo),
        ("Estimated", estimated_tempo),
    ]:
        _fig.add_hline(y=_tempo, line_width=1, line_dash="dash")
        _fig.add_annotation(
            x=xs[-1],
            y=_tempo,
            text=f"{_label} tempo ≈ {_tempo:.1f} BPM",
            showarrow=False,
            yanchor="bottom",
            xanchor="right",
            bgcolor="rgba(0,0,0,0.5)",
        )


    # Set axis labels
    _fig.update_layout(
        title="Tempogram",
        xaxis_title="Time (s)",
        yaxis_title="BPM",
    )
    return estimated_tempo, tempogram


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Getting the autocorrelation of the spectral flux. Peaks in the first plot are time lags where the spectral flux pattern most closely matches itself, which corresponding to beat intervals. The second plot is on a BPM (frequency) scale instead of lag time (period). Peaks on the this plot is usually multiples and factors of the actual tempo. Estimated tempo is inferred from where the peaks are and also what are the most common BPM that we see in music (usually between 30 - 300 BPM).""")
    return


@app.cell
def _(
    HOP_LENGTH,
    SAMPLE_RATE,
    estimated_tempo,
    go,
    inferred_tempo,
    librosa,
    make_subplots,
    np,
    px,
    spectral_flux,
    tempogram,
):
    ac_global = librosa.autocorrelate(spectral_flux, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    ac_local = np.mean(tempogram, axis=1)

    _xs_lag_time = x = np.linspace(
        0,
        tempogram.shape[0] * float(HOP_LENGTH) / SAMPLE_RATE,
        num=tempogram.shape[0],
    )
    _xs_bpm_freqs = librosa.tempo_frequencies(
        tempogram.shape[0], hop_length=HOP_LENGTH, sr=SAMPLE_RATE
    )[1:]

    # Create subplots
    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=["Autocorrelation (Time)", "Autocorrelation (Tempo)"],
    )

    _colors = px.colors.qualitative.Plotly
    for y, name, color in zip(
        [ac_global, ac_local],
        ["Global autocorrelation", "Local autocorrelation"],
        _colors[:2],
    ):
        _fig.add_trace(
            go.Scatter(
                x=_xs_lag_time, y=y, name=name, showlegend=True, line_color=color, legendgroup=name
            ),
            row=1,
            col=1,
        )

    for y, name, color in zip(
        [ac_global[1:], ac_local[1:]],
        ["Global autocorrelation", "Local autocorrelation"],
        _colors[:2],
    ):
        _fig.add_trace(
            go.Scatter(
                x=_xs_bpm_freqs, y=y, name=name, showlegend=False, line_color=color, legendgroup=name
            ),
            row=2,
            col=1,
        )

    # Set log scale for BPM axis
    _fig.update_xaxes(type="log", title_text="Tempo (BPM)", row=2, col=1)


    for _label, _tempo in [
        ("Inferred", inferred_tempo),
        ("Estimated", estimated_tempo),
    ]:
        _fig.add_vline(
            type="line",
            row=2,
            col=1,
            x=_tempo,
            line_width=1,
            line_dash="dash",
        )
        _fig.add_annotation(
            x=np.log10(_tempo),
            y=1,
            row=2,
            col=1,
            text=f"{_label} tempo <br> {_tempo:.1f} BPM",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0.5)",
        )


    # Set labels
    _fig.update_xaxes(title_text="Lag (s)", row=1, col=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Estimate beats

        From the spectral flux and the estimated tempo, we can estimated beat locations using a dynamic programming approach described in section 6.3.2 of

        Müller, M. (2021). _Fundamentals of Music Processing_. https://doi.org/10.1007/978-3-030-69808-9

        If finds an optimal sequence of beat locations that align with the expected periodicity of the rhythm. A Gaussian window is used to emphasize regions with periodic onsets, and a penalty function ensures that the intervals between detected beats are consistent with the estimated tempo. The algorithm traces back through the computed scores to reconstruct the most likely sequence of beat locations.
        """
    )
    return


@app.cell
def _(HOP_LENGTH, SAMPLE_RATE, librosa, np):
    def beat_track_dp(spectral_flux, estimated_tempo, tightness=100, alpha=0.5):
        period = (60 * SAMPLE_RATE) / (estimated_tempo * HOP_LENGTH)  # in number of frames

        # Convolves spectral flux with a Gaussian window (std=period) to emphasize regions with periodic onsets.
        gaussian_window = np.exp(-0.5 * (np.arange(-period, period + 1) * 32.0 / period) ** 2)
        local_score = np.convolve(gaussian_window, spectral_flux, mode="same")

        # cumulative_score[i] = best score of any beat‐chain ending exactly at frame i.
        cumulative_score = np.zeros_like(local_score)

        # backlink[i] = which earlier frame gave that best score, so we can trace back later.
        backlink = np.zeros_like(local_score, dtype=int)

        # Search range for previous beat
        # previous 2 periods -> 0.5 periods
        search_window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

        # penalty[i] = gaussian-like penalty on how far the candidate interval deviates from the ideal period, controlled by tightness. 
        # Deviation is -search_window[i] The closer deviation is to period, the smaller the penalty.
        penalty = -tightness * (np.log(-search_window / period) ** 2)

        is_first_beat = True
        # DP loop
        for frame, score in enumerate(local_score):
            # Are we reaching back before time 0?
            z_pad = max(0, min(-search_window[0], len(search_window)))

            # Score for all possible predecessors
            candidate_scores = np.pad(
                cumulative_score[search_window[z_pad:]],
                (z_pad, 0),
            ) + penalty

            # Find the best preceding beat
            beat_location = np.argmax(candidate_scores)

            # Add the local score
            cumulative_score[frame] = (1 - alpha) * score + alpha * candidate_scores[
                beat_location
            ]

            # Special case the first onset.  Stop if the localscore is small
            if is_first_beat and score < 0.01 * local_score.max():
                backlink[frame] = -1
            else:
                backlink[frame] = search_window[beat_location]
                is_first_beat = False

            # Update the time range
            search_window += 1

        beats = [librosa.beat.__last_beat(cumulative_score)]  # last local max from cumulative_score

        # Reconstruct the beat path from backlinks
        while backlink[beats[-1]] >= 0:
            beats.append(backlink[beats[-1]])

        # Put the beats in ascending order
        # Convert into an array of frame numbers
        beats = np.array(beats[::-1], dtype=int)

        # Discard spurious trailing beats
        # beats = librosa.beat.__trim_beats(spectral_flux, beats, True)

        # Convert beat times seconds
        beats = librosa.frames_to_time(beats, hop_length=HOP_LENGTH, sr=SAMPLE_RATE)

        return beats
    return (beat_track_dp,)


@app.cell
def _(
    annotated_beats_lines,
    beat_track_dp,
    create_beat_lines,
    estimated_tempo,
    go,
    px,
    spectral_flux,
    start,
    xs,
):
    estimated_beats = beat_track_dp(spectral_flux, estimated_tempo) + start
    _fig = go.Figure()

    _fig.add_trace(
        go.Scatter(
            x=xs,
            y=spectral_flux,
            mode="lines",
            name="Spectral Flux",
            line_color=px.colors.qualitative.Plotly[0],
            line_width=1,
        ),
    )

    estimated_beat_liens = create_beat_lines(estimated_beats)
    for _line in estimated_beat_liens:
        _fig.add_vline(
            name="Estimated beat",
            line_color=px.colors.qualitative.Plotly[1],
            **_line,
        )
    for _line in annotated_beats_lines:
        _fig.add_vline(
            name="Annotated beat",
            line_color=px.colors.qualitative.Plotly[2],
            **_line,
        )
    _fig.update_layout(
        title="Spectral Flux with estimated and annotated beats",
        xaxis_title="Time (s)",
        yaxis_title="Flux",
    )
    return (estimated_beats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Evaluate our estimates""")
    return


@app.cell
def _(annotated_beats, estimated_beats, mir_eval):
    mir_eval.beat.validate(annotated_beats, estimated_beats)
    return


@app.cell
def _(annotated_beats, estimated_beats, mir_eval):
    CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(annotated_beats, estimated_beats)
    AMLt
    return


if __name__ == "__main__":
    app.run()
