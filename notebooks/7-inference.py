import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    from beatdetect.model.postprocessing.inference import beam_search
    import torch
    from beatdetect.config_loader import load_config
    from beatdetect.data import BeatDataset
    from beatdetect.model import BeatDetectTCN
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import mir_eval
    return (
        BeatDataset,
        BeatDetectTCN,
        beam_search,
        go,
        load_config,
        make_subplots,
        mir_eval,
        px,
        torch,
    )


@app.cell
def _(load_config, torch):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config, device


@app.cell
def _(BeatDataset, config, device):
    val_dataset = BeatDataset(config, "val", "cpu", datasets=["guitarset"])

    id, mel, flux, target, hasdownbeat = val_dataset[3]
    mel, flux = mel.unsqueeze(0).to(device), flux.unsqueeze(0).to(device)  # add batch
    target = target.numpy()
    annotated_beats = val_dataset.get_annotation(id)
    return annotated_beats, flux, mel, target


@app.cell
def _(BeatDetectTCN, config, device, flux, mel, torch):
    model_name = "model.pt"
    model = BeatDetectTCN(config).to(device)
    model.load_state_dict(
        torch.load(config.paths.models / model_name, map_location=device)
    )
    model.eval()

    with torch.no_grad():
        nn_output = model(mel, flux)  # (1, 3, T)

    nn_output = nn_output.squeeze(0).cpu().numpy()  # (3, T)
    return (nn_output,)


@app.cell
def _(beam_search, config, nn_output):
    predicted_beats, lp = beam_search(config, nn_output)
    return (predicted_beats,)


@app.cell
def _(
    annotated_beats,
    config,
    go,
    make_subplots,
    nn_output,
    predicted_beats,
    px,
    target,
):
    def create_beat_lines(beats, group):
        beat_lines = []
        showed_legend = False
        for t in beats:
            beat_lines.append(
                dict(
                    showlegend=not showed_legend,
                    legendgroup=group,
                    mode="lines",
                    x=[t, t],
                    y=[0, 1],
                )
            )
            showed_legend = True

        return beat_lines


    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Prediciton", "Annotated"],
    )

    _fig.add_trace(
        go.Scatter(y=nn_output[0], mode="lines", name="Pred Beat Prob"),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(y=nn_output[1], mode="lines", name="Pred Downbeat Prob"),
        row=1,
        col=1,
    )


    _showed_legend = {"Downbeat": False, "Beat": False}


    for _line in create_beat_lines(
        [b[0] * config.spectrogram.fps for b in predicted_beats if b[1] != 1],
        "gtd",
    ):
        _fig.add_trace(
            go.Scatter(
                name="Post processed downbeat",
                line_color=px.colors.qualitative.Plotly[3],
                **_line,
            ),
            row=1,
            col=1,
        )

    for _line in create_beat_lines(
        [b[0] * config.spectrogram.fps for b in predicted_beats if b[1] == 1],
        "gtb",
    ):
        _fig.add_trace(
            go.Scatter(
                name="Post processed beat",
                line_color=px.colors.qualitative.Plotly[4],
                **_line,
            ),
            row=1,
            col=1,
        )

    _fig.add_trace(
        go.Scatter(
            y=target[0],
            mode="lines",
            name="GT Beat",
            line_color=px.colors.qualitative.Plotly[0],
        ),
        row=2,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            y=target[1],
            mode="lines",
            name="GT Downbeat",
            line_color=px.colors.qualitative.Plotly[1],
        ),
        row=2,
        col=1,
    )

    for _line in create_beat_lines(
        [b[0] * config.spectrogram.fps for b in annotated_beats if b[1] != 1], "ppd"
    ):
        _fig.add_trace(
            go.Scatter(
                name="GT downbeat",
                line_color=px.colors.qualitative.Plotly[3],
                **_line,
            ),
            row=2,
            col=1,
        )

    for _line in create_beat_lines(
        [b[0] * config.spectrogram.fps for b in annotated_beats if b[1] == 1], "ppb"
    ):
        _fig.add_trace(
            go.Scatter(
                name="GT beat",
                line_color=px.colors.qualitative.Plotly[4],
                **_line,
            ),
            row=2,
            col=1,
        )


    _fig
    return


@app.cell
def _(annotated_beats, mir_eval, predicted_beats):
    CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(annotated_beats[:, 0], predicted_beats[:, 0])
    CMLc, CMLt, AMLc, AMLt
    return


if __name__ == "__main__":
    app.run()
