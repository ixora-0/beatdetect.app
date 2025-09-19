import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import random

    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchinfo
    import torchview
    from pytorch_tcn import TCN

    from beatdetect.config_loader import load_config
    from beatdetect.model import BeatDetectTCN
    return BeatDetectTCN, load_config, mo, torchinfo, torchview


@app.cell
def _(load_config):
    config = load_config()
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook designs the model architecture for beat detection in music, using temporal convolutional networks (TCNs). It takes a mel spectrogram and spectral flux as input, and outputs a sequence of beat confidence scores for each frame in the song. Specifically, we have two TCNs:

    - `tcn1` processes the mel spectrogram. Its output is a 15-dimensional feature vector at each frame. With 5 layers, the receptive field is about 1 second (assuming 50 fps), so these features can capture local rhythmic events such as instrument onsets. The output of `tcn1` is concatenated with the spectral flux feature, producing a 16-channel sequence. This becomes the input to `tcn2`, which has 8 layers and a receptive field of roughly 10 seconds. This lets the model reason about longer-range rhythmic patterns such as bar- or phrase-level structure.

    - `tcn2` outputs a 16-channel hidden representation, which is then projected by a final 1×1 convolution (logit_head) into two channels: one for beat probability and one for downbeat probability.

    Each TCN block uses dropout, weight normalization, and skip connections. At inference, the two output streams are passed through a sigmoid to yield per-frame confidence values for beats and downbeats.
    """
    )
    return


@app.cell
def _(config):
    batch_size, T = 4, 3000
    melspect_shape = (batch_size, config.spectrogram.n_mels, T)
    spectral_flux_shape = (batch_size, T)
    return melspect_shape, spectral_flux_shape


@app.cell
def _(BeatDetectTCN, config, melspect_shape, spectral_flux_shape, torchview):
    model = BeatDetectTCN(config.spectrogram.n_mels)
    model_graph = torchview.draw_graph(
        model, input_size=(melspect_shape, spectral_flux_shape)
    )
    return model, model_graph


@app.cell
def _(mo, model_graph):
    mo.Html(model_graph.visual_graph.pipe(format="svg").decode("utf-8"))
    return


@app.cell
def _(melspect_shape, model, spectral_flux_shape, torchinfo):
    torchinfo.summary(
        model, input_size=(melspect_shape, spectral_flux_shape), verbose=0
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Size of receptive field for TCNs""")
    return


@app.cell
def _(kernel_size, tcn1_channels, tcn2_channels):
    def receptive_field_duration(kernel_size, num_layers, dilation_base=2, fps=50):
        return (
            1
            + (kernel_size - 1)
            * (dilation_base**num_layers - 1)
            / (dilation_base - 1)
        ) / fps


    print(
        "TCN #1: {}s".format(
            receptive_field_duration(kernel_size, num_layers=len(tcn1_channels))
        )
    )
    print(
        "TCN #2: {}s".format(
            receptive_field_duration(kernel_size, num_layers=len(tcn2_channels))
        )
    )
    return


if __name__ == "__main__":
    app.run()
