

import marimo

__generated_with = "0.13.2"
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

    from beatdetect.histogram_params import N_MELS
    return N_MELS, TCN, mo, nn, torch, torchinfo, torchview


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This notebook designs the model architecture for beat detection in music, using temporal convolutional networks (TCNs). It takes a mel spectrogram and spectral flux as input, and outputs a sequence of beat confidence scores for each frame in the song. Specifically, we have two TCNs:

        - `tcn1` processes the mel spectrogram. The output is a 15-dimensional feature representation for each time step. The hope is that each feature represent information about a certain instrument. It has 5 layers so that its receptive field is around 1 second (assuming 50 frames per second).
        - The output from `tcn1` is concatenated with the spectral flux, producing a 16-channel input for tcn2. With 8 layers, its receptive field spans approximately 10 seconds, allowing the model to reason over phrase-level rhythmic structures.

        - `tcn2`'s input is the output of `tcn1` concatenated with the spectral flux. The output is a single confidence score per frame indicating the likelihood of a beat at that time. It has 8 layers so its receptive field is around 10 seconds.

        Each of these TCNs also have dropout, weight norm, and skip connections.

        Finally, a 1×1 convolutional layer (logit_head) reduces the 16-dimensional representation to a single logit per time step, which is passed through a sigmoid during inference to yield per-frame beat confidence values.

        """
    )
    return


@app.cell
def _(N_MELS):
    batch_size, T = 4, 3000
    kernel_size = 3
    melspect_shape = (batch_size, N_MELS, T)
    spectral_flux_shape = (batch_size, T)

    tcn1_channels = [80, 55, 35, 25, 15]
    tcn2_channels = [16] * 8
    return (
        kernel_size,
        melspect_shape,
        spectral_flux_shape,
        tcn1_channels,
        tcn2_channels,
    )


@app.cell
def _(N_MELS, TCN, kernel_size, nn, tcn1_channels, tcn2_channels, torch):
    class BeatDetectTCN(nn.Module):
        def __init__(self):
            super().__init__()

            self.tcn1 = TCN(
                num_inputs=N_MELS,
                num_channels=tcn1_channels,
                kernel_size=kernel_size,
                dropout=0.1,
                causal=False,
                use_norm="weight_norm",
                activation="relu",
                use_skip_connections=True
            )

            self.tcn2 = TCN(
                num_inputs=16,
                num_channels=tcn2_channels,
                kernel_size=kernel_size,
                dropout=0.1,
                causal=False,
                use_norm="weight_norm",
                activation="relu",
                use_skip_connections=True,
            )
            self.logit_head = nn.Conv1d(16, 1, kernel_size=1, bias=True)


        def forward(self, mel, flux, return_logits=False):
            """
            mel:  (B, N_MELS, T)
            flux: (B,   T)
            """
            x = self.tcn1(mel)               # → (B, 15, T)

            flux = flux.unsqueeze(1)         # → (B, 1, T)
            x = torch.cat([x, flux], dim=1)  # → (B, 16, T)

            x = self.tcn2(x)                 # → (B, 16, T)

            logits = self.logit_head(x)      # → (B, 1, T)

            if return_logits:
                return logits
            return torch.sigmoid(logits)     # inference -> probabilities

    return (BeatDetectTCN,)


@app.cell
def _(BeatDetectTCN, melspect_shape, spectral_flux_shape, torchview):
    model = BeatDetectTCN()
    model_graph = torchview.draw_graph(model, input_size=(melspect_shape, spectral_flux_shape))
    return model, model_graph


@app.cell
def _(mo, model_graph):
    mo.Html(model_graph.visual_graph.pipe(format="svg").decode('utf-8'))
    return


@app.cell
def _(melspect_shape, model, spectral_flux_shape, torchinfo):
    torchinfo.summary(model, input_size=(melspect_shape, spectral_flux_shape), verbose=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Size of receptive field for TCNs""")
    return


@app.cell
def _(kernel_size, tcn1_channels, tcn2_channels):
    def receptive_field_duration(kernel_size, num_layers, dilation_base=2, fps=50):
        return (1 + (kernel_size - 1) * (dilation_base ** num_layers - 1) / (dilation_base - 1)) / fps
    print("TCN #1: {}s".format(receptive_field_duration(kernel_size, num_layers=len(tcn1_channels))))
    print("TCN #2: {}s".format(receptive_field_duration(kernel_size, num_layers=len(tcn2_channels))))

    return


if __name__ == "__main__":
    app.run()
