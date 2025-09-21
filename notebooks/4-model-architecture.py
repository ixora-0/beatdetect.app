import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torchinfo
    import torchview

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
    # Architecture description
    467K total parameters. Two stages:

    - Stage 1 (TCN1): A 16-layer TCN processes the combined mel spectrogram and spectral flux input. Receptive field = 10.86s

    - Stage 2 (TCN2): A 4-layer TCN takes the TCN1 output concatenated with the predicted beats to detect downbeats. Uses more aggressive dilations over a 6.82s receptive field to identify measure-level structure.

    The architecture follows the natural musical hierarchy where individual beats inform downbeat detection. The first network learns onset detection from low-level audio features, the second use both spectral representations and beat context to identify downbeats that mark musical measures.
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
    model = BeatDetectTCN(config)
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
def _(config):
    def receptive_field_duration(
        kernel_size, num_layers, dilations, fps=config.spectrogram.fps
    ):
        w = 1
        for layer in range(num_layers):
            w += (kernel_size - 1) * dilations[layer]
        return w / fps


    print(
        "TCN #1: {}s".format(
            receptive_field_duration(
                config.hypers.tcn1.kernel_size,
                len(config.hypers.tcn1.channels),
                config.hypers.tcn1.dilations,
            )
        )
    )
    print(
        "TCN #2: {}s".format(
            receptive_field_duration(
                config.hypers.tcn2.kernel_size,
                len(config.hypers.tcn2.channels),
                config.hypers.tcn2.dilations,
            )
        )
    )
    return


if __name__ == "__main__":
    app.run()
