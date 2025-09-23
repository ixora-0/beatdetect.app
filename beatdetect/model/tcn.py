import torch
import torch.nn as nn
from pytorch_tcn import TCN

from ..config_loader import Config


class BeatDetectTCN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hypers = config.hypers

        self.tcn = TCN(
            num_inputs=config.spectrogram.n_mels + 1,  # mel spect + spectral flux
            num_channels=hypers.channels,
            kernel_size=hypers.kernel_size,
            dilations=hypers.dilations,
            dropout=hypers.dropout,
            causal=False,
            use_norm="layer_norm",
            activation="relu",
            use_skip_connections=True,
            use_gate=True,
        )

        # two output channels, 0 -> beat, 1 -> downbeat
        self.logit_head = nn.Conv1d(hypers.channels[-1], 2, kernel_size=1, bias=True)

    def forward(self, mel, flux, return_logits=False):
        """
        mel:  (B, N_MELS, T)
        flux: (B, T)
        """
        flux = flux.unsqueeze(1)  # → (B, 1, T)
        x = torch.cat([mel, flux], dim=1)  # (B, 129, T)
        x = self.tcn(x)  # (B, 64, T)
        logits = self.logit_head(x)  # (B, 2, T)

        if return_logits:
            return logits
        return torch.sigmoid(logits)  # inference -> probabilities
