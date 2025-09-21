import torch
import torch.nn as nn
from pytorch_tcn import TCN

from ..config_loader import Config


class BeatDetectTCN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hypers = config.hypers

        self.tcn1 = TCN(
            num_inputs=config.spectrogram.n_mels,
            num_channels=hypers.tcn1.channels,
            kernel_size=hypers.tcn1.kernel_size,
            dilations=hypers.tcn1.dilations,
            dropout=hypers.dropout,
            causal=False,
            use_norm="weight_norm",
            activation="relu",
            use_skip_connections=True,
        )

        self.tcn2 = TCN(
            num_inputs=hypers.tcn1.channels[-1] + 1,  # out of tcn1 + spectral flux
            num_channels=hypers.tcn2.channels,
            kernel_size=hypers.tcn2.kernel_size,
            dilations=hypers.tcn2.dilations,
            dropout=hypers.dropout,
            causal=False,
            use_norm="weight_norm",
            activation="relu",
            use_skip_connections=True,
        )
        # two output channels: [0]=beat, [1]=downbeat
        self.logit_head = nn.Conv1d(16, 2, kernel_size=1, bias=True)

    def forward(self, mel, flux, return_logits=False):
        """
        mel:  (B, N_MELS, T)
        flux: (B, T)
        """
        x = self.tcn1(mel)  # → (B, 15, T)

        flux = flux.unsqueeze(1)  # → (B, 1, T)
        x = torch.cat([x, flux], dim=1)  # → (B, 16, T)

        x = self.tcn2(x)  # → (B, 16, T)

        logits = self.logit_head(x)  # → (B, 2, T)

        if return_logits:
            return logits
        return torch.sigmoid(logits)  # inference -> probabilities
