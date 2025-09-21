import torch
import torch.nn as nn
from pytorch_tcn import TCN

from ..config_loader import Config


class BeatDetectTCN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hypers = config.hypers

        self.tcn1 = TCN(
            num_inputs=config.spectrogram.n_mels + 1,  # mel spect + spectral flux
            num_channels=hypers.tcn1.channels,
            kernel_size=hypers.tcn1.kernel_size,
            dilations=hypers.tcn1.dilations,
            dropout=hypers.dropout,
            causal=False,
            use_norm="weight_norm",
            activation="relu",
            use_skip_connections=True,
        )

        self.beat_head = nn.Conv1d(
            hypers.tcn1.channels[-1], 1, kernel_size=1, bias=True
        )

        self.tcn2 = TCN(
            num_inputs=hypers.tcn1.channels[-1] + 1,  # out of tcn1 + beats
            num_channels=hypers.tcn2.channels,
            kernel_size=hypers.tcn2.kernel_size,
            dilations=hypers.tcn2.dilations,
            dropout=hypers.dropout,
            causal=False,
            use_norm="weight_norm",
            activation="relu",
            use_skip_connections=True,
        )

        self.downbeat_head = nn.Conv1d(
            hypers.tcn2.channels[-1], 1, kernel_size=1, bias=True
        )

    def forward(self, mel, flux, return_logits=False):
        """
        mel:  (B, N_MELS, T)
        flux: (B, T)
        """
        flux = flux.unsqueeze(1)  # → (B, 1, T)
        x = torch.cat([mel, flux], dim=1)  # (B, 129, T)
        x = self.tcn1(x)  # (B, 32, T)
        beat_logits = self.beat_head(x)  # (B, 1, T)

        x = torch.cat([x, beat_logits], dim=1)  # (B, 33, T)
        x = self.tcn2(x)  # → (B, 32, T)
        downbeat_logits = self.downbeat_head(x)  # (B, 1, T)

        logits = torch.cat([beat_logits, downbeat_logits], dim=1)  # (B, 2, T)

        if return_logits:
            return logits
        return torch.sigmoid(logits)  # inference -> probabilities
