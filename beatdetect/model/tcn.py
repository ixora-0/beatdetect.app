import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN

from ..config_loader import Config

kernel_size = 3
tcn1_channels = [80, 55, 35, 25, 15]
tcn2_channels = [16] * 8


class BeatDetectTCN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.tcn1 = TCN(
            num_inputs=config.spectrogram.n_mels,
            num_channels=tcn1_channels,
            kernel_size=kernel_size,
            dropout=0.1,
            causal=False,
            use_norm="weight_norm",
            activation="relu",
            use_skip_connections=True,
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
        flux: (B, T)
        """
        x = self.tcn1(mel)  # (B, 15, T)

        flux = flux.unsqueeze(1)  # (B, 1, T)
        x = torch.cat([x, flux], dim=1)  # (B, 16, T)

        x = self.tcn2(x)  # (B, 16, T)

        logits = self.logit_head(x)  # (B, 1, T)

        if return_logits:
            return logits
        return torch.sigmoid(logits)  # inference -> probabilities
