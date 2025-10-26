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

        # 3 output channels:
        # 0 -> beat-only, 1 -> downbeat, 2 -> no beat
        self.logit_head = nn.Conv1d(hypers.channels[-1], 3, kernel_size=1, bias=True)

    def forward(self, mel, flux, return_logits=False):
        """
        mel:  (B, N_MELS, T)
        flux: (B, T)
        """
        flux = flux.unsqueeze(1)  # → (B, 1, T)
        x = torch.cat([mel, flux], dim=1)  # (B, N_MELS, T)
        x = self.tcn(x)  # (B, C, T)
        logits = self.logit_head(x)  # (B, 3, T)

        if return_logits:
            return logits
        return torch.softmax(logits, dim=1)  # inference -> probabilities
