import torch

from ..config_loader import Config


def create_mock_inputs(config: Config, device: torch.device | None, time_steps=100):
    """Create mock inputs for BeatDetectTCN model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = config.training.batch_size
    n_bins = config.spectrogram.n_mels

    mock_mels = torch.randn(batch_size, n_bins, time_steps, device=device)
    mock_fluxes = torch.randn(batch_size, time_steps, device=device)

    return mock_mels, mock_fluxes
