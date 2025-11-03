from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from beatdetect.data import BeatDataset


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.downloads.datasets = ["dataset1", "dataset2"]
    config.paths.data.raw.spectrograms = Path("/mock/spectrograms")
    config.paths.data.processed.spectral_flux = Path("/mock/spectral_flux")

    # Mock the splits_info path and its exists method
    splits_info_mock = Mock()
    config.paths.data.processed.splits_info = splits_info_mock

    return config


@pytest.fixture
def sample_splits_csv():
    """Simple CSV content for testing."""
    return """dataset,name,split
dataset1,track1,train
dataset2,track2,val"""


def test_beat_dataset_initialization(mock_config, sample_splits_csv):
    """Test that BeatDataset initializes without errors."""
    mock_config.paths.data.processed.splits_info.exists.return_value = True

    with patch("builtins.open", mock_open(read_data=sample_splits_csv)):
        with patch("builtins.print"):  # Suppress print output
            # Test basic initialization
            dataset = BeatDataset(mock_config, "train")
            assert len(dataset) >= 0

            # Test with custom datasets
            dataset = BeatDataset(mock_config, "train", datasets=["dataset1"])
            assert len(dataset) >= 0


def test_splits_file_missing_raises_error(mock_config):
    """Test that missing splits file raises FileNotFoundError."""
    mock_config.paths.data.processed.splits_info.exists.return_value = False

    with pytest.raises(FileNotFoundError):
        BeatDataset(mock_config, "train")


@patch("beatdetect.data.dataset.PathResolver")
def test_dataset_output_shapes(
    mock_path_resolver_class, mock_config, sample_splits_csv
):
    """Test that dataset returns tensors with expected shapes."""
    import numpy as np
    import torch

    # Setup config mock
    mock_config.paths.data.processed.splits_info.exists.return_value = True

    # Mock PathResolver instance
    mock_resolver = Mock()
    mock_resolver.spectrograms_file = Path("/mock/spectrograms.npz")
    mock_resolver.encoded_beats_dir = Path("/mock/beats")
    mock_resolver.encoded_downbeats_dir = Path("/mock/downbeats")
    mock_path_resolver_class.return_value = mock_resolver

    # Create mock data with known shapes
    mel_data = np.random.randn(1000, 128).astype(
        np.float32
    )  # Will be transposed to (128, 1000)
    flux_data = torch.randn(1000)
    beats_data = torch.randn(1000)
    downbeats_data = torch.randn(1000)

    with patch("builtins.open", mock_open(read_data=sample_splits_csv)):
        with patch("builtins.print"):
            dataset = BeatDataset(mock_config, "train")

    # Mock the numpy archive as a context manager
    mock_archive = Mock()
    mock_archive.get.return_value = mel_data

    with patch("numpy.load") as mock_np_load:
        mock_np_load.return_value.__enter__.return_value = mock_archive
        mock_np_load.return_value.__exit__.return_value = None

        with patch("torch.load") as mock_torch_load:
            # Return different data based on which file is being loaded
            mock_torch_load.side_effect = [flux_data, beats_data, downbeats_data]

            # Get first sample
            mel, flux, target = dataset[0]

    # Test shapes
    assert mel.shape == (128, 1000), f"Expected mel shape (128, 1000), got {mel.shape}"
    assert flux.shape == (1000,), f"Expected flux shape (1000,), got {flux.shape}"
    assert target.shape == (2, 1000), (
        f"Expected target shape (2, 1000), got {target.shape}"
    )

    # Test types
    assert isinstance(mel, torch.Tensor)
    assert isinstance(flux, torch.Tensor)
    assert isinstance(target, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])
