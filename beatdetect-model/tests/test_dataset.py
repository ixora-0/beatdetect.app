from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from beatdetect_model.data import BeatDataset


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.random_seed = 0
    config.downloads.datasets = ["dataset1", "dataset2"]
    config.paths.data.raw.spectrograms = Path("/mock/spectrograms")
    config.paths.data.processed.spectral_flux = Path("/mock/spectral_flux")

    # Mock the splits_info path and its exists method
    splits_info_mock = Mock()
    config.paths.data.processed.splits_info = splits_info_mock

    # Mock the datasets_info path
    datasets_info_mock = Mock()
    config.paths.data.processed.datasets_info = datasets_info_mock

    return config


@pytest.fixture
def sample_datasets_info():
    return """
{
  "dataset1": {
    "has_downbeats": true
  },
  "dataset2": {
    "has_downbeats": true
  }
}
"""


@pytest.fixture
def sample_splits_csv():
    """Simple CSV content for testing."""
    return """dataset,name,split
dataset1,track1,train
dataset2,track2,val"""


def test_beat_dataset_initialization(
    mock_config, sample_datasets_info, sample_splits_csv
):
    """Test that BeatDataset initializes without errors."""
    import torch

    mock_config.paths.data.processed.splits_info.exists.return_value = True

    with (
        patch.object(
            mock_config.paths.data.processed.datasets_info,
            "open",
            mock_open(read_data=sample_datasets_info),
        ),
        patch("builtins.open", mock_open(read_data=sample_splits_csv)),
        patch("numpy.load"),  # Mock np.load to avoid actual file access
    ):
        with patch("builtins.print"):  # Suppress print output
            # Test basic initialization
            dataset = BeatDataset(mock_config, "train", torch.device("cpu"))
            assert len(dataset) >= 0

            # Test with custom datasets
            dataset = BeatDataset(
                mock_config, "train", torch.device("cpu"), datasets=["dataset1"]
            )
            assert len(dataset) >= 0


def test_splits_file_missing_raises_error(mock_config):
    """Test that missing splits file raises FileNotFoundError."""
    import torch

    mock_config.paths.data.processed.splits_info.exists.return_value = False

    with pytest.raises(FileNotFoundError):
        BeatDataset(mock_config, "train", torch.device("cpu"))


@patch("beatdetect_model.data.dataset.PathResolver")
def test_dataset_output_shapes(
    mock_path_resolver_class, mock_config, sample_datasets_info, sample_splits_csv
):
    """Test that dataset returns tensors with expected shapes."""
    import numpy as np
    import torch

    # Setup config mock
    mock_config.paths.data.processed.splits_info.exists.return_value = True

    # Mock PathResolver instance
    mock_resolver = Mock()
    mock_resolver.spectrograms_file = Path("/mock/spectrograms.npz")
    mock_resolver.encoded_annotations_dir = Path("/mock/annotations")
    mock_path_resolver_class.return_value = mock_resolver

    # Create mock data with known shapes
    T = 1000
    mel_data = np.random.randn(T, 128).astype(
        np.float32
    )  # Will be transposed to (128, T)
    flux_data = torch.randn(T)
    target_data = torch.randn(3, T)

    with (
        patch.object(
            mock_config.paths.data.processed.datasets_info,
            "open",
            mock_open(read_data=sample_datasets_info),
        ),
        patch("builtins.open", mock_open(read_data=sample_splits_csv)),
    ):
        # Mock the numpy archive
        mock_archive = Mock()
        mock_archive.get.return_value = mel_data

        with patch("numpy.load") as mock_np_load:
            mock_np_load.return_value = mock_archive

            with patch("builtins.print"):
                dataset = BeatDataset(mock_config, "train", torch.device("cpu"))

    with (
        patch("torch.load") as mock_torch_load,
        patch("numpy.load") as mock_np_load,
    ):
        # Mock the numpy archive for __getitem__
        mock_archive = Mock()
        mock_archive.get.return_value = mel_data
        mock_np_load.return_value = mock_archive

        # Mock torch.load to return flux and target data
        mock_torch_load.side_effect = [flux_data, target_data]

        id_str, mel, flux, target, has_downbeat = dataset[0]

    # Test shapes
    assert mel.shape == (128, T), f"Expected mel shape (128, {T}), got {mel.shape}"
    assert flux.shape == (T,), f"Expected flux shape ({T},), got {flux.shape}"
    assert target.shape == (3, T), f"Expected target shape (3, {T}), got {target.shape}"

    # Test types
    assert isinstance(id_str, str)
    assert isinstance(mel, torch.Tensor)
    assert isinstance(flux, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(has_downbeat, bool)


if __name__ == "__main__":
    pytest.main([__file__])
