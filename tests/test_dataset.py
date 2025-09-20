import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from beatdetect.data import BeatDataset


class TestBeatDataset:
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing"""
        config = Mock()
        config.downloads.datasets = ["dataset1", "dataset2"]

        # Create proper Path structure for config.paths
        config.paths = Mock()
        config.paths.data = Mock()
        config.paths.data.raw = Mock()
        config.paths.data.processed = Mock()
        config.paths.data.interim = Mock()

        # Use real Path objects for all path operations
        config.paths.data.raw.spectrograms = Path("/mock/spectrograms")
        config.paths.data.processed.spectral_flux = Path("/mock/spectral_flux")
        config.paths.data.processed.encoded_beats = Path("/mock/encoded_beats")
        config.paths.data.interim.annotations = Path("/mock/annotations")

        return config

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory structure for testing"""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure matching PathResolver expectations
        (temp_dir / "dataset1" / "beats").mkdir(parents=True)
        (temp_dir / "dataset1" / "downbeats").mkdir(parents=True)
        (temp_dir / "spectrograms" / "dataset1").mkdir(parents=True)
        (temp_dir / "spectral_flux" / "dataset1").mkdir(parents=True)

        # Create sample data files
        sample_beats = torch.rand(1000)  # 1000 time steps
        sample_downbeats = torch.rand(1000)
        sample_flux = torch.rand(1000)

        torch.save(sample_beats, temp_dir / "dataset1" / "beats" / "track1.pt")
        torch.save(sample_downbeats, temp_dir / "dataset1" / "downbeats" / "track1.pt")
        torch.save(sample_flux, temp_dir / "spectral_flux" / "dataset1" / "track1.pt")

        # Create mock spectrogram archive
        sample_spectrogram = np.random.rand(128, 1000)  # 128 mel bins, 1000 time steps
        np.savez(
            temp_dir / "spectrograms" / "dataset1" / "dataset1.npz",
            **{"track1/track": sample_spectrogram},
        )

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_dataset_initialization(self, mock_config):
        """Test that dataset initializes correctly"""
        # Use real PathResolver with properly mocked config paths
        mock_config.downloads.datasets = ["dataset1"]

        # Create mock Path objects for glob results
        mock_file1 = Mock()
        mock_file1.name = "track1.pt"
        mock_file2 = Mock()
        mock_file2.name = "track2.pt"

        # Mock glob to return sample files
        with patch.object(Path, "glob", return_value=[mock_file1, mock_file2]):
            with patch.object(Path, "exists", return_value=True):
                with patch("numpy.load") as mock_np_load:
                    mock_np_load.return_value = Mock()
                    dataset = BeatDataset(mock_config)

                    assert len(dataset.samples) > 0
                    assert all(
                        len(sample) == 2 for sample in dataset.samples
                    )  # (dataset, name) pairs

    def test_target_shape_and_structure(self, mock_config, temp_data_dir):
        """Test that target has correct shape with beats and downbeats"""
        # Setup config paths with real temp directory paths
        mock_config.paths.data.raw.spectrograms = temp_data_dir / "spectrograms"
        mock_config.paths.data.processed.spectral_flux = temp_data_dir / "spectral_flux"
        mock_config.paths.data.processed.encoded_beats = temp_data_dir
        mock_config.downloads.datasets = ["dataset1"]

        dataset = BeatDataset(mock_config)

        if len(dataset) > 0:
            mel, flux, target = dataset[0]

            # Check target shape - should be [2, sequence_length]
            assert target.dim() == 2, f"Target should be 2D, got {target.dim()}D"
            assert target.shape[0] == 2, (
                f"First dimension should be 2, got {target.shape[0]}"
            )

            # Check that beats and downbeats are different tensors
            beats = target[0, :]
            downbeats = target[1, :]
            assert beats.shape == downbeats.shape
            assert torch.is_tensor(beats) and torch.is_tensor(downbeats)


if __name__ == "__main__":
    pytest.main([__file__])
