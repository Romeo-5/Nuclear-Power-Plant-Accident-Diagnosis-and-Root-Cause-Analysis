"""Tests for dataset and data loading."""

import pytest
import torch
from pathlib import Path

from src.data.dataset import NPPADWindowDataset
from src.data.transforms import sliding_window

import numpy as np


class TestSlidingWindow:
    def test_basic(self):
        data = np.arange(100).reshape(50, 2).astype(np.float32)
        windows = sliding_window(data, window_size=10, stride=5)
        assert windows.shape == (9, 10, 2)

    def test_stride_1(self):
        data = np.arange(40).reshape(20, 2).astype(np.float32)
        windows = sliding_window(data, window_size=5, stride=1)
        assert windows.shape == (16, 5, 2)

    def test_window_content(self):
        data = np.arange(20).reshape(10, 2).astype(np.float32)
        windows = sliding_window(data, window_size=3, stride=1)
        np.testing.assert_array_equal(windows[0], data[0:3])
        np.testing.assert_array_equal(windows[1], data[1:4])

    def test_exact_fit(self):
        data = np.zeros((30, 5), dtype=np.float32)
        windows = sliding_window(data, window_size=10, stride=10)
        assert windows.shape == (3, 10, 5)


class TestNPPADWindowDataset:
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal dataset file for testing."""
        n_windows = 20
        window_size = 30
        n_features = 10

        data = {
            "windows": torch.randn(n_windows, window_size, n_features),
            "labels": torch.cat([
                torch.zeros(10, dtype=torch.long),
                torch.ones(10, dtype=torch.long),
            ]),
            "accident_types": torch.cat([
                torch.zeros(10, dtype=torch.long),
                torch.ones(10, dtype=torch.long),
            ]),
        }
        path = tmp_path / "test.pt"
        torch.save(data, path)
        return NPPADWindowDataset(path)

    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 20

    def test_getitem_shapes(self, sample_dataset):
        window, label = sample_dataset[0]
        assert window.shape == (30, 10)
        assert label.shape == ()

    def test_properties(self, sample_dataset):
        assert sample_dataset.n_features == 10
        assert sample_dataset.window_size == 30

    def test_normal_only(self, sample_dataset):
        normal = sample_dataset.normal_only()
        assert len(normal) == 10
        assert (normal.labels == 0).all()
