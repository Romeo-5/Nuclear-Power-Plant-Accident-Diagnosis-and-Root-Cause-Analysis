"""Tests for preprocessing logic."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler


class TestScalerFitting:
    def test_scaler_fit_on_normal_only(self):
        """Verify that the scaler is fit only on normal data."""
        np.random.seed(42)

        # Simulate normal data (mean=100, std=10)
        normal_data = np.random.normal(100, 10, size=(200, 5)).astype(np.float32)
        # Simulate accident data (mean=500, std=50) -- very different
        accident_data = np.random.normal(500, 50, size=(100, 5)).astype(np.float32)

        # Fit scaler on normal data only (correct approach)
        scaler = StandardScaler()
        scaler.fit(normal_data)

        # Transform both
        normal_scaled = scaler.transform(normal_data)
        accident_scaled = scaler.transform(accident_data)

        # Normal data should have ~0 mean and ~1 std
        np.testing.assert_allclose(normal_scaled.mean(axis=0), 0, atol=0.2)
        np.testing.assert_allclose(normal_scaled.std(axis=0), 1, atol=0.2)

        # Accident data should have high mean (since it's anomalous relative to normal)
        assert accident_scaled.mean() > 10  # clearly different from normal

    def test_scaler_fit_on_all_data_is_wrong(self):
        """Show why fitting on all data leaks information."""
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, size=(200, 5)).astype(np.float32)
        accident_data = np.random.normal(500, 50, size=(100, 5)).astype(np.float32)

        all_data = np.concatenate([normal_data, accident_data])

        # Fit on all data (wrong approach)
        scaler_all = StandardScaler()
        scaler_all.fit(all_data)

        # Fit on normal only (correct approach)
        scaler_normal = StandardScaler()
        scaler_normal.fit(normal_data)

        # With all-data scaler, the gap between normal and accident is compressed
        normal_gap_all = np.abs(
            scaler_all.transform(normal_data).mean()
            - scaler_all.transform(accident_data).mean()
        )
        normal_gap_correct = np.abs(
            scaler_normal.transform(normal_data).mean()
            - scaler_normal.transform(accident_data).mean()
        )

        # Normal-only scaler preserves the anomaly signal better
        assert normal_gap_correct > normal_gap_all
