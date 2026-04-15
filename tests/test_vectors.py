"""Tests for activation accumulation and vector computation (no GPU needed)."""

import numpy as np
import pytest


class TestRunningMean:
    def test_basic_mean(self):
        from emotion_probes.activations import RunningMean

        rm = RunningMean((4,))
        rm.update(np.array([[1.0, 2.0, 3.0, 4.0]]))
        rm.update(np.array([[3.0, 4.0, 5.0, 6.0]]))
        expected = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(rm.mean, expected)
        assert rm.count == 2

    def test_batch_update(self):
        from emotion_probes.activations import RunningMean

        rm = RunningMean((3,))
        batch = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [3.0, 4.0, 5.0]])
        rm.update(batch)
        expected = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(rm.mean, expected)
        assert rm.count == 3

    def test_empty_raises(self):
        from emotion_probes.activations import RunningMean

        rm = RunningMean((2,))
        with pytest.raises(ValueError, match="No values"):
            rm.mean

    def test_float64_precision(self):
        from emotion_probes.activations import RunningMean

        rm = RunningMean((1,))
        # Add many small values to test precision
        for _ in range(10000):
            rm.update(np.array([[1e-8]]))
        expected = np.array([1e-8], dtype=np.float32)
        np.testing.assert_allclose(rm.mean, expected, rtol=1e-5)


class TestDenoiseWithNeutralPCA:
    def test_projects_out_top_component(self):
        from emotion_probes.vectors import _denoise_with_neutral_pca

        rng = np.random.RandomState(42)
        # Create neutral data dominated by one direction
        dominant_dir = np.array([1.0, 0.0, 0.0, 0.0])
        neutral = rng.randn(100, 4) * 0.1
        neutral[:, 0] += rng.randn(100) * 10  # dominant variance along dim 0

        # Create emotion vectors with component along dominant direction
        vectors = rng.randn(5, 4)
        vectors[:, 0] += 5.0  # add dominant-direction bias

        denoised, info = _denoise_with_neutral_pca(vectors, neutral, 0.5)

        # After denoising, the dominant direction should be removed
        assert info["n_components"] >= 1
        assert info["variance_explained"] >= 0.5
        # Projection onto dominant direction should be near zero
        projections = denoised @ dominant_dir
        assert np.abs(projections).max() < 0.5  # much smaller than original ~5.0

    def test_threshold_boundary(self):
        from emotion_probes.vectors import _denoise_with_neutral_pca

        rng = np.random.RandomState(42)
        # 3 equal-variance dimensions
        neutral = rng.randn(100, 3)

        vectors = rng.randn(5, 3)
        # threshold = 0.5 should need ~2 components (each explains ~1/3)
        _, info = _denoise_with_neutral_pca(vectors, neutral, 0.5)
        assert info["n_components"] == 2  # 1/3 + 1/3 = 2/3 >= 0.5

    def test_preserves_shape(self):
        from emotion_probes.vectors import _denoise_with_neutral_pca

        rng = np.random.RandomState(42)
        neutral = rng.randn(50, 8)
        vectors = rng.randn(10, 8)

        denoised, _ = _denoise_with_neutral_pca(vectors, neutral, 0.5)
        assert denoised.shape == vectors.shape
