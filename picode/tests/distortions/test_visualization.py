"""Tests for visualization helpers."""

import torch

from picode.distortions.native import GaussianNoise
from picode.distortions.visualization import create_comparison, create_diff, create_intensity_grid


class TestCreateDiff:
    """Tests for create_diff function."""

    def test_output_shape(self, sample_image):
        """Diff should have same shape as inputs."""
        distorted = sample_image + torch.randn_like(sample_image) * 0.1
        diff = create_diff(sample_image, distorted)
        assert diff.shape == sample_image.shape

    def test_amplification(self, sample_image):
        """Diff should be amplified for visibility."""
        distorted = sample_image + 0.01  # Small difference
        diff = create_diff(sample_image, distorted, amplify=10.0)
        expected_diff = (distorted - sample_image).abs() * 10.0
        torch.testing.assert_close(diff, expected_diff.clamp(0, 1))


class TestCreateComparison:
    """Tests for create_comparison function."""

    def test_output_shape(self, sample_image):
        """Comparison should be 3x wider than input."""
        distorted = sample_image + torch.randn_like(sample_image) * 0.1
        comparison = create_comparison(sample_image, distorted)
        b, c, h, w = sample_image.shape
        assert comparison.shape == (b, c, h, w * 3)


class TestCreateIntensityGrid:
    """Tests for create_intensity_grid function."""

    def test_output_shape(self, sample_image):
        """Grid should be 3x3 layout."""
        distortion = GaussianNoise()
        grid = create_intensity_grid(sample_image, distortion, levels=9)
        b, c, h, w = sample_image.shape
        # 3x3 grid
        assert grid.shape == (b, c, h * 3, w * 3)

    def test_custom_levels(self, sample_image):
        """Should support custom number of levels."""
        distortion = GaussianNoise()
        grid = create_intensity_grid(sample_image, distortion, levels=4)
        b, c, h, w = sample_image.shape
        # 2x2 grid for 4 levels
        assert grid.shape == (b, c, h * 2, w * 2)
