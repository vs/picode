"""Tests for blur distortions."""

import math

import torch

from picode.distortions.native.blur import GaussianBlur, MotionBlur, RandomBlur


class TestRandomBlur:
    """Tests for RandomBlur distortion (StegaStamp's random_blur_kernel)."""

    def test_preserves_shape(self, sample_image):
        """Output shape should match input shape."""
        distortion = RandomBlur(intensity=1.0)
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        """Output should be in [0, 1]."""
        distortion = RandomBlur(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, sample_image):
        """Zero intensity should return input unchanged."""
        distortion = RandomBlur(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        """Gradients should flow through."""
        distortion = RandomBlur(intensity=1.0)
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None


class TestGaussianBlur:
    """Tests for GaussianBlur distortion."""

    def test_preserves_shape(self, sample_image):
        """Output shape should match input shape."""
        distortion = GaussianBlur()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        """Output should be in [0, 1]."""
        distortion = GaussianBlur(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, sample_image):
        """Zero intensity should return input unchanged."""
        distortion = GaussianBlur(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        """Gradients should flow through."""
        distortion = GaussianBlur()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_blur_reduces_variance(self, sample_image):
        """Blurring should reduce local variance."""
        distortion = GaussianBlur(intensity=1.0, kernel_size=7, sigma=2.0)
        output = distortion(sample_image)

        input_var = sample_image.var()
        output_var = output.var()
        assert output_var < input_var


class TestMotionBlur:
    """Tests for MotionBlur distortion."""

    def test_preserves_shape(self, sample_image):
        """Output shape should match input shape."""
        distortion = MotionBlur()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        """Output should be in [0, 1]."""
        distortion = MotionBlur(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, sample_image):
        """Zero intensity should return input unchanged."""
        distortion = MotionBlur(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        """Gradients should flow through."""
        distortion = MotionBlur()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_angle_parameter(self, sample_image):
        """Different angles should produce different results."""
        blur_0 = MotionBlur(intensity=1.0, angle=0.0)
        blur_90 = MotionBlur(intensity=1.0, angle=math.pi / 2)
        out_0 = blur_0(sample_image)
        out_90 = blur_90(sample_image)
        assert not torch.allclose(out_0, out_90)
