"""Tests for noise distortions."""

import torch

from distortions.noise import GaussianNoise


class TestGaussianNoise:
    """Tests for GaussianNoise distortion."""

    def test_preserves_shape(self, sample_image):
        """Output shape should match input shape."""
        distortion = GaussianNoise()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        """Output should be clamped to [0, 1]."""
        distortion = GaussianNoise(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, sample_image):
        """Zero intensity should return input unchanged."""
        distortion = GaussianNoise(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        """Gradients should flow through."""
        distortion = GaussianNoise()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_different_outputs_each_call(self, sample_image):
        """Each forward pass should produce different noise."""
        distortion = GaussianNoise(intensity=0.5)
        output1 = distortion(sample_image)
        output2 = distortion(sample_image)
        assert not torch.allclose(output1, output2)

    def test_batch_processing(self, batch_images):
        """Should handle batched inputs."""
        distortion = GaussianNoise()
        output = distortion(batch_images)
        assert output.shape == batch_images.shape

    def test_std_parameter(self, sample_image):
        """Higher std should produce more noise."""
        low_std = GaussianNoise(intensity=1.0, std=0.01)
        high_std = GaussianNoise(intensity=1.0, std=0.1)

        torch.manual_seed(42)
        out_low = low_std(sample_image)
        torch.manual_seed(42)
        out_high = high_std(sample_image)

        diff_low = (out_low - sample_image).abs().mean()
        diff_high = (out_high - sample_image).abs().mean()
        assert diff_high > diff_low
