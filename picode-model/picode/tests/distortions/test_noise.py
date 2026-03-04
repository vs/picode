"""Tests for noise distortions."""

import torch


class TestGaussianNoise:
    """Tests for GaussianNoise distortion."""

    def test_preserves_shape(self, distortion_module, sample_image):
        """Output shape should match input shape."""
        distortion = distortion_module.GaussianNoise()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        """Output should be clamped to [0, 1]."""
        distortion = distortion_module.GaussianNoise(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, distortion_module, sample_image):
        """Zero intensity should return input unchanged."""
        distortion = distortion_module.GaussianNoise(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, distortion_module, sample_image):
        """Gradients should flow through."""
        distortion = distortion_module.GaussianNoise()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_different_outputs_each_call(self, distortion_module, sample_image):
        """Each forward pass should produce different noise."""
        distortion = distortion_module.GaussianNoise(intensity=0.5)
        output1 = distortion(sample_image)
        output2 = distortion(sample_image)
        assert not torch.allclose(output1, output2)

    def test_batch_processing(self, distortion_module, batch_images):
        """Should handle batched inputs."""
        distortion = distortion_module.GaussianNoise()
        output = distortion(batch_images)
        assert output.shape == batch_images.shape

    def test_std_parameter(self, distortion_module, sample_image):
        """Higher std should produce more noise."""
        low_std = distortion_module.GaussianNoise(intensity=1.0, std=0.01)
        high_std = distortion_module.GaussianNoise(intensity=1.0, std=0.1)

        torch.manual_seed(42)
        out_low = low_std(sample_image)
        torch.manual_seed(42)
        out_high = high_std(sample_image)

        diff_low = (out_low - sample_image).abs().mean()
        diff_high = (out_high - sample_image).abs().mean()
        assert diff_high > diff_low
