"""Tests for compression distortions."""

import torch


class TestJPEGCompression:
    """Tests for JPEGCompression distortion."""

    def test_preserves_shape(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_low_quality_more_artifacts(self, distortion_module, sample_image):
        """Lower quality should produce more deviation from original."""
        high_q = distortion_module.JPEGCompression(intensity=1.0, quality=90)
        low_q = distortion_module.JPEGCompression(intensity=1.0, quality=10)

        out_high = high_q(sample_image)
        out_low = low_q(sample_image)

        diff_high = (out_high - sample_image).abs().mean()
        diff_low = (out_low - sample_image).abs().mean()

        assert diff_low > diff_high

    def test_batch_processing(self, distortion_module, batch_images):
        """Should handle batched inputs."""
        distortion = distortion_module.JPEGCompression()
        output = distortion(batch_images)
        assert output.shape == batch_images.shape
