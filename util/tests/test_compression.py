"""Tests for compression distortions."""

import torch

from distortions.compression import JPEGCompression


class TestJPEGCompression:
    """Tests for JPEGCompression distortion."""

    def test_preserves_shape(self, sample_image):
        distortion = JPEGCompression()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        distortion = JPEGCompression(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, sample_image):
        distortion = JPEGCompression(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        distortion = JPEGCompression()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_low_quality_more_artifacts(self, sample_image):
        """Lower quality should produce more deviation from original."""
        high_q = JPEGCompression(intensity=1.0, quality=90)
        low_q = JPEGCompression(intensity=1.0, quality=10)

        out_high = high_q(sample_image)
        out_low = low_q(sample_image)

        diff_high = (out_high - sample_image).abs().mean()
        diff_low = (out_low - sample_image).abs().mean()

        assert diff_low > diff_high

    def test_batch_processing(self, batch_images):
        """Should handle batched inputs."""
        distortion = JPEGCompression()
        output = distortion(batch_images)
        assert output.shape == batch_images.shape
