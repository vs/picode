"""Tests for color distortions."""

import torch

from picode.distortions.native.color import BrightnessHue, Contrast, Saturation


class TestBrightnessHue:
    """Tests for BrightnessHue distortion (StegaStamp's get_rnd_brightness_tf)."""

    def test_preserves_shape(self, sample_image):
        distortion = BrightnessHue()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        distortion = BrightnessHue(intensity=1.0, rnd_bri=0.3, rnd_hue=0.1)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, sample_image):
        distortion = BrightnessHue(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        distortion = BrightnessHue()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None


class TestContrast:
    """Tests for Contrast distortion."""

    def test_preserves_shape(self, sample_image):
        distortion = Contrast()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_zero_intensity_unchanged(self, sample_image):
        distortion = Contrast(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        distortion = Contrast()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_contrast_scales_values(self, sample_image):
        """Contrast > 1 should increase range, < 1 should decrease."""
        # Force high contrast
        distortion = Contrast(intensity=1.0, contrast_low=1.5, contrast_high=1.5)
        output = distortion(sample_image)
        # Values should be scaled up (multiplied by 1.5)
        expected = sample_image * 1.5
        torch.testing.assert_close(output, expected)


class TestSaturation:
    """Tests for Saturation distortion."""

    def test_preserves_shape(self, sample_image):
        distortion = Saturation()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        distortion = Saturation(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, sample_image):
        distortion = Saturation(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, sample_image):
        distortion = Saturation()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None
