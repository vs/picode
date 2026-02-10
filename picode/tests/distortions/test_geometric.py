"""Tests for geometric distortions."""

import torch


class TestPerspectiveWarp:
    """Tests for PerspectiveWarp distortion."""

    def test_preserves_shape(self, distortion_module, sample_image):
        distortion = distortion_module.PerspectiveWarp()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        distortion = distortion_module.PerspectiveWarp(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, distortion_module, sample_image):
        distortion = distortion_module.PerspectiveWarp(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, distortion_module, sample_image):
        distortion = distortion_module.PerspectiveWarp()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None


class TestRotation:
    """Tests for Rotation distortion."""

    def test_preserves_shape(self, distortion_module, sample_image):
        distortion = distortion_module.Rotation()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        distortion = distortion_module.Rotation(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, distortion_module, sample_image):
        distortion = distortion_module.Rotation(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, distortion_module, sample_image):
        distortion = distortion_module.Rotation()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None


class TestScale:
    """Tests for Scale distortion."""

    def test_preserves_shape(self, distortion_module, sample_image):
        distortion = distortion_module.Scale()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        distortion = distortion_module.Scale(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, distortion_module, sample_image):
        distortion = distortion_module.Scale(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, distortion_module, sample_image):
        distortion = distortion_module.Scale()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None


class TestCrop:
    """Tests for Crop distortion."""

    def test_preserves_shape(self, distortion_module, sample_image):
        distortion = distortion_module.Crop()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        distortion = distortion_module.Crop(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, distortion_module, sample_image):
        distortion = distortion_module.Crop(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, distortion_module, sample_image):
        distortion = distortion_module.Crop()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None
