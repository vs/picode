"""Tests for composite distortions."""

import torch


class TestCompose:
    """Tests for Compose distortion chain."""

    def test_empty_compose(self, distortion_module, sample_image):
        """Empty compose should return input unchanged."""
        compose = distortion_module.Compose([])
        output = compose(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_single_distortion(self, distortion_module, sample_image):
        """Single distortion should work."""
        compose = distortion_module.Compose([distortion_module.GaussianNoise(intensity=0.5)])
        output = compose(sample_image)
        assert output.shape == sample_image.shape

    def test_multiple_distortions(self, distortion_module, sample_image):
        """Multiple distortions should be applied in order."""
        compose = distortion_module.Compose([
            distortion_module.GaussianNoise(intensity=0.3),
            distortion_module.MotionBlur(intensity=0.3),
        ])
        output = compose(sample_image)
        assert output.shape == sample_image.shape

    def test_preserves_shape(self, distortion_module, sample_image):
        compose = distortion_module.Compose([
            distortion_module.GaussianNoise(),
            distortion_module.MotionBlur(),
        ])
        output = compose(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        compose = distortion_module.Compose([
            distortion_module.GaussianNoise(intensity=1.0),
            distortion_module.MotionBlur(intensity=1.0),
        ])
        output = compose(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_gradient_flow(self, distortion_module, sample_image):
        compose = distortion_module.Compose([
            distortion_module.GaussianNoise(),
            distortion_module.MotionBlur(),
        ])
        image = sample_image.clone().requires_grad_(True)
        output = compose(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_is_module(self, distortion_module):
        """Compose should be a torch.nn.Module."""
        compose = distortion_module.Compose([distortion_module.GaussianNoise()])
        assert isinstance(compose, torch.nn.Module)
