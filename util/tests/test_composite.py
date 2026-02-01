"""Tests for composite distortions."""

import torch

from distortions import Compose, GaussianNoise, MotionBlur


class TestCompose:
    """Tests for Compose distortion chain."""

    def test_empty_compose(self, sample_image):
        """Empty compose should return input unchanged."""
        compose = Compose([])
        output = compose(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_single_distortion(self, sample_image):
        """Single distortion should work."""
        compose = Compose([GaussianNoise(intensity=0.5)])
        output = compose(sample_image)
        assert output.shape == sample_image.shape

    def test_multiple_distortions(self, sample_image):
        """Multiple distortions should be applied in order."""
        compose = Compose([
            GaussianNoise(intensity=0.3),
            MotionBlur(intensity=0.3),
        ])
        output = compose(sample_image)
        assert output.shape == sample_image.shape

    def test_preserves_shape(self, sample_image):
        compose = Compose([GaussianNoise(), MotionBlur()])
        output = compose(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image):
        compose = Compose([GaussianNoise(intensity=1.0), MotionBlur(intensity=1.0)])
        output = compose(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_gradient_flow(self, sample_image):
        compose = Compose([GaussianNoise(), MotionBlur()])
        image = sample_image.clone().requires_grad_(True)
        output = compose(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_is_module(self):
        """Compose should be a torch.nn.Module."""
        compose = Compose([GaussianNoise()])
        assert isinstance(compose, torch.nn.Module)
