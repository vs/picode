"""Tests for base distortion class."""

import torch

from distortions.base import Distortion


class ConcreteDistortion(Distortion):
    """Concrete implementation for testing."""

    name = "test_distortion"

    def __init__(self, intensity: float = 0.5):
        super().__init__(intensity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple distortion: add scaled noise
        noise = torch.randn_like(x) * 0.1 * self.intensity
        return torch.clamp(x + noise, 0.0, 1.0)

    def sample_parameters(self) -> dict:
        return {"noise_scale": torch.rand(1).item() * self.intensity}


def test_distortion_is_module():
    """Distortion should be a torch.nn.Module."""
    distortion = ConcreteDistortion()
    assert isinstance(distortion, torch.nn.Module)


def test_distortion_has_intensity():
    """Distortion should store intensity parameter."""
    distortion = ConcreteDistortion(intensity=0.7)
    assert distortion.intensity == 0.7


def test_distortion_default_intensity():
    """Default intensity should be 0.5."""
    distortion = ConcreteDistortion()
    assert distortion.intensity == 0.5


def test_distortion_forward_preserves_shape(sample_image):
    """Forward pass should preserve input shape."""
    distortion = ConcreteDistortion()
    output = distortion(sample_image)
    assert output.shape == sample_image.shape


def test_distortion_output_in_valid_range(sample_image):
    """Output should be in [0, 1] range."""
    distortion = ConcreteDistortion(intensity=1.0)
    output = distortion(sample_image)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_distortion_zero_intensity_returns_input(sample_image):
    """Zero intensity should return input unchanged."""
    distortion = ConcreteDistortion(intensity=0.0)
    output = distortion(sample_image)
    torch.testing.assert_close(output, sample_image)


def test_distortion_gradient_flow(sample_image):
    """Gradients should flow through distortion."""
    distortion = ConcreteDistortion()
    image = sample_image.clone().requires_grad_(True)
    output = distortion(image)
    loss = output.mean()
    loss.backward()
    assert image.grad is not None


def test_distortion_set_parameters():
    """set_parameters should update internal state."""
    distortion = ConcreteDistortion()
    distortion.set_parameters(intensity=0.8)
    assert distortion.intensity == 0.8
