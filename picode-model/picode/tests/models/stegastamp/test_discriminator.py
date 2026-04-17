"""Tests for StegaStamp discriminator."""

import pytest
import torch

from picode.models.stegastamp.discriminator import Discriminator


class TestDiscriminator:
    """Tests for StegaStamp discriminator."""

    @pytest.fixture
    def discriminator(self) -> Discriminator:
        """Create discriminator for testing."""
        return Discriminator()

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Create sample image batch."""
        return torch.rand(2, 3, 400, 400)

    def test_output_shape(
        self, discriminator: Discriminator, sample_image: torch.Tensor
    ) -> None:
        """Discriminator outputs scalar per image."""
        output = discriminator(sample_image)
        assert output.shape == (2,)

    def test_gradient_flow(
        self, discriminator: Discriminator, sample_image: torch.Tensor
    ) -> None:
        """Gradients flow through discriminator."""
        sample_image.requires_grad_(True)
        output = discriminator(sample_image)
        output.sum().backward()

        assert sample_image.grad is not None
        assert not torch.all(sample_image.grad == 0)

    def test_different_outputs_for_different_inputs(
        self, discriminator: Discriminator
    ) -> None:
        """Discriminator produces different outputs for different inputs."""
        real = torch.rand(2, 3, 400, 400)
        fake = torch.rand(2, 3, 400, 400)

        out_real = discriminator(real)
        out_fake = discriminator(fake)

        # Outputs should differ (with high probability for random inputs)
        assert not torch.allclose(out_real, out_fake)

    def test_no_normalization_layers(self, discriminator: Discriminator) -> None:
        """Discriminator has no BatchNorm/LayerNorm (WGAN requirement)."""
        for module in discriminator.modules():
            assert not isinstance(
                module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)
            ), f"Found normalization layer: {type(module)}"
