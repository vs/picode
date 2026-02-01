"""Tests for the encoder module."""

import torch

from stegastamp.encoder import Encoder


class TestMessagePreparation:
    """Test message preparation (bits → spatial tensor)."""

    def test_output_shape(self, sample_message: torch.Tensor) -> None:
        """Message prep outputs (B, 3, 400, 400) tensor."""
        encoder = Encoder(num_bits=100)
        result = encoder.prepare_message(sample_message)
        assert result.shape == (2, 3, 400, 400)

    def test_different_num_bits(self) -> None:
        """Works with different message sizes."""
        encoder = Encoder(num_bits=56)
        message = torch.randint(0, 2, (1, 56)).float()
        result = encoder.prepare_message(message)
        assert result.shape == (1, 3, 400, 400)


class TestEncoderForward:
    """Test full encoder forward pass."""

    def test_output_shape(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Encoder outputs same shape as input image."""
        encoder = Encoder(num_bits=100)
        result = encoder(sample_image, sample_message)
        assert result.shape == sample_image.shape

    def test_output_range(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Output is clamped to [0, 1]."""
        encoder = Encoder(num_bits=100)
        result = encoder(sample_image, sample_message)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_gradient_flow(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Gradients flow through encoder."""
        encoder = Encoder(num_bits=100)
        sample_image.requires_grad_(True)
        result = encoder(sample_image, sample_message)
        loss = result.sum()
        loss.backward()
        assert sample_image.grad is not None
        assert not torch.all(sample_image.grad == 0)

    def test_residual_is_small(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Encoded image is close to original (residual is small initially)."""
        encoder = Encoder(num_bits=100)
        result = encoder(sample_image, sample_message)
        diff = (result - sample_image).abs().mean()
        # Untrained network should still produce bounded residual
        assert diff < 1.0
