"""Tests for the decoder module."""

import torch

from picode.models.stegastamp.decoder import Decoder


class TestDecoderForward:
    """Test decoder forward pass."""

    def test_output_shape(self, sample_image: torch.Tensor) -> None:
        """Decoder outputs (B, num_bits) tensor."""
        decoder = Decoder(num_bits=100)
        result = decoder(sample_image)
        assert result.shape == (2, 100)

    def test_output_range(self, sample_image: torch.Tensor) -> None:
        """Output probabilities are in [0, 1]."""
        decoder = Decoder(num_bits=100)
        result = decoder(sample_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_gradient_flow(self, sample_image: torch.Tensor) -> None:
        """Gradients flow through decoder."""
        decoder = Decoder(num_bits=100)
        sample_image.requires_grad_(True)
        result = decoder(sample_image)
        loss = result.sum()
        loss.backward()
        assert sample_image.grad is not None

    def test_different_num_bits(self, sample_image: torch.Tensor) -> None:
        """Works with different message sizes."""
        decoder = Decoder(num_bits=56)
        result = decoder(sample_image)
        assert result.shape == (2, 56)

    def test_decode_method(self, sample_image: torch.Tensor) -> None:
        """Decode method returns binary bits."""
        decoder = Decoder(num_bits=100)
        result = decoder.decode(sample_image)
        assert result.shape == (2, 100)
        assert torch.all((result == 0) | (result == 1))
