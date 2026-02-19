"""Tests for Picode decoder."""

import pytest
import torch

from picode.models.picode.decoder import Decoder


class TestDecoder:
    """Tests for Picode Decoder."""

    @pytest.fixture
    def decoder(self) -> Decoder:
        """Create decoder instance."""
        return Decoder(num_bits=100)

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Sample input image."""
        return torch.rand(2, 3, 400, 400)

    def test_output_shape(self, decoder: Decoder, sample_image: torch.Tensor) -> None:
        """Output shape is (batch, num_bits)."""
        logits = decoder(sample_image)
        assert logits.shape == (2, 100)

    def test_output_is_logits(self, decoder: Decoder, sample_image: torch.Tensor) -> None:
        """Output is unbounded logits, not probabilities."""
        logits = decoder(sample_image)
        # Logits can be any value, not constrained to [0, 1]
        # Just verify it's not all zeros
        assert logits.abs().mean() > 0

    def test_gradient_flow_to_early_layers(
        self, decoder: Decoder, sample_image: torch.Tensor
    ) -> None:
        """Gradients reach early layers (not vanishing)."""
        sample_image.requires_grad_(True)
        logits = decoder(sample_image)
        logits.sum().backward()

        # First conv layer should receive gradients
        first_conv = decoder.stem[0]
        assert first_conv.weight.grad is not None
        assert first_conv.weight.grad.abs().mean() > 1e-8

    def test_stn_receives_gradients_step_one(
        self, decoder: Decoder, sample_image: torch.Tensor
    ) -> None:
        """STN conv layers receive gradients from step 1 (not zero-init problem)."""
        logits = decoder(sample_image)
        logits.sum().backward()

        # STN first conv should have non-zero gradients
        stn_first_conv = decoder.stn_params[0]
        assert stn_first_conv.weight.grad is not None
        assert stn_first_conv.weight.grad.abs().mean() > 1e-10

    def test_decode_method(self, decoder: Decoder, sample_image: torch.Tensor) -> None:
        """decode() returns binary predictions."""
        bits = decoder.decode(sample_image)
        assert bits.shape == (2, 100)
        # All values should be 0 or 1
        assert ((bits == 0) | (bits == 1)).all()

    def test_different_num_bits(self) -> None:
        """Works with different message lengths."""
        for num_bits in [50, 100, 200]:
            decoder = Decoder(num_bits=num_bits)
            x = torch.rand(1, 3, 400, 400)
            logits = decoder(x)
            assert logits.shape == (1, num_bits)
