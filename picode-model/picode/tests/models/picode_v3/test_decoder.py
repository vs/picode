"""Tests for Picode v3 decoder."""

from typing import cast

import pytest
import torch
import torch.nn as nn

from picode.models.picode_v3.decoder import Decoder


class TestDecoder:
    """Tests for Picode v3 Decoder."""

    @pytest.fixture
    def decoder(self) -> Decoder:
        """Create decoder instance with default 127 bits (BCH codeword)."""
        return Decoder(num_bits=127)

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Sample input image."""
        return torch.rand(2, 3, 400, 400)

    def test_output_shape(self, decoder: Decoder, sample_image: torch.Tensor) -> None:
        """Output shape is (batch, num_bits)."""
        logits = decoder(sample_image)
        assert logits.shape == (2, 127)

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

        # First conv in decoder sequential should receive gradients
        first_conv = cast(nn.Conv2d, decoder.decoder[0])
        grad = first_conv.weight.grad
        assert grad is not None
        assert grad.abs().mean() > 1e-8

    def test_stn_has_identity_initialization(
        self, decoder: Decoder
    ) -> None:
        """STN is initialized to identity transform."""
        # STN FC bias should be [1, 0, 0, 0, 1, 0] (identity affine)
        expected_bias = torch.tensor([1., 0., 0., 0., 1., 0.])
        assert torch.allclose(decoder.stn_fc_bias, expected_bias)
        # STN FC weight should be zero (so output = bias = identity)
        assert decoder.stn_fc_weight.abs().max() == 0

    def test_decode_method(self, decoder: Decoder, sample_image: torch.Tensor) -> None:
        """decode() returns binary predictions."""
        bits = decoder.decode(sample_image)
        assert bits.shape == (2, 127)
        # All values should be 0 or 1
        assert ((bits == 0) | (bits == 1)).all()

    def test_default_num_bits_is_127(self) -> None:
        """Default num_bits is 127 (BCH codeword length)."""
        decoder = Decoder()
        assert decoder.num_bits == 127

    def test_different_num_bits(self) -> None:
        """Works with different message lengths."""
        for num_bits in [50, 100, 127, 200]:
            decoder = Decoder(num_bits=num_bits)
            x = torch.rand(1, 3, 400, 400)
            logits = decoder(x)
            assert logits.shape == (1, num_bits)
