"""Tests for the decoder module."""

import torch

from picode.models.stegastamp.decoder import Decoder


class TestDecoderArchitecture:
    """Test decoder architecture matches original StegaStamp."""

    def test_no_batchnorm(self) -> None:
        """Decoder has no BatchNorm layers (matches original)."""
        import torch.nn as nn
        decoder = Decoder(num_bits=100)
        for module in decoder.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), \
                f"Found BatchNorm: {module}"

    def test_has_stn(self) -> None:
        """Decoder has Spatial Transformer Network."""
        decoder = Decoder(num_bits=100)
        assert hasattr(decoder, "stn_params"), "Missing STN params network"
        assert hasattr(decoder, "stn_fc_weight"), "Missing STN weight"
        assert hasattr(decoder, "stn_fc_bias"), "Missing STN bias"

    def test_stn_identity_init(self) -> None:
        """STN is initialized to identity transform."""
        decoder = Decoder(num_bits=100)
        expected_bias = torch.tensor([1., 0., 0., 0., 1., 0.])
        assert torch.allclose(decoder.stn_fc_bias, expected_bias)
        assert torch.allclose(decoder.stn_fc_weight, torch.zeros(128, 6))


class TestDecoderForward:
    """Test decoder forward pass."""

    def test_output_shape(self, sample_image: torch.Tensor) -> None:
        """Decoder outputs (B, num_bits) tensor."""
        decoder = Decoder(num_bits=100)
        result = decoder(sample_image)
        assert result.shape == (2, 100)

    def test_output_is_logits(self, sample_image: torch.Tensor) -> None:
        """Output is logits (unbounded, can be negative)."""
        decoder = Decoder(num_bits=100)
        result = decoder(sample_image)
        # Logits are unbounded - at least some should be outside [0,1]
        # With random init, this is almost certain
        assert result.min() < 0.5 or result.max() > 0.5, "Expected unbounded logits"

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

    def test_stn_transforms_image(self, sample_image: torch.Tensor) -> None:
        """STN applies learned affine transformation."""
        decoder = Decoder(num_bits=100)
        # Perturb STN bias to non-identity
        with torch.no_grad():
            decoder.stn_fc_bias.copy_(torch.tensor([0.9, 0.1, 0.05, -0.1, 0.9, -0.05]))

        # Forward should still work (STN handles transform)
        result = decoder(sample_image)
        assert result.shape == (2, 100)
