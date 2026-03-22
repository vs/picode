"""Tests for picode_v2 MobileDecoder."""

import torch
import torch.nn as nn

from picode.models.picode_v2.decoder import MobileDecoder


class TestMobileDecoder:
    """Tests for MobileDecoder."""

    def test_output_shape(self) -> None:
        """Output shape is (batch, num_bits)."""
        decoder = MobileDecoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400)
        logits = decoder(x)
        assert logits.shape == (2, 100)

    def test_output_is_logits(self) -> None:
        """Output is unbounded logits, not probabilities."""
        decoder = MobileDecoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400)
        logits = decoder(x)
        # Logits can be any value
        assert logits.abs().mean() > 0

    def test_decode_returns_binary(self) -> None:
        """decode() returns binary predictions."""
        decoder = MobileDecoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400)
        bits = decoder.decode(x)
        assert bits.shape == (2, 100)
        assert ((bits == 0) | (bits == 1)).all()

    def test_gradient_flow(self) -> None:
        """Gradients flow to input."""
        decoder = MobileDecoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400, requires_grad=True)
        logits = decoder(x)
        logits.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().mean() > 0

    def test_no_stn(self) -> None:
        """No Spatial Transformer Network (mobile incompatible)."""
        decoder = MobileDecoder(num_bits=100)
        # Check no affine_grid or grid_sample in forward
        # These would appear as F.affine_grid calls
        assert not hasattr(decoder, 'stn_params')
        assert not hasattr(decoder, 'stn_fc')

    def test_uses_batchnorm_only(self) -> None:
        """Uses BatchNorm, not GroupNorm (mobile requirement)."""
        decoder = MobileDecoder(num_bits=100)
        has_batchnorm = any(
            isinstance(m, nn.BatchNorm2d) for m in decoder.modules()
        )
        has_groupnorm = any(
            isinstance(m, nn.GroupNorm) for m in decoder.modules()
        )
        assert has_batchnorm is True
        assert has_groupnorm is False

    def test_uses_relu6_only(self) -> None:
        """Uses ReLU6, not LeakyReLU (mobile requirement)."""
        decoder = MobileDecoder(num_bits=100)
        has_relu6 = any(
            isinstance(m, nn.ReLU6) for m in decoder.modules()
        )
        has_leaky = any(
            isinstance(m, nn.LeakyReLU) for m in decoder.modules()
        )
        assert has_relu6 is True
        assert has_leaky is False

    def test_different_num_bits(self) -> None:
        """Works with different message lengths."""
        for num_bits in [50, 100, 200]:
            decoder = MobileDecoder(num_bits=num_bits)
            x = torch.rand(1, 3, 400, 400)
            logits = decoder(x)
            assert logits.shape == (1, num_bits)

    def test_parameter_count_under_500k(self) -> None:
        """Total parameters under 500K (mobile requirement)."""
        decoder = MobileDecoder(num_bits=100)
        total_params = sum(p.numel() for p in decoder.parameters())
        assert total_params < 500_000, f"Got {total_params} params, expected < 500K"
