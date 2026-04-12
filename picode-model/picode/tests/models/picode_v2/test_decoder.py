"""Tests for picode_v2 StegaStamp-style Decoder."""

import torch
import torch.nn as nn

from picode.models.picode_v2.decoder import Decoder, MobileDecoder


class TestDecoder:
    """Tests for StegaStamp-style Decoder."""

    def test_output_shape(self) -> None:
        """Output shape is (batch, num_bits)."""
        decoder = Decoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400)
        logits = decoder(x)
        assert logits.shape == (2, 100)

    def test_output_is_logits(self) -> None:
        """Output is unbounded logits, not probabilities."""
        decoder = Decoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400)
        logits = decoder(x)
        # Logits can be any value
        assert logits.abs().mean() > 0

    def test_decode_returns_binary(self) -> None:
        """decode() returns binary predictions."""
        decoder = Decoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400)
        bits = decoder.decode(x)
        assert bits.shape == (2, 100)
        assert ((bits == 0) | (bits == 1)).all()

    def test_gradient_flow(self) -> None:
        """Gradients flow to input."""
        decoder = Decoder(num_bits=100)
        x = torch.rand(2, 3, 400, 400, requires_grad=True)
        logits = decoder(x)
        logits.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().mean() > 0

    def test_no_stn(self) -> None:
        """No Spatial Transformer Network (handled by detection pipeline)."""
        decoder = Decoder(num_bits=100)
        # Check no affine_grid or grid_sample in forward
        # These would appear as F.affine_grid calls
        assert not hasattr(decoder, 'stn_params')
        assert not hasattr(decoder, 'stn_fc')

    def test_no_batchnorm(self) -> None:
        """No BatchNorm (matches StegaStamp for training stability)."""
        decoder = Decoder(num_bits=100)
        has_batchnorm = any(
            isinstance(m, nn.BatchNorm2d) for m in decoder.modules()
        )
        assert has_batchnorm is False

    def test_uses_standard_convolutions(self) -> None:
        """Uses standard Conv2d (not depthwise separable)."""
        decoder = Decoder(num_bits=100)
        conv_layers = [m for m in decoder.modules() if isinstance(m, nn.Conv2d)]
        # All convolutions should have groups=1 (standard, not depthwise)
        for conv in conv_layers:
            assert conv.groups == 1, f"Found grouped conv with groups={conv.groups}"

    def test_different_num_bits(self) -> None:
        """Works with different message lengths."""
        for num_bits in [50, 100, 200]:
            decoder = Decoder(num_bits=num_bits)
            x = torch.rand(1, 3, 400, 400)
            logits = decoder(x)
            assert logits.shape == (1, num_bits)

    def test_parameter_count(self) -> None:
        """Total parameters ~11.5M (StegaStamp-style with flatten+FC)."""
        decoder = Decoder(num_bits=100)
        total_params = sum(p.numel() for p in decoder.parameters())
        # Expect ~11.5M params (mostly from FC1: 21632 * 512 = 11M)
        assert 10_000_000 < total_params < 15_000_000, f"Got {total_params} params"

    def test_mobile_decoder_alias(self) -> None:
        """MobileDecoder is alias for Decoder (backwards compatibility)."""
        assert MobileDecoder is Decoder


# Keep old test class name as alias for backwards compatibility
TestMobileDecoder = TestDecoder
