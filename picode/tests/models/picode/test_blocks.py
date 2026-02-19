"""Tests for Picode building blocks."""

import torch

from picode.models.picode.blocks import ResBlock


class TestResBlock:
    """Tests for ResBlock."""

    def test_output_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        block = ResBlock(channels=64)
        x = torch.randn(2, 64, 32, 32)
        y = block(x)
        assert y.shape == x.shape

    def test_gradient_flows_through_skip(self) -> None:
        """Gradients flow through skip connection."""
        block = ResBlock(channels=32)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        y = block(x)
        y.sum().backward()

        # Input should receive gradients
        assert x.grad is not None
        assert x.grad.abs().mean() > 0

    def test_skip_connection_exists(self) -> None:
        """Skip connection allows identity-like behavior."""
        block = ResBlock(channels=32)

        # Zero out conv weights - output should still be non-zero due to skip
        with torch.no_grad():
            block.conv1.weight.zero_()
            assert block.conv1.bias is not None
            block.conv1.bias.zero_()
            block.conv2.weight.zero_()
            assert block.conv2.bias is not None
            block.conv2.bias.zero_()
            # Also zero norm parameters to ensure clean pass-through
            block.norm1.weight.fill_(1.0)
            block.norm1.bias.zero_()
            block.norm2.weight.fill_(1.0)
            block.norm2.bias.zero_()

        x = torch.randn(2, 32, 16, 16)
        y = block(x)

        # Output should be LeakyReLU(x) due to skip connection
        expected = torch.nn.functional.leaky_relu(x, 0.2)
        assert torch.allclose(y, expected, atol=1e-5)

    def test_handles_various_channel_counts(self) -> None:
        """Works with different channel counts."""
        for channels in [16, 32, 64, 128]:
            block = ResBlock(channels=channels)
            x = torch.randn(1, channels, 8, 8)
            y = block(x)
            assert y.shape == x.shape

    def test_groups_adjusted_for_small_channels(self) -> None:
        """GroupNorm groups adjusted when channels < default groups."""
        # 16 channels with default groups=8 should work
        block = ResBlock(channels=16, groups=8)
        x = torch.randn(1, 16, 8, 8)
        y = block(x)
        assert y.shape == x.shape
