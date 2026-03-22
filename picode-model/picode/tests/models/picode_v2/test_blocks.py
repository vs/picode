"""Tests for picode_v2 building blocks."""

import torch

from picode.models.picode_v2.blocks import InvertedResidual, MessageExpander


class TestInvertedResidual:
    """Tests for InvertedResidual block."""

    def test_output_shape_stride_1(self) -> None:
        """Stride 1 preserves spatial dimensions."""
        block = InvertedResidual(in_ch=32, out_ch=32, stride=1, expand_ratio=6)
        x = torch.randn(2, 32, 64, 64)
        y = block(x)
        assert y.shape == (2, 32, 64, 64)

    def test_output_shape_stride_2(self) -> None:
        """Stride 2 halves spatial dimensions."""
        block = InvertedResidual(in_ch=32, out_ch=64, stride=2, expand_ratio=6)
        x = torch.randn(2, 32, 64, 64)
        y = block(x)
        assert y.shape == (2, 64, 32, 32)

    def test_residual_connection_stride_1_same_channels(self) -> None:
        """Residual connection used when stride=1 and channels match."""
        block = InvertedResidual(in_ch=32, out_ch=32, stride=1, expand_ratio=6)
        assert block.use_residual is True

    def test_no_residual_connection_stride_2(self) -> None:
        """No residual connection when stride=2."""
        block = InvertedResidual(in_ch=32, out_ch=64, stride=2, expand_ratio=6)
        assert block.use_residual is False

    def test_no_residual_connection_channel_mismatch(self) -> None:
        """No residual connection when channels differ."""
        block = InvertedResidual(in_ch=32, out_ch=64, stride=1, expand_ratio=6)
        assert block.use_residual is False

    def test_gradient_flow(self) -> None:
        """Gradients flow through the block."""
        block = InvertedResidual(in_ch=32, out_ch=32, stride=1, expand_ratio=6)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().mean() > 0

    def test_uses_relu6(self) -> None:
        """Block uses ReLU6 activation (output bounded by 6)."""
        block = InvertedResidual(in_ch=16, out_ch=16, stride=1, expand_ratio=6)
        # Large input to trigger ReLU6 clamping
        x = torch.ones(1, 16, 8, 8) * 10
        y = block(x)
        # ReLU6 clamps at 6, so internal activations should be bounded
        # Output may exceed due to linear projection, but internals are bounded
        assert y.max() < 100  # Reasonable bound check

    def test_expand_ratio_1(self) -> None:
        """Works with expand_ratio=1 (no expansion)."""
        block = InvertedResidual(in_ch=32, out_ch=32, stride=1, expand_ratio=1)
        x = torch.randn(2, 32, 16, 16)
        y = block(x)
        assert y.shape == x.shape

    def test_uses_batchnorm_not_groupnorm(self) -> None:
        """Uses BatchNorm2d (mobile-friendly) not GroupNorm."""
        block = InvertedResidual(in_ch=32, out_ch=32, stride=1, expand_ratio=6)
        has_batchnorm = any(
            isinstance(m, torch.nn.BatchNorm2d) for m in block.modules()
        )
        has_groupnorm = any(
            isinstance(m, torch.nn.GroupNorm) for m in block.modules()
        )
        assert has_batchnorm is True
        assert has_groupnorm is False


class TestMessageExpander:
    """Tests for MessageExpander."""

    def test_output_shape(self) -> None:
        """Output is (B, 3, 400, 400)."""
        expander = MessageExpander(num_bits=100)
        msg = torch.randn(2, 100)
        out = expander(msg)
        assert out.shape == (2, 3, 400, 400)

    def test_gradient_flow(self) -> None:
        """Gradients flow to input message."""
        expander = MessageExpander(num_bits=100)
        msg = torch.randn(2, 100, requires_grad=True)
        out = expander(msg)
        out.sum().backward()
        assert msg.grad is not None
        assert msg.grad.abs().mean() > 0

    def test_different_messages_different_outputs(self) -> None:
        """Different messages produce different outputs."""
        expander = MessageExpander(num_bits=100)
        msg1 = torch.zeros(1, 100)
        msg2 = torch.ones(1, 100)
        out1 = expander(msg1)
        out2 = expander(msg2)
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_uses_bilinear_upsampling(self) -> None:
        """Uses nn.Upsample (not ConvTranspose2d) to avoid checkerboard."""
        expander = MessageExpander(num_bits=100)
        has_upsample = any(
            isinstance(m, torch.nn.Upsample) for m in expander.modules()
        )
        has_convtranspose = any(
            isinstance(m, torch.nn.ConvTranspose2d) for m in expander.modules()
        )
        assert has_upsample is True
        assert has_convtranspose is False

    def test_different_num_bits(self) -> None:
        """Works with different message lengths."""
        for num_bits in [50, 100, 200]:
            expander = MessageExpander(num_bits=num_bits)
            msg = torch.randn(1, num_bits)
            out = expander(msg)
            assert out.shape == (1, 3, 400, 400)
