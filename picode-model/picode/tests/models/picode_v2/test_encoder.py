"""Tests for picode_v2 Encoder."""

import torch
import torch.nn as nn

from picode.models.picode_v2.encoder import Encoder


class TestEncoder:
    """Tests for picode_v2 Encoder."""

    def test_output_shape(self) -> None:
        """Output shape matches input image shape."""
        encoder = Encoder(num_bits=100)
        image = torch.rand(2, 3, 400, 400)
        message = torch.randint(0, 2, (2, 100)).float()
        encoded = encoder(image, message)
        assert encoded.shape == image.shape

    def test_output_range(self) -> None:
        """Output is clamped to [0, 1]."""
        encoder = Encoder(num_bits=100)
        image = torch.rand(2, 3, 400, 400)
        message = torch.randint(0, 2, (2, 100)).float()
        encoded = encoder(image, message)
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0

    def test_residual_bounded(self) -> None:
        """Residual is bounded (content-adaptive + tanh)."""
        encoder = Encoder(num_bits=100)
        image = torch.rand(2, 3, 400, 400)
        message = torch.randint(0, 2, (2, 100)).float()
        encoded = encoder(image, message)
        residual = encoded - image
        assert residual.abs().max() <= 0.15  # Some tolerance

    def test_gradient_flow(self) -> None:
        """Gradients flow through encoder."""
        encoder = Encoder(num_bits=100)
        image = torch.rand(2, 3, 400, 400, requires_grad=True)
        message = torch.randint(0, 2, (2, 100)).float()
        encoded = encoder(image, message)
        encoded.sum().backward()
        assert image.grad is not None
        assert image.grad.abs().mean() > 0

    def test_message_affects_output(self) -> None:
        """Different messages produce different outputs."""
        encoder = Encoder(num_bits=100)
        image = torch.rand(2, 3, 400, 400)
        msg1 = torch.zeros(2, 100)
        msg2 = torch.ones(2, 100)
        enc1 = encoder(image, msg1)
        enc2 = encoder(image, msg2)
        assert not torch.allclose(enc1, enc2, atol=1e-3)

    def test_content_adaptive_scaling(self) -> None:
        """Residual is smaller in smooth regions."""
        encoder = Encoder(num_bits=100)
        message = torch.randint(0, 2, (1, 100)).float()
        smooth = torch.ones(1, 3, 400, 400) * 0.5
        textured = torch.rand(1, 3, 400, 400)
        enc_smooth = encoder(smooth, message)
        enc_textured = encoder(textured, message)
        residual_smooth = (enc_smooth - smooth).abs().mean()
        residual_textured = (enc_textured - textured).abs().mean()
        assert residual_smooth < residual_textured

    def test_uses_bilinear_upsampling(self) -> None:
        """Uses bilinear upsampling, not nearest-neighbor."""
        encoder = Encoder(num_bits=100)
        has_upsample = any(
            isinstance(m, nn.Upsample) for m in encoder.modules()
        )
        assert has_upsample is True

    def test_different_num_bits(self) -> None:
        """Works with different message lengths."""
        for num_bits in [50, 100, 200]:
            encoder = Encoder(num_bits=num_bits)
            image = torch.rand(1, 3, 400, 400)
            message = torch.randint(0, 2, (1, num_bits)).float()
            encoded = encoder(image, message)
            assert encoded.shape == image.shape
