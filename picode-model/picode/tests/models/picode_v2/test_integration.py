"""Integration tests for picode_v2 encoder/decoder."""

import torch

from picode.models.picode_v2 import Decoder, Encoder


class TestEncoderDecoderIntegration:
    """Test encoder and decoder work together."""

    def test_encode_decode_cycle(self) -> None:
        """Full encode-decode cycle produces reasonable output."""
        encoder = Encoder(num_bits=100)
        decoder = Decoder(num_bits=100)

        image = torch.rand(2, 3, 400, 400)
        message = torch.randint(0, 2, (2, 100)).float()

        # Encode
        encoded = encoder(image, message)
        assert encoded.shape == image.shape
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0

        # Decode
        logits = decoder(encoded)
        assert logits.shape == (2, 100)

        # Decode to bits
        bits = decoder.decode(encoded)
        assert bits.shape == (2, 100)
        assert ((bits == 0) | (bits == 1)).all()

    def test_gradient_flows_encoder_to_decoder(self) -> None:
        """Gradients flow from decoder loss back to encoder."""
        encoder = Encoder(num_bits=100)
        decoder = Decoder(num_bits=100)

        image = torch.rand(2, 3, 400, 400)
        message = torch.randint(0, 2, (2, 100)).float()

        # Forward pass
        encoded = encoder(image, message)
        logits = decoder(encoded)

        # Backward pass
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, message)
        loss.backward()

        # Check encoder received gradients
        encoder_conv = encoder.conv1
        assert encoder_conv.weight.grad is not None
        assert encoder_conv.weight.grad.abs().mean() > 0

    def test_different_num_bits_consistency(self) -> None:
        """Encoder and decoder work together with various bit lengths."""
        for num_bits in [50, 100, 200]:
            encoder = Encoder(num_bits=num_bits)
            decoder = Decoder(num_bits=num_bits)

            image = torch.rand(1, 3, 400, 400)
            message = torch.randint(0, 2, (1, num_bits)).float()

            encoded = encoder(image, message)
            bits = decoder.decode(encoded)

            assert bits.shape == (1, num_bits)

    def test_residual_is_content_adaptive(self) -> None:
        """Verify residual magnitude varies with image content when enabled."""
        # Content-adaptive scaling is optional; test with it enabled
        encoder = Encoder(num_bits=100, use_content_adaptive=True)
        message = torch.randint(0, 2, (1, 100)).float()

        # Smooth image
        smooth = torch.ones(1, 3, 400, 400) * 0.5

        # Textured image (random noise)
        textured = torch.rand(1, 3, 400, 400)

        enc_smooth = encoder(smooth, message)
        enc_textured = encoder(textured, message)

        residual_smooth = (enc_smooth - smooth).abs()
        residual_textured = (enc_textured - textured).abs()

        # Mean residual should be smaller for smooth image
        assert residual_smooth.mean() < residual_textured.mean()

    def test_decoder_parameter_count(self) -> None:
        """Decoder has ~11.5M parameters (StegaStamp-style with flatten+FC)."""
        decoder = Decoder(num_bits=100)
        param_count = sum(p.numel() for p in decoder.parameters())
        # Expect ~11.5M params (mostly from FC1: 21632 * 512 = 11M)
        assert 10_000_000 < param_count < 15_000_000, f"Decoder has {param_count} params"

    def test_imports_from_package(self) -> None:
        """Can import all components from package."""
        from picode.models.picode_v2 import (  # noqa: F401
            Decoder,
            Encoder,
            FocalFrequencyLoss,
            InvertedResidual,
            MessageExpander,
            MobileDecoder,
            PatchDiscriminator,
            discriminator_loss,
            generator_loss,
        )

        # Verify classes are what we expect
        assert Decoder is MobileDecoder
        assert callable(discriminator_loss)
        assert callable(generator_loss)
