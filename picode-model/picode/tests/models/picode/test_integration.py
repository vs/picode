# picode/tests/models/picode/test_integration.py
"""Integration tests for Picode encoder-decoder."""

import pytest
import torch
import torch.nn.functional as F

from picode.models.picode import Decoder, Encoder


class TestEncoderDecoderIntegration:
    """Test encoder and decoder work together."""

    @pytest.fixture
    def models(self) -> tuple[Encoder, Decoder]:
        """Create encoder and decoder."""
        encoder = Encoder(num_bits=100)
        decoder = Decoder(num_bits=100)
        return encoder, decoder

    @pytest.fixture
    def sample_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample data."""
        torch.manual_seed(42)
        images = torch.rand(2, 3, 400, 400)
        messages = torch.randint(0, 2, (2, 100)).float()
        return images, messages

    def test_forward_pass_shapes(
        self, models: tuple[Encoder, Decoder], sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Forward pass produces correct shapes."""
        encoder, decoder = models
        images, messages = sample_data

        encoded = encoder(images, messages)
        logits = decoder(encoded)

        assert encoded.shape == images.shape
        assert logits.shape == messages.shape

    def test_backward_pass_no_errors(
        self, models: tuple[Encoder, Decoder], sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Backward pass completes without errors."""
        encoder, decoder = models
        images, messages = sample_data

        encoded = encoder(images, messages)
        logits = decoder(encoded)
        loss = F.binary_cross_entropy_with_logits(logits, messages)

        # Should not raise
        loss.backward()

        # Both models should have gradients
        assert any(p.grad is not None for p in encoder.parameters())
        assert any(p.grad is not None for p in decoder.parameters())

    def test_same_api_as_stegastamp(self) -> None:
        """Picode has same API as StegaStamp."""
        from picode.models.stegastamp import Decoder as StegaDecoder
        from picode.models.stegastamp import Encoder as StegaEncoder

        # Same constructor signature
        picode_enc = Encoder(num_bits=50)
        stega_enc = StegaEncoder(num_bits=50)

        picode_dec = Decoder(num_bits=50)
        stega_dec = StegaDecoder(num_bits=50)

        # Same forward signature
        x = torch.rand(1, 3, 400, 400)
        msg = torch.randint(0, 2, (1, 50)).float()

        # Encoder: (image, message) -> encoded_image
        enc1 = picode_enc(x, msg)
        enc2 = stega_enc(x, msg)
        assert enc1.shape == enc2.shape

        # Decoder: (image) -> logits
        dec1 = picode_dec(x)
        dec2 = stega_dec(x)
        assert dec1.shape == dec2.shape

        # decode method
        bits1 = picode_dec.decode(x)
        bits2 = stega_dec.decode(x)
        assert bits1.shape == bits2.shape
