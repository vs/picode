"""Integration tests for Picode v3 encoder-decoder pipeline."""

import pytest
import torch

from picode.ecc import BCH
from picode.models.picode_v3 import Decoder, Encoder


class TestEncoderDecoderIntegration:
    """Integration tests for Picode v3 encoder-decoder."""

    @pytest.fixture
    def encoder(self) -> Encoder:
        """Create encoder with 127 bits (BCH codeword)."""
        return Encoder(num_bits=127)

    @pytest.fixture
    def decoder(self) -> Decoder:
        """Create decoder with 127 bits (BCH codeword)."""
        return Decoder(num_bits=127)

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Sample input image."""
        return torch.rand(2, 3, 400, 400)

    @pytest.fixture
    def sample_message(self) -> torch.Tensor:
        """Sample 127-bit BCH codeword."""
        return torch.randint(0, 2, (2, 127)).float()

    def test_encode_decode_roundtrip(
        self,
        encoder: Encoder,
        decoder: Decoder,
        sample_image: torch.Tensor,
        sample_message: torch.Tensor,
    ) -> None:
        """Encoder and decoder work together for forward pass."""
        # Encode
        encoded = encoder(sample_image, sample_message)
        assert encoded.shape == sample_image.shape

        # Decode
        logits = decoder(encoded)
        assert logits.shape == (2, 127)

        # Get binary predictions
        predicted = decoder.decode(encoded)
        assert predicted.shape == sample_message.shape

    def test_gradient_flow_through_pipeline(
        self,
        encoder: Encoder,
        decoder: Decoder,
        sample_image: torch.Tensor,
        sample_message: torch.Tensor,
    ) -> None:
        """Gradients flow through entire pipeline."""
        # Forward pass
        encoded = encoder(sample_image, sample_message)
        logits = decoder(encoded)

        # Backward pass
        loss = logits.sum()
        loss.backward()

        # Check encoder gradients
        assert encoder.conv1.weight.grad is not None
        assert encoder.conv1.weight.grad.abs().mean() > 0

        # Check decoder gradients
        assert decoder.stn_fc_weight.grad is not None

    def test_message_recovery_clean(
        self,
        encoder: Encoder,
        decoder: Decoder,
        sample_image: torch.Tensor,
    ) -> None:
        """Test that different messages produce different decoded outputs."""
        msg1 = torch.zeros(2, 127)
        msg2 = torch.ones(2, 127)

        # Encode with different messages
        enc1 = encoder(sample_image, msg1)
        enc2 = encoder(sample_image, msg2)

        # Decode
        dec1 = decoder(enc1)
        dec2 = decoder(enc2)

        # Decoded outputs should differ
        assert not torch.allclose(dec1, dec2, atol=1e-2)


class TestBCHIntegration:
    """Test BCH error correction with Picode v3."""

    def test_bch_127_50_parameters(self) -> None:
        """BCH(127, 50) has correct parameters for Picode v3."""
        bch = BCH(127, 50)
        assert bch.codeword_length == 127
        assert bch.message_length == 50
        assert bch.t == 13  # Corrects up to 13 errors

    def test_bch_encode_decode_roundtrip(self) -> None:
        """BCH encoding/decoding works correctly."""
        bch = BCH(127, 50)

        # Create random 50-bit messages
        messages = torch.randint(0, 2, (4, 50)).float()

        # Encode to 127-bit codewords
        codewords = bch.encode(messages)
        assert codewords.shape == (4, 127)

        # Decode back to 50-bit messages
        decoded, success = bch.decode(codewords)
        assert decoded.shape == (4, 50)
        assert success.all()
        assert torch.equal(decoded, messages)

    def test_bch_error_correction(self) -> None:
        """BCH can correct up to 13 bit errors."""
        bch = BCH(127, 50)

        # Create a message and encode
        message = torch.randint(0, 2, (1, 50)).float()
        codeword = bch.encode(message)

        # Introduce 10 bit errors (within correction capability)
        corrupted = codeword.clone()
        error_positions = torch.randperm(127)[:10]
        for pos in error_positions:
            corrupted[0, pos] = 1.0 - corrupted[0, pos]

        # Decode should recover original
        decoded, success = bch.decode(corrupted)
        assert success.all()
        assert torch.equal(decoded, message)

    def test_bch_beyond_correction_capability(self) -> None:
        """BCH fails gracefully when too many errors."""
        bch = BCH(127, 50)

        # Create a message and encode
        message = torch.randint(0, 2, (1, 50)).float()
        codeword = bch.encode(message)

        # Introduce 20 bit errors (beyond correction capability of 13)
        corrupted = codeword.clone()
        error_positions = torch.randperm(127)[:20]
        for pos in error_positions:
            corrupted[0, pos] = 1.0 - corrupted[0, pos]

        # Decode should fail
        _, success = bch.decode(corrupted)
        assert not success.all()
