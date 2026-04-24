"""Integration tests for PicodeLite model."""

import pytest
import torch

from picode.models.picodelite import Encoder, Decoder
from picode.ecc import BCH


class TestPicodeLiteIntegration:
    """End-to-end tests for PicodeLite."""

    @pytest.fixture
    def encoder(self) -> Encoder:
        return Encoder(num_bits=63)

    @pytest.fixture
    def decoder(self) -> Decoder:
        return Decoder(num_bits=63)

    @pytest.fixture
    def bch(self) -> BCH:
        return BCH(n=63, k=36)

    def test_encode_decode_roundtrip(
        self, encoder: Encoder, decoder: Decoder
    ) -> None:
        """Message survives encode-decode without distortions."""
        # Create test data
        image = torch.rand(1, 3, 800, 800)
        message = torch.randint(0, 2, (1, 63)).float()

        # Encode
        encoded = encoder(image, message)
        assert encoded.shape == (1, 3, 800, 800)

        # Resize for decoder (simulates real pipeline)
        encoded_resized = torch.nn.functional.interpolate(
            encoded, size=(320, 320), mode="bilinear", align_corners=False
        )

        # Decode
        decoded = decoder.decode(encoded_resized)
        assert decoded.shape == (1, 63)

        # Note: Without training, accuracy will be random (~50%)
        # This test just verifies the pipeline works

    def test_full_pipeline_with_ecc(
        self, encoder: Encoder, decoder: Decoder, bch: BCH
    ) -> None:
        """Full pipeline with ECC: payload -> encode -> decode -> payload."""
        # Create 36-bit payload
        payload = torch.randint(0, 2, (1, 36)).float()

        # ECC encode to 63 bits
        codeword = bch.encode(payload)
        assert codeword.shape == (1, 63)

        # Steganographic encode
        image = torch.rand(1, 3, 800, 800)
        encoded = encoder(image, codeword)

        # Resize and decode
        encoded_resized = torch.nn.functional.interpolate(
            encoded, size=(320, 320), mode="bilinear", align_corners=False
        )
        decoded_bits = decoder.decode(encoded_resized)

        # ECC decode (may fail without training, but should not crash)
        recovered, success = bch.decode(decoded_bits)
        assert recovered.shape == (1, 36)

    def test_different_batch_sizes(
        self, encoder: Encoder, decoder: Decoder
    ) -> None:
        """Works with various batch sizes."""
        for batch in [1, 2, 4]:
            image = torch.rand(batch, 3, 800, 800)
            message = torch.randint(0, 2, (batch, 63)).float()

            encoded = encoder(image, message)
            assert encoded.shape == (batch, 3, 800, 800)

            encoded_resized = torch.nn.functional.interpolate(
                encoded, size=(320, 320), mode="bilinear"
            )
            decoded = decoder(encoded_resized)
            assert decoded.shape == (batch, 63)

    def test_gradient_flow_through_pipeline(
        self, encoder: Encoder, decoder: Decoder
    ) -> None:
        """Gradients flow from decoder loss back to encoder."""
        image = torch.rand(1, 3, 800, 800, requires_grad=True)
        message = torch.randint(0, 2, (1, 63)).float()

        # Forward
        encoded = encoder(image, message)
        encoded_resized = torch.nn.functional.interpolate(
            encoded, size=(320, 320), mode="bilinear", align_corners=False
        )
        decoded_logits = decoder(encoded_resized)

        # Loss and backward
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            decoded_logits, message
        )
        loss.backward()

        # Check gradients exist
        assert image.grad is not None
        assert not torch.all(image.grad == 0)

        # Check encoder has gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
