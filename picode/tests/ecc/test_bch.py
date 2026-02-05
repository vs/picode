"""BCH error correction tests."""

import pytest
import torch

from picode.ecc.bch import BCH


class TestBCHProperties:
    """Test BCH properties and initialization."""

    def test_default_parameters(self):
        """Default BCH(127, 64) for print-and-scan robustness."""
        bch = BCH()
        assert bch.codeword_length == 127
        assert bch.message_length == 64
        assert bch.t == 10
        assert bch.rate == 64 / 127

    def test_custom_parameters(self):
        """Custom BCH parameters."""
        bch = BCH(n=255, k=131)
        assert bch.codeword_length == 255
        assert bch.message_length == 131
        assert bch.t == 18


class TestBCHEncode:
    """Test BCH encoding."""

    def test_encode_output_shape(self):
        """Encode (B, k) -> (B, n)."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (4, 64)).float()
        encoded = bch.encode(msg)
        assert encoded.shape == (4, 127)

    def test_encode_output_binary(self):
        """Encoded output contains only 0s and 1s."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (2, 64)).float()
        encoded = bch.encode(msg)
        assert ((encoded == 0) | (encoded == 1)).all()

    def test_encode_preserves_device(self):
        """Output stays on same device as input."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (2, 64)).float()
        encoded = bch.encode(msg)
        assert encoded.device == msg.device

    def test_encode_deterministic(self):
        """Same input produces same output."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (2, 64)).float()
        encoded1 = bch.encode(msg)
        encoded2 = bch.encode(msg)
        assert (encoded1 == encoded2).all()


class TestBCHDecode:
    """Test BCH decoding."""

    def test_decode_no_errors(self):
        """Perfect codeword decodes correctly."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (4, 64)).float()
        encoded = bch.encode(msg)
        decoded, success = bch.decode(encoded)
        assert success.all()
        assert (decoded == msg).all()

    def test_decode_output_shape(self):
        """Decode (B, n) -> (B, k)."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (4, 64)).float()
        encoded = bch.encode(msg)
        decoded, success = bch.decode(encoded)
        assert decoded.shape == (4, 64)
        assert success.shape == (4,)

    def test_decode_preserves_device(self):
        """Output stays on same device as input."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (2, 64)).float()
        encoded = bch.encode(msg)
        decoded, success = bch.decode(encoded)
        assert decoded.device == msg.device
        assert success.device == msg.device

    def test_decode_corrects_errors_up_to_t(self):
        """Corrects up to t bit errors."""
        bch = BCH(127, 64)  # t=10
        msg = torch.randint(0, 2, (1, 64)).float()
        encoded = bch.encode(msg)

        # Flip exactly t bits (should correct)
        corrupted = encoded.clone()
        corrupted[0, :10] = 1 - corrupted[0, :10]

        decoded, success = bch.decode(corrupted)
        assert success.all()
        assert (decoded == msg).all()

    def test_decode_fails_beyond_t(self):
        """Too many errors returns success=False."""
        bch = BCH(127, 64)  # t=10
        msg = torch.randint(0, 2, (1, 64)).float()
        encoded = bch.encode(msg)

        # Flip more than t bits at random positions (uncorrectable)
        corrupted = encoded.clone()
        error_positions = torch.randperm(127)[:15]
        corrupted[0, error_positions] = 1 - corrupted[0, error_positions]

        decoded, success = bch.decode(corrupted)
        assert not success.all()

    def test_decode_soft_values_thresholded(self):
        """Soft decoder outputs are thresholded at 0.5."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (1, 64)).float()
        encoded = bch.encode(msg)

        # Add noise but keep values interpretable
        noisy = encoded + torch.randn_like(encoded) * 0.1
        noisy = noisy.clamp(0, 1)

        decoded, success = bch.decode(noisy)
        assert success.all()
        assert (decoded == msg).all()
