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
