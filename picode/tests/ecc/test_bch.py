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
