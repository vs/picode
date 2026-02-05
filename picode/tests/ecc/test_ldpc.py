"""Tests for LDPC error correction."""

import pytest
import torch

from picode.ecc.ldpc import LDPC


class TestLDPCConstruction:
    """Tests for LDPC construction and properties."""

    def test_constructor_default_params(self) -> None:
        """LDPC can be constructed with just codeword length."""
        ldpc = LDPC(n=200)

        assert ldpc.codeword_length == 200
        assert ldpc.message_length > 0
        assert ldpc.message_length < 200
        assert 0 < ldpc.rate < 1

    def test_constructor_custom_params(self) -> None:
        """LDPC can be constructed with custom d_v and d_c."""
        ldpc = LDPC(n=200, d_v=3, d_c=6)

        assert ldpc.codeword_length == 200
        assert ldpc.message_length > 0

    def test_rate_calculation(self) -> None:
        """Rate equals message_length / codeword_length."""
        ldpc = LDPC(n=200, d_v=3, d_c=6)

        expected_rate = ldpc.message_length / ldpc.codeword_length
        assert abs(ldpc.rate - expected_rate) < 1e-6

    def test_matrices_exposed(self) -> None:
        """H and G matrices are accessible."""
        ldpc = LDPC(n=200, d_v=3, d_c=6)

        assert ldpc.H is not None
        assert ldpc.G is not None
        assert ldpc.H.shape[1] == 200  # n columns
        assert ldpc.G.shape[0] == 200  # n rows

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical matrices."""
        ldpc1 = LDPC(n=200, d_v=3, d_c=6, seed=42)
        ldpc2 = LDPC(n=200, d_v=3, d_c=6, seed=42)

        assert (ldpc1.H == ldpc2.H).all()
        assert (ldpc1.G == ldpc2.G).all()
        assert ldpc1.message_length == ldpc2.message_length
