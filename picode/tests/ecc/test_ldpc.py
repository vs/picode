"""Tests for LDPC error correction."""

import pytest
import torch

from picode.ecc.ldpc import LDPC


class TestLDPCConstruction:
    """Tests for LDPC construction and properties."""

    def test_constructor_default_params(self) -> None:
        """LDPC can be constructed with just codeword length."""
        ldpc = LDPC(n=198)

        assert ldpc.codeword_length == 198
        assert ldpc.message_length > 0
        assert ldpc.message_length < 198
        assert 0 < ldpc.rate < 1

    def test_constructor_custom_params(self) -> None:
        """LDPC can be constructed with custom d_v and d_c."""
        ldpc = LDPC(n=198, d_v=3, d_c=6)

        assert ldpc.codeword_length == 198
        assert ldpc.message_length > 0

    def test_rate_calculation(self) -> None:
        """Rate equals message_length / codeword_length."""
        ldpc = LDPC(n=198, d_v=3, d_c=6)

        expected_rate = ldpc.message_length / ldpc.codeword_length
        assert abs(ldpc.rate - expected_rate) < 1e-6

    def test_matrices_exposed(self) -> None:
        """H and G matrices are accessible."""
        ldpc = LDPC(n=198, d_v=3, d_c=6)

        assert ldpc.H is not None
        assert ldpc.G is not None
        assert ldpc.H.shape[1] == 198  # n columns
        assert ldpc.G.shape[0] == 198  # n rows

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical matrices."""
        ldpc1 = LDPC(n=198, d_v=3, d_c=6, seed=42)
        ldpc2 = LDPC(n=198, d_v=3, d_c=6, seed=42)

        assert (ldpc1.H == ldpc2.H).all()
        assert (ldpc1.G == ldpc2.G).all()
        assert ldpc1.message_length == ldpc2.message_length


class TestLDPCEncode:
    """Tests for LDPC encoding."""

    def test_encode_shape(self) -> None:
        """Encoded output has shape (B, n)."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        message = torch.randint(0, 2, (4, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)

        assert encoded.shape == (4, 198)

    def test_encode_binary_output(self) -> None:
        """Encoded output contains only 0s and 1s."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        message = torch.randint(0, 2, (4, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)

        assert torch.all((encoded == 0) | (encoded == 1))

    def test_encode_batch_size_one(self) -> None:
        """Encoding works with batch size 1."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        message = torch.randint(0, 2, (1, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)

        assert encoded.shape == (1, 198)

    def test_encode_deterministic(self) -> None:
        """Same message always produces same codeword."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        message = torch.randint(0, 2, (2, ldpc.message_length), dtype=torch.float32)

        encoded1 = ldpc.encode(message)
        encoded2 = ldpc.encode(message)

        assert torch.equal(encoded1, encoded2)
