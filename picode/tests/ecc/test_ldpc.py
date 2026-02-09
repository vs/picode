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


class TestLDPCDecode:
    """Tests for LDPC decoding."""

    def test_decode_shape(self) -> None:
        """Decoded output has shape (B, k)."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        # Soft values in [0, 1]
        received = torch.rand(4, 198)

        decoded = ldpc.decode(received)

        assert decoded.shape == (4, ldpc.message_length)

    def test_decode_binary_output(self) -> None:
        """Decoded output contains only 0s and 1s."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        received = torch.rand(4, 198)

        decoded = ldpc.decode(received)

        assert torch.all((decoded == 0) | (decoded == 1))

    def test_decode_batch_size_one(self) -> None:
        """Decoding works with batch size 1."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        received = torch.rand(1, 198)

        decoded = ldpc.decode(received)

        assert decoded.shape == (1, ldpc.message_length)


class TestLDPCRoundtrip:
    """Tests for encode -> decode roundtrip."""

    def test_roundtrip_no_noise(self) -> None:
        """Perfect recovery when no noise is added."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        torch.manual_seed(123)
        message = torch.randint(0, 2, (4, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)
        # Perfect channel: codeword as soft values
        decoded = ldpc.decode(encoded)

        assert torch.equal(decoded, message)

    def test_roundtrip_low_noise(self) -> None:
        """Recovery with ~5% bit flip noise."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42, snr=8.0)
        torch.manual_seed(123)
        message = torch.randint(0, 2, (8, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)

        # Add noise: flip ~5% of bits
        noise_mask = torch.rand_like(encoded) < 0.05
        noisy = torch.where(noise_mask, 1 - encoded, encoded)

        decoded = ldpc.decode(noisy)

        # Should recover at least half of messages perfectly
        correct = (decoded == message).all(dim=1).sum()
        assert correct >= 4, f"Only {correct}/8 messages recovered"

    def test_roundtrip_moderate_noise(self) -> None:
        """Recovery with ~7% bit flip noise - verify some messages decode perfectly."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42, snr=7.0)
        torch.manual_seed(456)
        message = torch.randint(0, 2, (8, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)

        # Add noise: flip ~7% of bits (within LDPC correction capability)
        noise_mask = torch.rand_like(encoded) < 0.07
        noisy = torch.where(noise_mask, 1 - encoded, encoded)

        decoded = ldpc.decode(noisy)

        # Should recover at least some messages perfectly
        correct = (decoded == message).all(dim=1).sum()
        assert correct >= 2, f"Only {correct}/8 messages recovered perfectly"

    def test_soft_input_improves_recovery(self) -> None:
        """Soft inputs (confidence values) improve recovery vs hard decisions."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)
        torch.manual_seed(789)
        message = torch.randint(0, 2, (4, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)

        # Add moderate noise
        noise_mask = torch.rand_like(encoded) < 0.08
        noisy_hard = torch.where(noise_mask, 1 - encoded, encoded)

        # Soft version: uncertain where noise was added
        noisy_soft = noisy_hard.clone()
        noisy_soft[noise_mask] = 0.5  # Mark flipped bits as uncertain

        decoded_hard = ldpc.decode(noisy_hard)
        decoded_soft = ldpc.decode(noisy_soft)

        # Soft should be at least as good as hard
        errors_hard = (decoded_hard != message).sum()
        errors_soft = (decoded_soft != message).sum()
        assert errors_soft <= errors_hard


class TestLDPCParameters:
    """Tests for various LDPC parameter configurations."""

    @pytest.mark.parametrize("n,d_v,d_c", [
        (198, 3, 6),   # Standard rate ~0.5
        (150, 3, 6),   # Smaller codeword
        (300, 2, 4),   # Different degrees
        (200, 4, 8),   # Higher degrees (200 is divisible by 8)
    ])
    def test_various_parameters(self, n: int, d_v: int, d_c: int) -> None:
        """LDPC works across different code configurations."""
        ldpc = LDPC(n=n, d_v=d_v, d_c=d_c, seed=42)

        torch.manual_seed(42)
        message = torch.randint(0, 2, (2, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)
        decoded = ldpc.decode(encoded)

        assert torch.equal(decoded, message)
        assert encoded.shape[1] == n
        assert decoded.shape[1] == ldpc.message_length

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        """Works with various batch sizes."""
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)

        torch.manual_seed(42)
        message = torch.randint(0, 2, (batch_size, ldpc.message_length), dtype=torch.float32)

        encoded = ldpc.encode(message)
        decoded = ldpc.decode(encoded)

        assert encoded.shape == (batch_size, 198)
        assert decoded.shape == (batch_size, ldpc.message_length)
        assert torch.equal(decoded, message)

    def test_stegastamp_compatible(self) -> None:
        """Configuration that gives k close to 100 for StegaStamp."""
        # With d_v=3, d_c=6, rate ~0.5, so n=198 gives k~99
        ldpc = LDPC(n=198, d_v=3, d_c=6, seed=42)

        # k should be close to 100 (exact value depends on matrix construction)
        assert 90 <= ldpc.message_length <= 110, (
            f"Expected k~100, got {ldpc.message_length}"
        )

        # Verify roundtrip works
        torch.manual_seed(42)
        message = torch.randint(0, 2, (2, ldpc.message_length), dtype=torch.float32)
        encoded = ldpc.encode(message)
        decoded = ldpc.decode(encoded)

        assert torch.equal(decoded, message)
