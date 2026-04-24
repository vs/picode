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


class TestBCHParameterVariations:
    """Test BCH with various parameter configurations."""

    @pytest.mark.parametrize(
        "n,k,expected_t",
        [
            (127, 64, 10),  # Default: high error correction
            (127, 106, 3),  # Near 100 bits, low correction
            (127, 113, 2),  # Max capacity for n=127
            (255, 131, 18),  # Larger codeword, high correction
            (63, 36, 5),  # Smaller codeword
            (31, 16, 3),  # Minimal codeword
        ],
    )
    def test_various_parameters(self, n: int, k: int, expected_t: int):
        """Verify BCH works with different (n, k) configurations."""
        bch = BCH(n, k)
        assert bch.codeword_length == n
        assert bch.message_length == k
        assert bch.t == expected_t

        # Round-trip test
        msg = torch.randint(0, 2, (2, k)).float()
        encoded = bch.encode(msg)
        assert encoded.shape == (2, n)

        decoded, success = bch.decode(encoded)
        assert success.all()
        assert (decoded == msg).all()

    @pytest.mark.parametrize(
        "n,k",
        [
            (127, 106),  # 106 message bits, embeds 127 bits
            (255, 131),  # 131 message bits, embeds 255 bits
        ],
    )
    def test_stegastamp_compatible_configs(self, n: int, k: int):
        """Configurations suitable for ~100-bit messages."""
        bch = BCH(n, k)

        msg = torch.randint(0, 2, (4, k)).float()
        encoded = bch.encode(msg)

        # Simulate some bit errors from distortions
        corrupted = encoded.clone()
        num_errors = min(bch.t, 3)  # Inject a few errors
        for i in range(corrupted.shape[0]):
            flip_idx = torch.randperm(n)[:num_errors]
            corrupted[i, flip_idx] = 1 - corrupted[i, flip_idx]

        decoded, success = bch.decode(corrupted)
        assert success.all()
        assert (decoded == msg).all()


class TestBCHBatchBehavior:
    """Test BCH batch processing behavior."""

    def test_partial_batch_failure(self):
        """Some codewords fail, others succeed."""
        bch = BCH(127, 64)  # t=10
        msg = torch.randint(0, 2, (3, 64)).float()
        encoded = bch.encode(msg)

        corrupted = encoded.clone()
        # First: 5 errors (correctable)
        corrupted[0, :5] = 1 - corrupted[0, :5]
        # Second: 15 errors (uncorrectable)
        corrupted[1, :15] = 1 - corrupted[1, :15]
        # Third: 0 errors (perfect)

        decoded, success = bch.decode(corrupted)

        assert success[0].item() is True
        assert success[1].item() is False
        assert success[2].item() is True
        assert (decoded[0] == msg[0]).all()
        assert (decoded[2] == msg[2]).all()

    def test_single_batch(self):
        """Single item batch works correctly."""
        bch = BCH(127, 64)
        msg = torch.randint(0, 2, (1, 64)).float()
        encoded = bch.encode(msg)
        decoded, success = bch.decode(encoded)
        assert decoded.shape == (1, 64)
        assert success.shape == (1,)
        assert success.all()
        assert (decoded == msg).all()


def test_bch_63_36() -> None:
    """BCH(63, 36) works for PicodeLite."""
    bch = BCH(n=63, k=36)

    # Check properties
    assert bch.codeword_length == 63
    assert bch.message_length == 36
    assert bch.t >= 5  # Should correct at least 5 errors

    # Test encode/decode
    message = torch.randint(0, 2, (4, 36)).float()
    codeword = bch.encode(message)
    assert codeword.shape == (4, 63)

    # Decode without errors
    decoded, success = bch.decode(codeword)
    assert decoded.shape == (4, 36)
    assert success.all()
    assert torch.equal(decoded, message)

    # Decode with 5 errors (should correct)
    corrupted = codeword.clone()
    corrupted[0, :5] = 1 - corrupted[0, :5]  # Flip 5 bits
    decoded, success = bch.decode(corrupted)
    assert success[0]
    assert torch.equal(decoded[0], message[0])
