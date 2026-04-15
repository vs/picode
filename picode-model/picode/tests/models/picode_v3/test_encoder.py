"""Tests for Picode v3 encoder."""

import pytest
import torch

from picode.models.picode_v3.encoder import Encoder


class TestEncoder:
    """Tests for Picode v3 Encoder."""

    @pytest.fixture
    def encoder(self) -> Encoder:
        """Create encoder instance with default 127 bits (BCH codeword)."""
        return Encoder(num_bits=127)

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Sample input image."""
        return torch.rand(2, 3, 400, 400)

    @pytest.fixture
    def sample_message(self) -> torch.Tensor:
        """Sample message bits (127-bit BCH codeword)."""
        return torch.randint(0, 2, (2, 127)).float()

    def test_output_shape(
        self, encoder: Encoder, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Output shape matches input image shape."""
        encoded = encoder(sample_image, sample_message)
        assert encoded.shape == sample_image.shape

    def test_output_range(
        self, encoder: Encoder, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Output is clamped to [0, 1]."""
        encoded = encoder(sample_image, sample_message)
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0

    def test_residual_not_zero(
        self, encoder: Encoder, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Encoder produces non-zero residual."""
        encoded = encoder(sample_image, sample_message)
        residual = encoded - sample_image
        assert residual.abs().mean() > 0.01

    def test_gradient_flow(
        self, encoder: Encoder, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """Gradients flow through encoder."""
        sample_image.requires_grad_(True)
        encoded = encoder(sample_image, sample_message)
        encoded.sum().backward()

        assert sample_image.grad is not None
        assert sample_image.grad.abs().mean() > 0

    def test_message_affects_output(self, encoder: Encoder, sample_image: torch.Tensor) -> None:
        """Different messages produce different outputs."""
        msg1 = torch.zeros(2, 127)
        msg2 = torch.ones(2, 127)

        enc1 = encoder(sample_image, msg1)
        enc2 = encoder(sample_image, msg2)

        # Outputs should differ
        assert not torch.allclose(enc1, enc2, atol=1e-3)

    def test_default_num_bits_is_127(self) -> None:
        """Default num_bits is 127 (BCH codeword length)."""
        encoder = Encoder()
        assert encoder.num_bits == 127

    def test_different_num_bits(self) -> None:
        """Works with different message lengths."""
        for num_bits in [50, 100, 127, 200]:
            encoder = Encoder(num_bits=num_bits)
            x = torch.rand(1, 3, 400, 400)
            msg = torch.randint(0, 2, (1, num_bits)).float()
            encoded = encoder(x, msg)
            assert encoded.shape == x.shape
