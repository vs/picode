"""Tests for the PicodeLite encoder module."""

import pytest
import torch
import torch.nn as nn

from picode.models.picodelite.encoder import Encoder


class TestEncoderArchitecture:
    """Test PicodeLite encoder architecture - learned upsampling, no BatchNorm."""

    def test_has_transposed_conv_layers(self) -> None:
        """Encoder uses ConvTranspose2d for learned upsampling."""
        encoder = Encoder(num_bits=63)
        transposed_convs = [m for m in encoder.modules() if isinstance(m, nn.ConvTranspose2d)]
        assert len(transposed_convs) > 0, "Should have ConvTranspose2d for learned upsampling"

    def test_no_batchnorm(self) -> None:
        """Encoder has no BatchNorm layers."""
        encoder = Encoder(num_bits=63)
        for module in encoder.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), \
                f"Found BatchNorm: {module}"

    def test_num_bits_attribute(self) -> None:
        """Encoder stores num_bits attribute."""
        encoder = Encoder(num_bits=63)
        assert encoder.num_bits == 63

    def test_default_num_bits_is_63(self) -> None:
        """Default num_bits is 63 for PicodeLite."""
        encoder = Encoder()
        assert encoder.num_bits == 63


class TestMessagePreparation:
    """Test message preparation (bits -> spatial tensor) with learned upsampling."""

    @pytest.fixture
    def sample_message_63(self) -> torch.Tensor:
        """Random 63-bit message batch."""
        return torch.randint(0, 2, (2, 63)).float()

    def test_output_shape(self, sample_message_63: torch.Tensor) -> None:
        """Message prep outputs (B, 3, 800, 800) tensor."""
        encoder = Encoder(num_bits=63)
        result = encoder.prepare_message(sample_message_63, target_size=(800, 800))
        assert result.shape == (2, 3, 800, 800)

    def test_output_shape_batch_1(self) -> None:
        """Works with batch size 1."""
        encoder = Encoder(num_bits=63)
        message = torch.randint(0, 2, (1, 63)).float()
        result = encoder.prepare_message(message, target_size=(800, 800))
        assert result.shape == (1, 3, 800, 800)

    def test_output_shape_different_sizes(self) -> None:
        """Message prep works with different target sizes."""
        encoder = Encoder(num_bits=63)
        message = torch.randint(0, 2, (1, 63)).float()
        # Test with 400x400 (derived from input, not hardcoded)
        result = encoder.prepare_message(message, target_size=(400, 400))
        assert result.shape == (1, 3, 400, 400)

    def test_smooth_output(self, sample_message_63: torch.Tensor) -> None:
        """Learned upsampling produces smooth output (low gradient magnitude).

        Transposed convolutions should produce smoother spatial patterns
        than nearest-neighbor upsampling, reducing wave artifacts.
        """
        encoder = Encoder(num_bits=63)
        result = encoder.prepare_message(sample_message_63, target_size=(800, 800))

        # Compute spatial gradients (Sobel-like)
        dx = result[:, :, :, 1:] - result[:, :, :, :-1]
        dy = result[:, :, 1:, :] - result[:, :, :-1, :]

        # Gradient magnitude should be relatively low for smooth output
        grad_mag = (dx.abs().mean() + dy.abs().mean()) / 2
        # This is a soft check - learned upsampling should be smoother than
        # nearest-neighbor, but we just ensure it's not wildly discontinuous
        assert grad_mag < 1.0, f"Gradient magnitude too high: {grad_mag}"


class TestEncoderForward:
    """Test full encoder forward pass."""

    @pytest.fixture
    def sample_image_800(self) -> torch.Tensor:
        """Random 800x800 RGB image batch."""
        return torch.rand(2, 3, 800, 800)

    @pytest.fixture
    def sample_message_63(self) -> torch.Tensor:
        """Random 63-bit message batch."""
        return torch.randint(0, 2, (2, 63)).float()

    def test_output_shape(
        self, sample_image_800: torch.Tensor, sample_message_63: torch.Tensor
    ) -> None:
        """Encoder outputs same shape as input image (B, 3, 800, 800)."""
        encoder = Encoder(num_bits=63)
        result = encoder(sample_image_800, sample_message_63)
        assert result.shape == sample_image_800.shape
        assert result.shape == (2, 3, 800, 800)

    def test_output_differs_from_input(
        self, sample_image_800: torch.Tensor, sample_message_63: torch.Tensor
    ) -> None:
        """Output differs from input (message is embedded)."""
        encoder = Encoder(num_bits=63)
        result = encoder(sample_image_800, sample_message_63)
        residual = result - sample_image_800
        # Residual should not be all zeros
        assert not torch.allclose(residual, torch.zeros_like(residual)), \
            "Output should differ from input (message must be embedded)"

    def test_gradient_flow(
        self, sample_image_800: torch.Tensor, sample_message_63: torch.Tensor
    ) -> None:
        """Gradients flow through encoder."""
        encoder = Encoder(num_bits=63)
        sample_image_800.requires_grad_(True)
        result = encoder(sample_image_800, sample_message_63)
        loss = result.sum()
        loss.backward()
        assert sample_image_800.grad is not None
        assert not torch.all(sample_image_800.grad == 0)

    def test_different_messages_different_encodings(
        self, sample_image_800: torch.Tensor
    ) -> None:
        """Different messages produce different encodings."""
        encoder = Encoder(num_bits=63)
        message1 = torch.zeros(2, 63)
        message2 = torch.ones(2, 63)
        result1 = encoder(sample_image_800, message1)
        result2 = encoder(sample_image_800, message2)
        # Different messages should produce different outputs
        assert not torch.allclose(result1, result2), \
            "Different messages should produce different encodings"

    def test_output_not_clamped(
        self, sample_image_800: torch.Tensor, sample_message_63: torch.Tensor
    ) -> None:
        """Output is NOT clamped to allow gradient flow."""
        encoder = Encoder(num_bits=63)
        result = encoder(sample_image_800, sample_message_63)
        # Output should be image + residual (unbounded)
        # Most values should be near [0, 1] but some can exceed
        residual = result - sample_image_800
        # Residual should exist (not all zeros)
        assert not torch.allclose(residual, torch.zeros_like(residual))

    def test_batch_size_1(self) -> None:
        """Works with batch size 1."""
        encoder = Encoder(num_bits=63)
        image = torch.rand(1, 3, 800, 800)
        message = torch.randint(0, 2, (1, 63)).float()
        result = encoder(image, message)
        assert result.shape == (1, 3, 800, 800)

    def test_batch_size_4(self) -> None:
        """Works with batch size 4."""
        encoder = Encoder(num_bits=63)
        image = torch.rand(4, 3, 800, 800)
        message = torch.randint(0, 2, (4, 63)).float()
        result = encoder(image, message)
        assert result.shape == (4, 3, 800, 800)


class TestEncoderInit:
    """Test encoder initialization."""

    def test_kaiming_normal_init(self) -> None:
        """Weights are initialized with Kaiming normal."""
        encoder = Encoder(num_bits=63)
        # Check that conv weights have reasonable variance (not all zeros or ones)
        for name, module in encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                weight = module.weight
                # Kaiming normal should have non-trivial variance
                assert weight.std() > 0.01, f"Conv {name} weights may not be initialized"
                assert weight.std() < 1.0, f"Conv {name} weights may have too high variance"

    def test_biases_are_zero(self) -> None:
        """Biases are initialized to zero."""
        encoder = Encoder(num_bits=63)
        for name, module in encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias)), \
                        f"Bias {name} should be zero initialized"
