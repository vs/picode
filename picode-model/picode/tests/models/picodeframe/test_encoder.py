"""Tests for the PicodeFrame encoder module."""

import pytest
import torch
import torch.nn as nn

from picode.models.picodeframe.encoder import Encoder


class TestEncoderArchitecture:
    """Test encoder architecture."""

    def test_no_batchnorm(self) -> None:
        """Encoder has no BatchNorm layers."""
        encoder = Encoder(num_bits=96)
        for module in encoder.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), \
                f"Found BatchNorm: {module}"

    def test_weight_initialization(self) -> None:
        """Weights use Kaiming normal initialization."""
        encoder = Encoder(num_bits=96)
        conv_weight = encoder.conv1.weight
        assert conv_weight.std() > 0.01, "Weights appear uninitialized"

    def test_seven_channel_input(self) -> None:
        """First conv layer accepts 7 input channels."""
        encoder = Encoder(num_bits=96)
        assert encoder.conv1.in_channels == 7

    def test_num_bits_attribute(self) -> None:
        """Encoder stores num_bits attribute."""
        encoder = Encoder(num_bits=96)
        assert encoder.num_bits == 96


class TestMessagePreparation:
    """Test message preparation (bits -> spatial tensor)."""

    def test_output_shape(self, sample_message: torch.Tensor) -> None:
        """Message prep outputs (B, 3, 400, 400) tensor."""
        encoder = Encoder(num_bits=96)
        result = encoder.prepare_message(sample_message)
        assert result.shape == (2, 3, 400, 400)


class TestEncoderForward:
    """Test full encoder forward pass."""

    def test_output_shape(
        self, padded_image: torch.Tensor, sample_message: torch.Tensor, frame_width: int
    ) -> None:
        """Encoder outputs same shape as input image (400x400)."""
        encoder = Encoder(num_bits=96)
        result = encoder(padded_image, sample_message, frame_width=frame_width)
        assert result.shape == padded_image.shape
        assert result.shape == (2, 3, 400, 400)

    def test_center_pixels_preserved(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor, frame_width: int
    ) -> None:
        """Center pixels are exactly preserved (hard mask guarantee)."""
        encoder = Encoder(num_bits=96)
        fw = frame_width

        # Extract inner and pad
        inner = sample_image[:, :, fw:-fw, fw:-fw]
        padded = torch.nn.functional.pad(inner, (fw, fw, fw, fw), mode="reflect")

        result = encoder(padded, sample_message, frame_width=fw)

        # Center pixels must be exactly equal to the padded image center
        # (which equals the original inner image)
        assert torch.equal(
            result[:, :, fw:-fw, fw:-fw],
            padded[:, :, fw:-fw, fw:-fw],
        ), "Center pixels were modified!"

    @pytest.mark.parametrize("fw", [8, 16, 20])
    def test_different_frame_widths(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor, fw: int
    ) -> None:
        """Works with different frame widths."""
        encoder = Encoder(num_bits=96)

        inner = sample_image[:, :, fw:-fw, fw:-fw]
        padded = torch.nn.functional.pad(inner, (fw, fw, fw, fw), mode="reflect")

        result = encoder(padded, sample_message, frame_width=fw)
        assert result.shape == (2, 3, 400, 400)

        # Verify center preservation
        assert torch.equal(
            result[:, :, fw:-fw, fw:-fw],
            padded[:, :, fw:-fw, fw:-fw],
        )

    def test_gradient_flow_through_frame(
        self, sample_message: torch.Tensor, frame_width: int
    ) -> None:
        """Gradients flow through frame pixels."""
        encoder = Encoder(num_bits=96)
        fw = frame_width

        image = torch.rand(2, 3, 400, 400, requires_grad=True)
        inner = image[:, :, fw:-fw, fw:-fw]
        padded = torch.nn.functional.pad(inner, (fw, fw, fw, fw), mode="reflect")

        result = encoder(padded, sample_message, frame_width=fw)

        # Loss only on frame pixels
        frame_mask = torch.ones_like(result)
        frame_mask[:, :, fw:-fw, fw:-fw] = 0
        loss = (result * frame_mask).sum()
        loss.backward()

        assert image.grad is not None

    def test_frame_pixels_differ_from_input(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor, frame_width: int
    ) -> None:
        """Frame pixels should differ from the reflection-padded input."""
        encoder = Encoder(num_bits=96)
        fw = frame_width

        inner = sample_image[:, :, fw:-fw, fw:-fw]
        padded = torch.nn.functional.pad(inner, (fw, fw, fw, fw), mode="reflect")

        result = encoder(padded, sample_message, frame_width=fw)

        # Frame region should have some residual (not identical to input)
        frame_mask = torch.ones_like(result)
        frame_mask[:, :, fw:-fw, fw:-fw] = 0
        frame_diff = ((result - padded) * frame_mask).abs().sum()
        assert frame_diff > 0, "Frame pixels are identical to input (no encoding)"
