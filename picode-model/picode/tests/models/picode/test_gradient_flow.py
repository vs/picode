# picode/tests/models/picode/test_gradient_flow.py
"""Tests comparing gradient flow between Picode and StegaStamp."""

from typing import cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from picode.models.picode import Decoder as PicodeDecoder
from picode.models.picode import Encoder as PicodeEncoder
from picode.models.stegastamp import Decoder as StegaDecoder
from picode.models.stegastamp import Encoder as StegaEncoder


class TestGradientFlowComparison:
    """Compare gradient flow between Picode and StegaStamp."""

    @pytest.fixture
    def sample_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample data for testing."""
        torch.manual_seed(42)
        images = torch.rand(2, 3, 400, 400)
        messages = torch.randint(0, 2, (2, 100)).float()
        return images, messages

    def _compute_gradient_ratio(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        images: torch.Tensor,
        messages: torch.Tensor,
    ) -> float:
        """Compute encoder/decoder gradient ratio."""
        # Forward pass
        encoded = encoder(images, messages)
        decoded_logits = decoder(encoded)

        # Compute loss and backprop
        loss = F.binary_cross_entropy_with_logits(decoded_logits, messages)
        loss.sum().backward()

        # Sum gradient magnitudes
        enc_grad = sum(
            p.grad.abs().mean().item()
            for p in encoder.parameters()
            if p.grad is not None
        )
        dec_grad = sum(
            p.grad.abs().mean().item()
            for p in decoder.parameters()
            if p.grad is not None
        )

        return enc_grad / dec_grad if dec_grad > 0 else 0.0

    def test_picode_better_gradient_ratio(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Picode should have better encoder/decoder gradient ratio."""
        images, messages = sample_data

        # StegaStamp
        stega_enc = StegaEncoder(num_bits=100)
        stega_dec = StegaDecoder(num_bits=100)
        stega_ratio = self._compute_gradient_ratio(stega_enc, stega_dec, images.clone(), messages)

        # Picode
        picode_enc = PicodeEncoder(num_bits=100)
        picode_dec = PicodeDecoder(num_bits=100)
        picode_ratio = self._compute_gradient_ratio(
            picode_enc, picode_dec, images.clone(), messages
        )

        # Picode should have higher ratio (target: >0.2 vs StegaStamp's ~0.06)
        assert picode_ratio > stega_ratio, (
            f"Picode {picode_ratio:.4f} should be > StegaStamp {stega_ratio:.4f}"
        )
        assert picode_ratio > 0.1, f"Picode ratio {picode_ratio:.4f} should be > 0.1"

    def test_picode_stn_receives_gradients_step_one(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Picode STN should receive gradients from step 1."""
        images, messages = sample_data

        picode_dec = PicodeDecoder(num_bits=100)

        # Forward + backward
        logits = picode_dec(images)
        logits.sum().backward()

        # STN first conv should have non-zero gradients
        stn_first_conv = cast(nn.Conv2d, picode_dec.stn_params[0])
        grad = stn_first_conv.weight.grad
        assert grad is not None
        grad_mean = grad.abs().mean().item()

        assert grad_mean > 1e-8, f"STN first conv grad {grad_mean} should be > 1e-8"

    def test_stegastamp_stn_zero_gradients_step_one(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """StegaStamp STN has zero gradients on step 1 (known issue)."""
        images, messages = sample_data

        stega_dec = StegaDecoder(num_bits=100)

        # Forward + backward
        logits = stega_dec(images)
        logits.sum().backward()

        # STN first conv should have zero gradients due to zero-init weight
        stn_first_conv = cast(nn.Conv2d, stega_dec.stn_params[0])
        grad = stn_first_conv.weight.grad
        assert grad is not None
        grad_mean = grad.abs().mean().item()

        # This documents the known issue
        assert grad_mean < 1e-8, f"StegaStamp STN should have ~0 grad, got {grad_mean}"
