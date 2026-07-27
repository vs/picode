"""Tests for PicoGrain-specific loss functions."""

import torch

from picode.training.grain_losses import (
    blurred_l2_loss,
    envelope_smoothness_loss,
    luminance_fidelity_loss,
)


class TestBlurredL2Loss:

    def test_identical_images_zero_loss(self):
        img = torch.rand(2, 3, 64, 64)
        loss = blurred_l2_loss(img, img, blur_sigma=2.0)
        assert loss.item() < 1e-6

    def test_different_images_positive_loss(self):
        a = torch.rand(2, 3, 64, 64)
        b = torch.rand(2, 3, 64, 64)
        loss = blurred_l2_loss(a, b, blur_sigma=2.0)
        assert loss.item() > 0.0

    def test_grain_noise_low_loss(self):
        """High-frequency grain should produce low blurred L2 loss."""
        img = torch.rand(2, 3, 64, 64)
        grain = img + 0.05 * torch.randn_like(img)
        loss_grain = blurred_l2_loss(img, grain, blur_sigma=2.0)
        shifted = img + 0.05
        loss_shift = blurred_l2_loss(img, shifted, blur_sigma=2.0)
        assert loss_grain < loss_shift

    def test_output_is_scalar(self):
        img = torch.rand(2, 3, 64, 64)
        loss = blurred_l2_loss(img, img + 0.01, blur_sigma=2.0)
        assert loss.shape == ()

    def test_gradient_flow(self):
        img = torch.rand(2, 3, 64, 64)
        encoded = (img + 0.01 * torch.randn_like(img)).requires_grad_(True)
        loss = blurred_l2_loss(img, encoded, blur_sigma=2.0)
        loss.backward()
        assert encoded.grad is not None


class TestLuminanceFidelityLoss:

    def test_dark_area_residual_penalized(self):
        """Residual in dark areas should incur higher loss than bright areas."""
        dark_image = torch.ones(1, 3, 64, 64) * 0.1
        bright_image = torch.ones(1, 3, 64, 64) * 0.9
        residual = torch.ones(1, 1, 64, 64) * 0.05
        loss_dark = luminance_fidelity_loss(dark_image, residual, lum_floor=0.1)
        loss_bright = luminance_fidelity_loss(bright_image, residual, lum_floor=0.1)
        assert loss_dark > loss_bright

    def test_zero_residual_zero_loss(self):
        img = torch.rand(1, 3, 64, 64)
        residual = torch.zeros(1, 1, 64, 64)
        loss = luminance_fidelity_loss(img, residual, lum_floor=0.1)
        assert loss.item() < 1e-6

    def test_gradient_flow(self):
        img = torch.rand(1, 3, 64, 64)
        residual = torch.randn(1, 1, 64, 64, requires_grad=True)
        loss = luminance_fidelity_loss(img, residual, lum_floor=0.1)
        loss.backward()
        assert residual.grad is not None


class TestEnvelopeSmoothness:

    def test_smooth_envelope_low_loss(self):
        envelope = torch.ones(1, 1, 64, 64) * 0.5
        loss = envelope_smoothness_loss(envelope)
        assert loss.item() < 1e-6

    def test_noisy_envelope_high_loss(self):
        envelope = torch.randn(1, 1, 64, 64)
        loss = envelope_smoothness_loss(envelope)
        assert loss.item() > 0.01

    def test_gradient_flow(self):
        envelope = torch.randn(1, 1, 64, 64, requires_grad=True)
        loss = envelope_smoothness_loss(envelope)
        loss.backward()
        assert envelope.grad is not None
