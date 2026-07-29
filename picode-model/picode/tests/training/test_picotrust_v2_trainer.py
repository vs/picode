"""Tests for PicoTrust v2 trainer features."""

import torch

from picode.training.trainer import Trainer


class TestSSIMComputation:

    def test_ssim_loss_identical_images(self):
        image = torch.rand(2, 3, 64, 64)
        loss = Trainer._compute_ssim_loss(image, image)
        assert loss.item() < 0.01

    def test_ssim_loss_different_images(self):
        a = torch.rand(2, 3, 64, 64)
        b = torch.rand(2, 3, 64, 64)
        loss = Trainer._compute_ssim_loss(a, b)
        assert loss.item() > 0.1

    def test_ssim_loss_has_gradient(self):
        a = torch.rand(2, 3, 64, 64)
        b = torch.rand(2, 3, 64, 64, requires_grad=True)
        loss = Trainer._compute_ssim_loss(a, b)
        loss.backward()
        assert b.grad is not None


class TestMaskRegularization:

    def test_mask_reg_computes(self):
        image = torch.rand(2, 3, 64, 64)
        mask = torch.rand(2, 1, 64, 64)
        loss = Trainer._compute_mask_reg_loss(mask, image)
        assert loss.item() >= 0

    def test_mask_reg_has_gradient(self):
        image = torch.rand(2, 3, 64, 64)
        mask = torch.rand(2, 1, 64, 64, requires_grad=True)
        loss = Trainer._compute_mask_reg_loss(mask, image)
        loss.backward()
        assert mask.grad is not None


class TestLaplacianLoss:

    def test_laplacian_zero_for_uniform_residual(self):
        """A spatially uniform residual should have near-zero Laplacian.

        Note: small non-zero value expected from zero-padding border effects.
        """
        residual = torch.ones(2, 3, 64, 64) * 0.01
        loss = Trainer._compute_laplacian_loss(residual)
        assert loss.item() < 1e-3

    def test_laplacian_high_for_checkerboard(self):
        """A high-frequency checkerboard pattern should have high Laplacian."""
        residual = torch.zeros(2, 3, 64, 64)
        residual[:, :, 0::2, 0::2] = 0.01
        residual[:, :, 1::2, 1::2] = 0.01
        residual[:, :, 0::2, 1::2] = -0.01
        residual[:, :, 1::2, 0::2] = -0.01
        loss = Trainer._compute_laplacian_loss(residual)
        assert loss.item() > 0.01

    def test_laplacian_has_gradient(self):
        """Laplacian loss should propagate gradients."""
        residual = torch.randn(2, 3, 64, 64, requires_grad=True)
        loss = Trainer._compute_laplacian_loss(residual)
        loss.backward()
        assert residual.grad is not None


class TestStrengthAnnealing:

    def test_strength_before_anneal(self):
        s = Trainer._compute_strength(
            initial=0.05, target=0.03, start_step=60000,
            anneal_steps=20000, current_step=50000,
        )
        assert s == 0.05

    def test_strength_after_anneal(self):
        s = Trainer._compute_strength(
            initial=0.05, target=0.03, start_step=60000,
            anneal_steps=20000, current_step=100000,
        )
        assert abs(s - 0.03) < 1e-6

    def test_strength_midway(self):
        s = Trainer._compute_strength(
            initial=0.05, target=0.03, start_step=60000,
            anneal_steps=20000, current_step=70000,
        )
        assert abs(s - 0.04) < 1e-6

    def test_exponential_before_anneal(self):
        s = Trainer._compute_strength(
            initial=1.0, target=0.014, start_step=10000,
            anneal_steps=120000, current_step=5000,
            schedule="exponential",
        )
        assert s == 1.0

    def test_exponential_after_anneal(self):
        s = Trainer._compute_strength(
            initial=1.0, target=0.014, start_step=10000,
            anneal_steps=120000, current_step=200000,
            schedule="exponential",
        )
        assert abs(s - 0.014) < 1e-6

    def test_exponential_midway(self):
        """At progress=0.5, exponential should give sqrt(target/initial) * initial."""
        s = Trainer._compute_strength(
            initial=1.0, target=0.014, start_step=10000,
            anneal_steps=120000, current_step=70000,
            schedule="exponential",
        )
        expected = 1.0 * (0.014 / 1.0) ** 0.5  # ~0.1183
        assert abs(s - expected) < 1e-4

    def test_linear_still_works_with_schedule_param(self):
        """Existing linear behavior unchanged when schedule='linear'."""
        s = Trainer._compute_strength(
            initial=0.05, target=0.03, start_step=60000,
            anneal_steps=20000, current_step=70000,
            schedule="linear",
        )
        assert abs(s - 0.04) < 1e-6

    def test_default_schedule_is_linear(self):
        """Omitting schedule param defaults to linear (backward compat)."""
        s = Trainer._compute_strength(
            initial=0.05, target=0.03, start_step=60000,
            anneal_steps=20000, current_step=70000,
        )
        assert abs(s - 0.04) < 1e-6
