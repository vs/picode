"""Tests for PicoTrust v2 trainer features."""

import torch
import torch.nn.functional as F

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
