"""Tests for PatchGAN discriminator."""

import torch

from picode.models.stegastamp.patchgan import PatchGANDiscriminator


def test_patchgan_output_is_spatial():
    disc = PatchGANDiscriminator()
    image = torch.rand(2, 3, 512, 512)
    scores = disc(image)
    assert scores.dim() == 4
    assert scores.shape[0] == 2
    assert scores.shape[1] == 1


def test_patchgan_gradient_flow():
    disc = PatchGANDiscriminator()
    image = torch.rand(2, 3, 512, 512, requires_grad=True)
    scores = disc(image)
    scores.mean().backward()
    assert image.grad is not None


def test_patchgan_works_at_256():
    disc = PatchGANDiscriminator()
    image = torch.rand(2, 3, 256, 256)
    scores = disc(image)
    assert scores.dim() == 4


def test_patchgan_lsgan_loss():
    disc = PatchGANDiscriminator()
    real = torch.rand(2, 3, 256, 256)
    fake = torch.rand(2, 3, 256, 256)
    d_real = disc(real)
    d_fake = disc(fake)
    loss_d = 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()
    assert loss_d.item() >= 0
    loss_g = 0.5 * ((d_fake - 1) ** 2).mean()
    assert loss_g.item() >= 0
