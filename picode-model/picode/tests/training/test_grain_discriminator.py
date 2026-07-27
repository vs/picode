"""Tests for PicoGrain patch discriminator."""

import torch

from picode.training.grain_discriminator import GrainPatchDiscriminator


def test_output_shape():
    disc = GrainPatchDiscriminator(in_channels=1)
    patch = torch.randn(4, 1, 64, 64)
    out = disc(patch)
    assert out.dim() == 4
    assert out.shape[0] == 4
    assert out.shape[1] == 1


def test_gradient_flow():
    disc = GrainPatchDiscriminator(in_channels=1)
    patch = torch.randn(2, 1, 64, 64, requires_grad=True)
    out = disc(patch)
    out.sum().backward()
    assert patch.grad is not None


def test_different_scores_for_different_input():
    disc = GrainPatchDiscriminator(in_channels=1)
    noise = torch.randn(2, 1, 64, 64)
    smooth = torch.ones(2, 1, 64, 64) * 0.1
    score_noise = disc(noise).mean()
    score_smooth = disc(smooth).mean()
    assert score_noise != score_smooth
