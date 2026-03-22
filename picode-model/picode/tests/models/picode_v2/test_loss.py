"""Tests for picode_v2 loss functions."""

import torch

from picode.models.picode_v2.discriminator import PatchDiscriminator
from picode.models.picode_v2.loss import (
    FocalFrequencyLoss,
    discriminator_loss,
    generator_loss,
)


class TestFocalFrequencyLoss:
    """Tests for FocalFrequencyLoss."""

    def test_identical_images_zero_loss(self) -> None:
        """Loss is zero for identical images."""
        ffl = FocalFrequencyLoss()
        x = torch.rand(2, 3, 64, 64)
        loss = ffl(x, x)
        assert loss.item() < 1e-6

    def test_different_images_positive_loss(self) -> None:
        """Loss is positive for different images."""
        ffl = FocalFrequencyLoss()
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss = ffl(x, y)
        assert loss.item() > 0

    def test_gradient_flow(self) -> None:
        """Gradients flow through loss."""
        ffl = FocalFrequencyLoss()
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)
        loss = ffl(x, y)
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().mean() > 0

    def test_output_is_scalar(self) -> None:
        """Loss output is a scalar tensor."""
        ffl = FocalFrequencyLoss()
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss = ffl(x, y)
        assert loss.dim() == 0

    def test_different_alpha_values(self) -> None:
        """Works with different alpha (focal weight) values."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        for alpha in [0.5, 1.0, 2.0]:
            ffl = FocalFrequencyLoss(alpha=alpha)
            loss = ffl(x, y)
            assert loss.item() > 0

    def test_symmetric(self) -> None:
        """Loss is symmetric: ffl(x, y) == ffl(y, x)."""
        ffl = FocalFrequencyLoss()
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss_xy = ffl(x, y)
        loss_yx = ffl(y, x)
        assert torch.allclose(loss_xy, loss_yx, atol=1e-5)


class TestGANLosses:
    """Tests for GAN loss functions."""

    def test_discriminator_loss_computes(self) -> None:
        """Discriminator loss computes without error."""
        disc = PatchDiscriminator()
        real = torch.rand(2, 3, 64, 64)
        fake = torch.rand(2, 3, 64, 64)
        loss = discriminator_loss(disc, real, fake)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_generator_loss_computes(self) -> None:
        """Generator loss computes without error."""
        disc = PatchDiscriminator()
        fake = torch.rand(2, 3, 64, 64)
        fake_pred = disc(fake)
        loss = generator_loss(fake_pred)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_discriminator_loss_gradient_to_disc(self) -> None:
        """Gradients flow to discriminator parameters."""
        disc = PatchDiscriminator()
        real = torch.rand(2, 3, 64, 64)
        fake = torch.rand(2, 3, 64, 64)
        loss = discriminator_loss(disc, real, fake)
        loss.backward()
        # Check first conv has gradients
        first_conv = disc.model[0][0]
        assert first_conv.weight.grad is not None
        assert first_conv.weight.grad.abs().mean() > 0

    def test_generator_loss_gradient_to_input(self) -> None:
        """Gradients flow from generator loss to input."""
        disc = PatchDiscriminator()
        fake = torch.rand(2, 3, 64, 64, requires_grad=True)
        fake_pred = disc(fake)
        loss = generator_loss(fake_pred)
        loss.backward()
        assert fake.grad is not None
        assert fake.grad.abs().mean() > 0
