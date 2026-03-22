"""Loss functions for picode_v2 training."""

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["FocalFrequencyLoss", "discriminator_loss", "generator_loss"]


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for reducing structured artifacts.

    Penalizes frequency-domain differences with focal weighting:
    harder frequencies (larger errors) get higher weight.

    Reference: Jiang et al., ICCV 2021 "Focal Frequency Loss"

    Args:
        alpha: Focal weight exponent (default: 1.0).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute focal frequency loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Scalar loss value
        """
        # 2D FFT
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')

        # Magnitude difference
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)
        freq_distance = (pred_mag - target_mag) ** 2

        # Focal weighting: harder frequencies get higher weight
        weight = freq_distance ** self.alpha
        weight = weight / (weight.sum() + 1e-8)

        # Weighted frequency loss
        loss = (weight * freq_distance).sum()
        return loss


def discriminator_loss(
    discriminator: nn.Module,
    real: Tensor,
    fake: Tensor,
    lambda_gp: float = 10.0,
) -> Tensor:
    """WGAN-GP discriminator loss.

    Args:
        discriminator: PatchDiscriminator module
        real: Real images (B, C, H, W)
        fake: Fake/encoded images (B, C, H, W)
        lambda_gp: Gradient penalty weight

    Returns:
        Discriminator loss scalar
    """
    real_pred = discriminator(real)
    fake_pred = discriminator(fake.detach())

    # Wasserstein distance
    d_loss = fake_pred.mean() - real_pred.mean()

    # Gradient penalty
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolates = alpha * real + (1 - alpha) * fake.detach()
    interpolates.requires_grad_(True)

    d_interp = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    total_loss: Tensor = d_loss + lambda_gp * gradient_penalty
    return total_loss


def generator_loss(fake_pred: Tensor) -> Tensor:
    """Generator (encoder) adversarial loss.

    Args:
        fake_pred: Discriminator predictions on fake/encoded images

    Returns:
        Generator loss scalar
    """
    loss: Tensor = -fake_pred.mean()
    return loss
