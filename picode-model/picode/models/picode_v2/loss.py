"""Loss functions for picode_v2 training."""

import torch
import torch.nn as nn
from torch import Tensor


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
