"""PicoGrain patch discriminator for grain authenticity.

Operates on residual patches (encoded - original) to judge whether
the grain texture looks like real film grain. Based on PatchGAN
architecture but with 1-channel input (grayscale residuals).
"""

import torch.nn as nn
from torch import Tensor


class GrainPatchDiscriminator(nn.Module):
    """PatchGAN discriminator for grain texture classification.

    Args:
        in_channels: Input channels (1 for grayscale residual).
        ndf: Base number of discriminator filters.
    """

    def __init__(self, in_channels: int = 1, ndf: int = 64) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with Gaussian weights per pix2pix convention."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, residual: Tensor) -> Tensor:
        """Compute per-patch realness scores.

        Args:
            residual: (B, 1, H, W) grayscale residual patches.

        Returns:
            (B, 1, H', W') spatial score map.
        """
        return self.model(residual)  # type: ignore[no-any-return]
