"""PatchGAN discriminator for LSGAN training.

Classifies each NxN patch as real/fake independently, producing a spatial
score map. Used with LSGAN loss (least-squares) for stable training.

Architecture based on pix2pix PatchGAN (Isola et al., CVPR 2017).
"""

import torch.nn as nn
from torch import Tensor


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with LSGAN-compatible output.

    Outputs a spatial score map where each value represents the
    "realness" score for the corresponding image patch.
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64) -> None:
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: no norm on first layer
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: stride 1
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 1-channel score map
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with Gaussian weights (std=0.02) per pix2pix convention."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: Tensor) -> Tensor:
        """Compute per-patch realness scores.

        Args:
            image: (B, 3, H, W) in [0, 1].

        Returns:
            (B, 1, H', W') spatial score map.
        """
        x = image - 0.5
        return self.model(x)
