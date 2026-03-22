"""PatchGAN discriminator for adversarial training."""

import torch.nn as nn
from torch import Tensor


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for real/encoded classification.

    Uses 70x70 receptive field patches to detect local artifacts.
    Training-only component - not used during inference.

    Args:
        in_channels: Number of input channels (default: 3).
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        def block(
            in_ch: int, out_ch: int, stride: int = 2, norm: bool = True
        ) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64, norm=False),   # 200x200
            block(64, 128),                        # 100x100
            block(128, 256),                       # 50x50
            block(256, 512, stride=1),             # 50x50
            nn.Conv2d(512, 1, 4, padding=1),       # 49x49
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute patch-wise real/fake predictions.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Patch predictions (B, 1, H', W')
        """
        out: Tensor = self.model(x)
        return out
