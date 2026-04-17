"""StegaStamp discriminator for WGAN training.

Matches original TensorFlow implementation:
- No normalization layers (WGAN requirement)
- Simple conv stack with ReLU
- Outputs scalar per image
"""

import torch.nn as nn
from torch import Tensor


class Discriminator(nn.Module):
    """WGAN discriminator for StegaStamp.

    Simple convolutional discriminator that outputs a scalar
    "realness" score per image. No normalization layers per
    WGAN requirements.

    Architecture matches original StegaStamp:
    - Conv(8) -> Conv(16) -> Conv(32) -> Conv(64) -> Conv(1)
    - All stride 2 except final
    - ReLU activations except final
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 400 -> 200
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 200 -> 100
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 100 -> 50
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 50 -> 25
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 25 -> 25 (no stride)
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: Tensor) -> Tensor:
        """Compute realness score for image.

        Args:
            image: (B, 3, H, W) image tensor in [0, 1].

        Returns:
            (B,) scalar realness score per image.
        """
        # Normalize input
        x = image - 0.5

        # Extract features
        features = self.features(x)  # (B, 1, H', W')

        # Global average to get scalar
        score: Tensor = features.mean(dim=(1, 2, 3))  # (B,)
        return score
