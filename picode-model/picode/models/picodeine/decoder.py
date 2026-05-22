"""Picodeine decoder with Spatial Transformer Network.

StegaStamp-style CNN decoder adapted for 512x512 input and 127 bits.
Includes STN for geometric correction during camera capture.

Architecture:
- STN: 3 stride-2 convs → flatten → FC(128) → affine (2x3)
- CNN: 5 stride-2 + 2 stride-1 convs → flatten → FC(512) → FC(127)
- Output: raw logits (no sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """CNN decoder with STN that extracts message bits from an encoded image.

    Args:
        num_bits: Number of bits in the message (default: 127).
        input_size: Expected input spatial dimension (default: 512).
        freeze_stn_linear: If True, freeze the STN linear transformation parameters.
    """

    def __init__(
        self, num_bits: int = 127, input_size: int = 512,
        freeze_stn_linear: bool = False,
    ) -> None:
        super().__init__()
        if input_size % 32 != 0:
            raise ValueError(
                f"Picodeine decoder requires input_size divisible by 32 "
                f"(5 stride-2 convolutions reduce spatial by 2^5). Got {input_size}."
            )
        self.num_bits = num_bits

        # STN parameter predictor
        stn_spatial = input_size // 8  # 512 → 64
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # /8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * stn_spatial * stn_spatial, 128),
            nn.ReLU(),
        )

        # STN affine transform parameters (identity init)
        self.stn_fc_weight = nn.Parameter(torch.zeros(128, 6))
        self.stn_fc_bias = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        if freeze_stn_linear:
            self.stn_fc_weight.requires_grad = False
            self.stn_fc_bias.requires_grad = False

        # Main decoder CNN — 5 stride-2 + 2 stride-1 = 7 total
        # 512 → 256 → 256 → 128 → 128 → 64 → 32 → 16
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # /2
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),             # same
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # /2
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),             # same
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   # /2
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), # /2
            nn.ReLU(),
            nn.Flatten(),
        )

        # FC head: 5 stride-2 convs reduce spatial by 2^5 = 32
        spatial = input_size // 32  # 512 → 16
        flatten_size = 128 * spatial * spatial

        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_bits),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def stn_scale_reg(self) -> Tensor:
        """Compute STN regularization loss toward identity transform.

        Returns:
            Scalar regularization loss.
        """
        identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=self.stn_fc_bias.device
        )
        bias_reg = F.mse_loss(self.stn_fc_bias, identity)
        weight_reg = (self.stn_fc_weight**2).mean()
        return bias_reg + weight_reg

    def forward(self, image: Tensor) -> Tensor:
        """Extract message logits from image.

        Args:
            image: (B, 3, H, W) in [0, 1].

        Returns:
            Message logits (B, num_bits) — unbounded.
        """
        image_norm = image - 0.5

        # STN: predict affine transform
        stn_features = self.stn_params(image_norm)
        theta = torch.mm(stn_features, self.stn_fc_weight) + self.stn_fc_bias
        theta = theta.view(-1, 2, 3)

        # Apply spatial transform
        grid = F.affine_grid(theta, list(image_norm.size()), align_corners=False)
        transformed = F.grid_sample(
            image_norm, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )

        # Decode
        x = self.decoder(transformed)
        logits: Tensor = self.fc(x)
        return logits

    def decode(self, image: Tensor) -> Tensor:
        """Extract binary message from an image.

        Args:
            image: (B, C, H, W) in [0, 1].

        Returns:
            Binary message tensor (B, num_bits).
        """
        logits = self.forward(image)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()

    def unfreeze_stn_linear(self) -> None:
        """Unfreeze the STN linear parameters."""
        self.stn_fc_weight.requires_grad = True
        self.stn_fc_bias.requires_grad = True

    def freeze_stn_linear(self) -> None:
        """Freeze the STN linear parameters."""
        self.stn_fc_weight.requires_grad = False
        self.stn_fc_bias.requires_grad = False
