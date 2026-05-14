"""PicodeFrame decoder network.

Based on the StegaStamp decoder with STN and CNN architecture.
Key differences from StegaStamp decoder:
- 96 output bits instead of 100
- STN scale regularization to prevent zoom-in that would crop the frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """CNN decoder with STN that extracts message bits from a framed image.

    Architecture matches StegaStamp decoder with 96-bit output and
    STN scale regularization to prevent the STN from learning to
    zoom in and crop the message-bearing frame.

    Args:
        num_bits: Number of bits in the message (default: 96).
        height: Image height for STN output (default: 400).
        width: Image width for STN output (default: 400).
        freeze_stn_linear: If True, freeze the STN linear transformation parameters.
    """

    def __init__(
        self, num_bits: int = 96, height: int = 400, width: int = 400,
        freeze_stn_linear: bool = False,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.height = height
        self.width = width

        # Spatial Transformer Network (STN) parameter predictor
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 200x200
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 100x100
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 50x50
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 50 * 50, 128),
            nn.ReLU(),
        )

        # STN affine transform parameters
        # Initialized to identity transform: [[1, 0, 0], [0, 1, 0]]
        self.stn_fc_weight = nn.Parameter(torch.zeros(128, 6))
        self.stn_fc_bias = nn.Parameter(torch.tensor([1., 0., 0., 0., 1., 0.]))

        if freeze_stn_linear:
            self.stn_fc_weight.requires_grad = False
            self.stn_fc_bias.requires_grad = False

        # Main decoder CNN - no BatchNorm
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 200x200
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 100x100
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 50x50
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 25x25
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 13x13
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 13 * 13, 512),
            nn.ReLU(),
            nn.Linear(512, num_bits),  # Raw logits, no activation
        )

        # Initialize weights (He normal)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal (He normal)."""
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
        """Compute L2 regularization on STN scale deviation from identity.

        Penalizes the STN affine parameters for deviating from identity,
        particularly the scale components. This prevents the STN from
        learning to zoom in and crop the message-bearing frame.

        Returns:
            Scalar regularization loss.
        """
        # Current affine params: theta = features @ weight + bias
        # At initialization, weight=0 so theta = bias = [1, 0, 0, 0, 1, 0] (identity)
        # We penalize deviation of bias from identity and non-zero weight magnitude
        identity = torch.tensor([1., 0., 0., 0., 1., 0.], device=self.stn_fc_bias.device)
        bias_reg = F.mse_loss(self.stn_fc_bias, identity)
        weight_reg = (self.stn_fc_weight ** 2).mean()
        return bias_reg + weight_reg

    def forward(self, image: Tensor) -> Tensor:
        """Extract message logits from framed image.

        Args:
            image: (B, 3, H, W) in [0, 1]

        Returns:
            Message logits (B, num_bits) - unbounded, apply sigmoid for probabilities
        """
        # Normalize input
        image_norm = image - 0.5

        # Compute STN affine parameters
        stn_features = self.stn_params(image_norm)
        theta = torch.mm(stn_features, self.stn_fc_weight) + self.stn_fc_bias
        theta = theta.view(-1, 2, 3)

        # Apply spatial transform
        grid = F.affine_grid(theta, list(image_norm.size()), align_corners=False)
        transformed = F.grid_sample(
            image_norm, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )

        # Decode from transformed image
        logits: Tensor = self.decoder(transformed)
        return logits

    def decode(self, image: Tensor) -> Tensor:
        """Extract binary message from an image.

        Args:
            image: Input image tensor (B, C, H, W) in [0, 1].

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
