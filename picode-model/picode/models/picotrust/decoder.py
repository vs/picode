"""PicoTrust decoder: pretrained ResNet50 with compact STN.

Architecture:
- Compact STN: 3 stride-2 convs -> AdaptiveAvgPool2d(1) -> FC(128->128) -> affine (2x3)
  Uses GAP instead of flatten+FC, eliminating 67M parameters from StegaStamp's STN.
- ResNet50 backbone (ImageNet pretrained) -> global avg pool -> 2048-dim features
- Custom head: FC(2048->512) + ReLU -> FC(512->num_bits) -> logits
- Output: raw logits (no sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """ResNet50 decoder with compact STN for perspective correction.

    Uses AdaptiveAvgPool2d in the STN to eliminate the massive FC layer
    that dominates parameter count in StegaStamp's decoder. The pretrained
    ResNet50 backbone provides strong feature extraction from the start.

    Args:
        num_bits: Number of bits in the message (default: 100).
        freeze_stn_linear: If True, freeze STN affine parameters.
    """

    def __init__(
        self, num_bits: int = 100, freeze_stn_linear: bool = False,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Compact STN: resolution-independent via AdaptiveAvgPool2d
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # STN affine parameters (identity init)
        self.stn_fc_weight = nn.Parameter(torch.zeros(128, 6))
        self.stn_fc_bias = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        if freeze_stn_linear:
            self.stn_fc_weight.requires_grad = False
            self.stn_fc_bias.requires_grad = False

        # ResNet50 backbone (pretrained on ImageNet)
        import torchvision.models as models

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Keep everything except final FC: conv1->bn1->relu->maxpool->layer1-4->avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Output: (B, 2048, 1, 1)

        # Custom classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_bits),
        )

        # Initialize custom layers only (NOT the pretrained backbone)
        self._init_custom_weights()

    def _init_custom_weights(self) -> None:
        """Initialize STN and head weights with Kaiming normal.

        Does NOT touch pretrained ResNet50 backbone weights.
        """
        for m in self.stn_params.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
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
            Message logits (B, num_bits) -- unbounded.
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

        # Decode: ResNet50 backbone + custom head
        features = self.backbone(transformed)
        logits: Tensor = self.head(features)
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
