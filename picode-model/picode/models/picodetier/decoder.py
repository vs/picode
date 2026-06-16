"""PicodeTier decoder: PicoTrust CNN with tier classifier and conditioned output.

Changes from PicoTrust decoder:
- nn.Sequential split into backbone + heads
- Tier classifier head: FC(512, NUM_TIERS)
- Tier-conditioned output: cat(features, tier_embed) -> FC(544, 512) -> FC(512, MAX_BITS)
- decode() returns (tier_index, variable-length bits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder
from picode.models.picodetier.tiers import EMBED_DIM, MAX_BITS, NUM_TIERS, TIERS


class Decoder(BaseDecoder):
    """CNN decoder with tier auto-detection and conditioned bit extraction.

    Args:
        image_size: Input image spatial size (default: 256).
        freeze_stn_linear: If True, freeze STN affine parameters.
    """

    def __init__(
        self, image_size: int = 256, freeze_stn_linear: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size

        # Tier embedding (independent from encoder's embedding)
        self.tier_embedding = nn.Embedding(NUM_TIERS, EMBED_DIM)

        # Compact STN (same as PicoTrust)
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

        self.stn_fc_weight = nn.Parameter(torch.zeros(128, 6))
        self.stn_fc_bias = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        if freeze_stn_linear:
            self.stn_fc_weight.requires_grad = False
            self.stn_fc_bias.requires_grad = False

        # Main CNN backbone (same conv layers as PicoTrust, split from Sequential)
        spatial = image_size // 32
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * spatial * spatial, 512),
            nn.ReLU(),
        )

        # Tier classifier head
        self.tier_classifier = nn.Linear(512, NUM_TIERS)

        # Tier-conditioned bit output head
        self.bit_head = nn.Sequential(
            nn.Linear(512 + EMBED_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, MAX_BITS),
        )

        self._init_weights()

    def _init_weights(self) -> None:
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
        """Compute STN regularization loss toward identity transform."""
        identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=self.stn_fc_bias.device
        )
        bias_reg = F.mse_loss(self.stn_fc_bias, identity)
        weight_reg = (self.stn_fc_weight**2).mean()
        return bias_reg + weight_reg

    def forward(
        self, image: Tensor, tier: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Extract message logits and tier classification from image.

        Args:
            image: (B, 3, H, W) in [0, 1].
            tier: (B,) int tensor — ground truth tier (training) or None (inference).
                  When None, uses argmax of tier_classifier.

        Returns:
            Tuple of:
            - bit_logits: (B, MAX_BITS) — raw logits for all 96 bit positions
            - tier_logits: (B, NUM_TIERS) — raw logits for tier classification
        """
        image_norm = image - 0.5

        # STN
        stn_features = self.stn_params(image_norm)
        theta = torch.mm(stn_features, self.stn_fc_weight) + self.stn_fc_bias
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, list(image_norm.size()), align_corners=False)
        transformed = F.grid_sample(
            image_norm, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )

        # Backbone
        features = self.backbone(transformed)  # (B, 512)

        # Tier classification
        tier_logits = self.tier_classifier(features)  # (B, NUM_TIERS)

        # Determine tier for conditioning
        if tier is None:
            tier = tier_logits.argmax(dim=1)  # (B,)

        # Tier-conditioned bit extraction
        tier_embed = self.tier_embedding(tier)  # (B, EMBED_DIM)
        conditioned = torch.cat([features, tier_embed], dim=1)  # (B, 512 + EMBED_DIM)
        bit_logits = self.bit_head(conditioned)  # (B, MAX_BITS)

        return bit_logits, tier_logits

    def decode(self, image: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Extract binary message with auto-detected tier.

        Args:
            image: (B, C, H, W) in [0, 1].

        Returns:
            Tuple of:
            - tier_idx: (B,) detected tier index per sample
            - bits: list of binary tensors, one per sample, each truncated to tier's bit count
        """
        bit_logits, tier_logits = self.forward(image)
        tier_idx = tier_logits.argmax(dim=1)  # (B,)
        probs = torch.sigmoid(bit_logits)
        hard_bits = (probs > 0.5).float()

        # Truncate each sample to its tier's bit count
        results = []
        for i in range(image.shape[0]):
            t = tier_idx[i].item()
            n_bits = int(TIERS[t]["bits"])
            results.append(hard_bits[i, :n_bits])

        return tier_idx, results

    def unfreeze_stn_linear(self) -> None:
        self.stn_fc_weight.requires_grad = True
        self.stn_fc_bias.requires_grad = True

    def freeze_stn_linear(self) -> None:
        self.stn_fc_weight.requires_grad = False
        self.stn_fc_bias.requires_grad = False
