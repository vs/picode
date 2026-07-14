"""PicoComposite encoder: PicoTrust U-Net with tier conditioning.

Changes from PicoTrust encoder:
- Tier embedding concatenated with message before secret_dense
- Tier embedding projected and added to U-Net bottleneck
- Per-sample softsign strength looked up from tier table
- Bilinear message upsampling (from v10)
- Dilated E_post for larger receptive field (from v10)
- Zero-init E_post final layer so residual starts at zero (from v15)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder
from picode.models.picomposite.tiers import EMBED_DIM, MAX_BITS, NUM_TIERS, TIERS


class Encoder(BaseEncoder):
    """U-Net encoder with tier-conditioned residual strength.

    Args:
        image_size: Target image size (default: 512).
    """

    tier_strengths: Tensor  # registered buffer — declared for mypy

    def __init__(self, image_size: int = 512) -> None:
        super().__init__()
        self.image_size = image_size

        # Tier embedding
        self.tier_embedding = nn.Embedding(NUM_TIERS, EMBED_DIM)

        # Message preparation: (MAX_BITS + EMBED_DIM) -> 7500 -> (3, 50, 50)
        self.secret_dense = nn.Linear(MAX_BITS + EMBED_DIM, 7500)

        # Bottleneck conditioning: project tier embedding to conv5 channels (256)
        self.bottleneck_proj = nn.Linear(EMBED_DIM, 256)

        # Encoder (downsampling path)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Decoder (upsampling path)
        self.up6 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.up7 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.up8 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.up9 = nn.Conv2d(32, 32, 2, padding=0)
        self.conv9 = nn.Conv2d(70, 32, 3, padding=1)  # 32 + 32 + 6 = 70

        # E_post: grayscale residual with dilated conv for larger receptive field
        self.e_post = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),  # dilated, effective RF=7
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )

        # Build strength lookup tensor (registered as buffer for device tracking)
        strengths = torch.tensor(
            [TIERS[i]["strength"] for i in range(NUM_TIERS)], dtype=torch.float32
        )
        self.register_buffer("tier_strengths", strengths)

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

        # Zero-init E_post's final layer so residual starts at zero
        last_conv = self.e_post[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, image: Tensor, message: Tensor, tier: Tensor) -> dict[str, Tensor]:  # type: ignore[override]
        """Encode message into image with tier-specific strength.

        Args:
            image: (B, 3, H, W) in [0, 1].
            message: (B, MAX_BITS) zero-padded binary tensor.
            tier: (B,) int tensor — tier index per sample.

        Returns:
            Dict with "encoded" key: (B, 3, H, W).
        """
        # Normalize inputs
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Get tier embeddings
        tier_embed = self.tier_embedding(tier)  # (B, EMBED_DIM)

        # Concatenate message + tier embedding before projection
        msg_input = torch.cat([message_norm, tier_embed], dim=1)  # (B, MAX_BITS + EMBED_DIM)
        x = F.relu(self.secret_dense(msg_input))  # (B, 7500)
        x = x.view(-1, 3, 50, 50)
        secret_enlarged = F.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, H, W)

        # Encoder path
        c1 = F.relu(self.conv1(inputs))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))

        # Bottleneck conditioning: add projected tier embedding
        bottleneck_bias = self.bottleneck_proj(tier_embed)  # (B, 256)
        c5 = c5 + bottleneck_bias.unsqueeze(-1).unsqueeze(-1)  # broadcast (B, 256, 1, 1)

        # Decoder path with skip connections
        x = F.interpolate(c5, scale_factor=2, mode="nearest")
        x = F.relu(self.up6(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c4, x], dim=1)
        x = F.relu(self.conv6(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up7(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c3, x], dim=1)
        x = F.relu(self.conv7(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up8(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c2, x], dim=1)
        x = F.relu(self.conv8(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up9(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, inputs], dim=1)  # 32 + 32 + 6 = 70
        x = F.relu(self.conv9(x))

        # E_post -> 1-channel grayscale residual
        raw_residual = self.e_post(x)  # (B, 1, H, W)
        raw_residual = raw_residual.expand(-1, 3, -1, -1)

        # Per-sample softsign with tier-specific strength
        per_strength = self.tier_strengths[tier]  # (B,)
        per_strength = per_strength.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting
        residual = per_strength * raw_residual / (1.0 + raw_residual.abs())

        encoded = image + residual
        return {"encoded": encoded}
