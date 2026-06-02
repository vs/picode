"""PicodeFrame decoder network with border-pooling branch.

Hybrid architecture that combines:
1. CNN branch: 7-layer strided CNN (400→13×13) with STN for perspective correction
2. Border-pooling branch: strip-pooled border features at full resolution

The CNN branch alone cannot learn from fresh initialization because the 7-layer
spatial compression reduces the 32px border to ~1px in the feature map, drowning
the border signal in center noise. The border-pooling branch provides a direct
path for border information to reach the output, bypassing spatial compression.

Both branches are concatenated before the FC head, allowing the model to learn
from border features immediately while the CNN branch learns over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """Hybrid CNN + border-pooling decoder for framed steganography.

    The CNN branch processes the full image through strided convolutions with an
    STN for perspective correction. The border-pooling branch extracts features
    from the four border strips at full resolution by averaging along the narrow
    dimension (frame_width → 1) while preserving the long dimension (400 pixels).

    The border branch operates on the ORIGINAL image (not STN-transformed) to
    preserve border spatial information that would be distorted by the STN's
    bilinear interpolation.

    Args:
        num_bits: Number of bits in the message (default: 127 for BCH(127,64)).
        height: Image height (default: 400).
        width: Image width (default: 400).
        freeze_stn_linear: If True, freeze the STN linear transformation parameters.
    """

    def __init__(
        self, num_bits: int = 127, height: int = 400, width: int = 400,
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

        # CNN branch — 4 input channels (3 RGB + 1 border indicator)
        # 5 stride-2 + 2 stride-1 convolutions: 400→13×13
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),    # 200x200
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),             # 200x200
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 100x100
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),             # 100x100
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   # 50x50
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 25x25
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), # 13x13
            nn.ReLU(),
        )

        # Border-pooling branch: strip-pool each border side along the narrow
        # dimension (fw→1), keeping the long dimension (H or W pixels).
        # 4 strips × 3 channels × height = 4800 features for 400×400
        border_features = 3 * 4 * height
        self.border_fc = nn.Sequential(
            nn.Linear(border_features, 256),
            nn.ReLU(),
        )

        # Combined FC head: CNN features + border features
        cnn_flat = 128 * 13 * 13  # 21632
        self.fc1 = nn.Linear(cnn_flat + 256, 512)
        self.fc2 = nn.Linear(512, num_bits)

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

    def _pool_border_strips(self, image: Tensor, frame_width: int) -> Tensor:
        """Extract strip-pooled features from each border side.

        Averages each border strip along its narrow dimension (frame_width → 1)
        while preserving the long dimension. This gives 4 × 3 × H features
        that capture the full-resolution border pattern.

        Args:
            image: (B, 3, H, W) input image.
            frame_width: Width of the border frame in pixels.

        Returns:
            Flattened border features (B, 3 * 4 * H).
        """
        B, C, H, W = image.shape
        fw = frame_width
        top = image[:, :, :fw, :].mean(dim=2)      # (B, 3, W)
        bottom = image[:, :, H - fw:, :].mean(dim=2)  # (B, 3, W)
        left = image[:, :, :, :fw].mean(dim=3)      # (B, 3, H)
        right = image[:, :, :, W - fw:].mean(dim=3)   # (B, 3, H)
        return torch.cat([
            top.reshape(B, -1), bottom.reshape(B, -1),
            left.reshape(B, -1), right.reshape(B, -1),
        ], dim=1)

    def stn_scale_reg(self) -> Tensor:
        """Compute L2 regularization on STN scale deviation from identity.

        Penalizes the STN affine parameters for deviating from identity,
        particularly the scale components. This prevents the STN from
        learning to zoom in and crop the message-bearing frame.

        Returns:
            Scalar regularization loss.
        """
        identity = torch.tensor([1., 0., 0., 0., 1., 0.], device=self.stn_fc_bias.device)
        bias_reg = F.mse_loss(self.stn_fc_bias, identity)
        weight_reg = (self.stn_fc_weight ** 2).mean()
        return bias_reg + weight_reg

    def forward(self, image: Tensor, **kwargs: Tensor) -> Tensor:
        """Extract message logits from framed image.

        Args:
            image: (B, 3, H, W) in [0, 1]
            **kwargs:
                mask: (B, 1, H, W) with 1 in center, 0 in border.
                    When provided, border_mask = 1 - mask is concatenated as a
                    4th CNN input channel. When absent, all-ones is used.
                frame_width: Width of the border frame in pixels (default: 32).

        Returns:
            Message logits (B, num_bits) - unbounded, apply sigmoid for probabilities
        """
        mask: Tensor | None = kwargs.get("mask", None)
        frame_width: int = int(kwargs.get("frame_width", 32))

        # Normalize input
        image_norm = image - 0.5

        # Compute STN affine parameters (uses full image for perspective estimation)
        stn_features = self.stn_params(image_norm)
        theta = torch.mm(stn_features, self.stn_fc_weight) + self.stn_fc_bias
        theta = theta.view(-1, 2, 3)

        # Apply spatial transform
        grid = F.affine_grid(theta, list(image_norm.size()), align_corners=False)
        transformed = F.grid_sample(
            image_norm, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )

        # Build border indicator channel (1=border, 0=center)
        if mask is not None:
            transformed_mask = F.grid_sample(
                mask, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
            )
            border_mask = 1.0 - transformed_mask
        else:
            # No mask — all ones (full-image processing)
            border_mask = torch.ones(
                image.shape[0], 1, image.shape[2], image.shape[3],
                device=image.device, dtype=image.dtype,
            )

        # CNN branch: RGB + border indicator → strided CNN → flatten
        cnn_input = torch.cat([transformed, border_mask], dim=1)
        cnn_features = self.decoder_cnn(cnn_input).flatten(1)  # (B, 21632)

        # Border-pooling branch: strip-pool on ORIGINAL image (not STN-transformed)
        # to preserve border spatial information
        border_features = self._pool_border_strips(image, frame_width)  # (B, 4800)
        border_features = self.border_fc(border_features)  # (B, 256)

        # Combine and classify
        combined = torch.cat([cnn_features, border_features], dim=1)  # (B, 21888)
        x = F.relu(self.fc1(combined))
        logits: Tensor = self.fc2(x)
        return logits

    def decode(self, image: Tensor, **kwargs: Tensor) -> Tensor:
        """Extract binary message from an image.

        Args:
            image: Input image tensor (B, C, H, W) in [0, 1].
            **kwargs: Optional ``mask`` and ``frame_width`` passed to forward().

        Returns:
            Binary message tensor (B, num_bits).
        """
        logits = self.forward(image, **kwargs)
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
