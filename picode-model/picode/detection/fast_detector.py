# picode/detection/fast_detector.py
"""FastDetector model for mobile-optimized watermark detection."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch import Tensor
from torchvision import transforms

from picode.detection.types import Detection, Point, Quadrilateral


class FastDetectorModel(nn.Module):
    """Single-pass watermark detector with quadrilateral output.

    Architecture:
        - MobileNetV3-Small backbone (mobile-safe ops: BatchNorm, ReLU6)
        - Global average pooling
        - Three heads: classification, corner regression, confidence

    Args:
        input_size: Expected input image size (default 320)
        pretrained: Use ImageNet pretrained backbone

    Example:
        >>> model = FastDetectorModel(input_size=320)
        >>> x = torch.rand(1, 3, 320, 320)
        >>> output = model(x)
        >>> output["is_watermark"].shape
        torch.Size([1, 1])
        >>> output["corners"].shape
        torch.Size([1, 8])
    """

    def __init__(self, input_size: int = 320, pretrained: bool = True) -> None:
        super().__init__()
        self.input_size = input_size

        # MobileNetV3-Small backbone
        # Uses BatchNorm (fuses with conv) and ReLU6/HardSwish (hardware accelerated)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # Output: 576 channels

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head: is there a watermark?
        self.cls_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Corner regression head: 4 corners x 2 coords = 8 values
        self.corner_head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 8),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Corner confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Initialize heads
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize head weights."""
        for module in [self.cls_head, self.corner_head, self.conf_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W) normalized to [0, 1]

        Returns:
            Dictionary with:
                - is_watermark: (B, 1) classification logits
                - corners: (B, 8) normalized corner coordinates [0, 1]
                - corner_confidence: (B, 1) confidence score [0, 1]
        """
        features = self.features(x)  # (B, 576, H/32, W/32)
        pooled = self.pool(features).flatten(1)  # (B, 576)

        return {
            "is_watermark": self.cls_head(pooled),
            "corners": self.corner_head(pooled),
            "corner_confidence": self.conf_head(pooled),
        }


class FastDetector:
    """High-level API for fast watermark detection.

    Example:
        >>> detector = FastDetector.from_checkpoint("fast_detector.pt")
        >>> result = detector.detect("photo.jpg")
        >>> if result:
        ...     print(f"Watermark at {result.corners}")
    """

    def __init__(
        self,
        model: FastDetectorModel,
        threshold: float = 0.5,
        corner_confidence_threshold: float = 0.3,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize FastDetector.

        Args:
            model: FastDetectorModel instance
            threshold: Classification threshold for detection
            corner_confidence_threshold: Minimum corner confidence to accept
            device: Device to run inference on
        """
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.threshold = threshold
        self.corner_conf_threshold = corner_confidence_threshold
        self.device = device if isinstance(device, str) else str(device)
        self.input_size = model.input_size

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
            ]
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        **kwargs,
    ) -> "FastDetector":
        """Load from saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run on
            **kwargs: Additional arguments for FastDetector

        Returns:
            Initialized FastDetector
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_config = ckpt.get("model_config", {})
        model = FastDetectorModel(**model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, device=device, **kwargs)

    def _preprocess(self, image: Tensor | str | Path) -> tuple[Tensor, tuple[int, int]]:
        """Preprocess image for model input.

        Args:
            image: Input image (tensor, path, or PIL Image)

        Returns:
            Tuple of (preprocessed tensor, original size (H, W))
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            original_size = (pil_image.height, pil_image.width)
            tensor = self._transform(pil_image)
        elif isinstance(image, Tensor):
            if image.dim() == 3:
                original_size = (image.shape[1], image.shape[2])
            else:
                original_size = (image.shape[2], image.shape[3])
            # Resize to input size
            tensor = F.interpolate(
                image.unsqueeze(0) if image.dim() == 3 else image,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return tensor, original_size

    def _denormalize_corners(
        self, corners_norm: Tensor, original_size: tuple[int, int]
    ) -> Tensor:
        """Convert normalized corners to pixel coordinates.

        Args:
            corners_norm: (8,) tensor of normalized [0, 1] coordinates
            original_size: (H, W) of original image

        Returns:
            (8,) tensor of pixel coordinates
        """
        h, w = original_size
        corners_px = corners_norm.clone()
        corners_px[0::2] *= w  # x coordinates
        corners_px[1::2] *= h  # y coordinates
        return corners_px

    def detect(self, image: Tensor | str | Path) -> Detection | None:
        """Detect watermark in image.

        Args:
            image: Input image (path, tensor, or numpy array)

        Returns:
            Detection if watermark found, None otherwise
        """
        # Preprocess
        img_tensor, original_size = self._preprocess(image)

        # Run inference
        with torch.no_grad():
            output = self.model(img_tensor.unsqueeze(0).to(self.device))

        # Check classification threshold
        is_watermark = torch.sigmoid(output["is_watermark"]).item()
        if is_watermark < self.threshold:
            return None

        # Check corner confidence threshold
        corner_conf = output["corner_confidence"].item()
        if corner_conf < self.corner_conf_threshold:
            return None

        # Denormalize corners to pixel coordinates
        corners_norm = output["corners"][0]
        corners_px = self._denormalize_corners(corners_norm, original_size)

        return Detection(
            corners=Quadrilateral.from_tensor(corners_px),
            confidence=is_watermark,
            detector_type="fast",
        )

    def detect_batch(self, images: list[Tensor | str | Path]) -> list[Detection | None]:
        """Batch detection for efficiency.

        Args:
            images: List of input images

        Returns:
            List of Detection or None for each image
        """
        # Preprocess all images
        tensors = []
        sizes = []
        for img in images:
            t, size = self._preprocess(img)
            tensors.append(t)
            sizes.append(size)

        batch = torch.stack(tensors).to(self.device)

        # Single forward pass
        with torch.no_grad():
            outputs = self.model(batch)

        # Process results
        results = []
        for i in range(len(images)):
            is_wm = torch.sigmoid(outputs["is_watermark"][i]).item()
            corner_conf = outputs["corner_confidence"][i].item()

            if is_wm < self.threshold or corner_conf < self.corner_conf_threshold:
                results.append(None)
            else:
                corners_px = self._denormalize_corners(outputs["corners"][i], sizes[i])
                results.append(
                    Detection(
                        corners=Quadrilateral.from_tensor(corners_px),
                        confidence=is_wm,
                        detector_type="fast",
                    )
                )

        return results
