# picode/detection/seg_detector.py
"""Segmentation-based detector model for watermark region localization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch import Tensor
from torchvision import transforms

from picode.detection.types import Detection, Point, Quadrilateral


class SegDetectorModel(nn.Module):
    """Segmentation-based watermark detector producing a pixel-wise mask.

    Architecture:
        - MobileNetV3-Small backbone (576 channels, stride 32)
        - Spatial decoder: 576→128 (1×1) → 2× upsample → 128→64 (3×3) →
          2× upsample → 64→32 (3×3) → 2× upsample → 32→1 (1×1) → sigmoid
        - Output: (B, 1, H/4, W/4) probability mask in [0, 1]

    Args:
        input_size: Expected input image size (default 320)
        pretrained: Use ImageNet pretrained backbone

    Example:
        >>> model = SegDetectorModel(input_size=320)
        >>> x = torch.rand(2, 3, 320, 320)
        >>> mask = model(x)
        >>> mask.shape
        torch.Size([2, 1, 80, 80])
    """

    def __init__(self, input_size: int = 320, pretrained: bool = True) -> None:
        super().__init__()
        self.input_size = input_size

        # MobileNetV3-Small backbone — stride 32, 576 output channels
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # (B, 576, H/32, W/32)

        # Spatial decoder: 3× bilinear upsample brings stride 32 → stride 4
        # Layer 1: 576 → 128 channels (1×1 conv)
        self.dec1 = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )
        # Layer 2: 128 → 64 channels (3×3 conv, after 2× upsample)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )
        # Layer 3: 64 → 32 channels (3×3 conv, after 2× upsample)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # Output head: 32 → 1 channel (1×1 conv, after 2× upsample)
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize decoder weights."""
        for module in [self.dec1, self.dec2, self.dec3]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.output_conv.weight, mode="fan_out", nonlinearity="relu")
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W) in [0, 1]

        Returns:
            Segmentation mask (B, 1, H/4, W/4) in [0, 1]
        """
        # Backbone: stride 32
        feat = self.features(x)  # (B, 576, H/32, W/32)

        # Decode with 3× 2× upsamples: stride 32 → 16 → 8 → 4
        x = self.dec1(feat)  # (B, 128, H/32, W/32)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/16
        x = self.dec2(x)  # (B, 64, H/16, W/16)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/8
        x = self.dec3(x)  # (B, 32, H/8, W/8)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/4
        x = self.output_conv(x)  # (B, 1, H/4, W/4)
        return torch.sigmoid(x)


def _order_corners(pts: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Order 4 points as TL, TR, BR, BL.

    Args:
        pts: (4, 2) array of corner points (x, y)

    Returns:
        (4, 2) ordered array: TL, TR, BR, BL
    """
    # Sort by sum (x+y): TL has smallest sum, BR has largest
    s = pts.sum(axis=1)
    tl = pts[s.argmin()]
    br = pts[s.argmax()]
    # Sort by difference (y-x): TR has smallest diff, BL has largest
    d = np.diff(pts, axis=1).ravel()
    tr = pts[d.argmin()]
    bl = pts[d.argmax()]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class SegDetector:
    """High-level API for segmentation-based watermark detection.

    Example:
        >>> detector = SegDetector.from_checkpoint("seg_detector.pt")
        >>> result = detector.detect("photo.jpg")
        >>> if result:
        ...     print(f"Watermark region at {result.corners}")
    """

    def __init__(
        self,
        model: SegDetectorModel,
        threshold: float = 0.5,
        min_area_ratio: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize SegDetector.

        Args:
            model: SegDetectorModel instance
            threshold: Mask binarization threshold
            min_area_ratio: Minimum region area as fraction of mask area
            device: Device to run inference on
        """
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
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
        **kwargs: float,
    ) -> SegDetector:
        """Load from saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run on
            **kwargs: Additional arguments for SegDetector

        Returns:
            Initialized SegDetector
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_config = ckpt.get("model_config", {})
        model = SegDetectorModel(**model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, device=device, **kwargs)

    def _preprocess(self, image: Tensor | str | Path) -> tuple[Tensor, tuple[int, int]]:
        """Preprocess image for model input.

        Args:
            image: Input image (tensor, path, or PIL Image)

        Returns:
            Tuple of (preprocessed tensor (3, H, W), original size (H, W))
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            original_size = (pil_image.height, pil_image.width)
            tensor = self._transform(pil_image)
        elif isinstance(image, Tensor):
            if image.dim() == 3:
                original_size = (image.shape[1], image.shape[2])
                img = image.unsqueeze(0)
            else:
                original_size = (image.shape[2], image.shape[3])
                img = image
            tensor = F.interpolate(
                img,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return tensor, original_size

    def _extract_corners(
        self,
        mask_2d: Tensor,
        original_size: tuple[int, int],
    ) -> Tensor | None:
        """Extract quadrilateral corners from a 2D probability mask.

        Thresholds the mask, finds contours, fits a quadrilateral, and scales
        the corners back to the original image size.

        Args:
            mask_2d: (H, W) float tensor of probabilities in [0, 1]
            original_size: (H, W) of original image for coordinate scaling

        Returns:
            (4, 2) tensor of corner pixel coordinates (TL, TR, BR, BL) in
            original image space, or None if no valid region found.
        """
        # Binarize
        binary = (mask_2d >= self.threshold).cpu().numpy().astype(np.uint8) * 255
        mask_h, mask_w = binary.shape
        min_area = self.min_area_ratio * mask_h * mask_w

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Pick largest contour by area
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            return None

        # Approximate polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Use bounding-rect corners if approxPolyDP doesn't give exactly 4 pts
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            pts = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )

        # Order corners TL, TR, BR, BL
        pts = _order_corners(pts)

        # Scale to original image coordinates
        orig_h, orig_w = original_size
        scale_x = orig_w / mask_w
        scale_y = orig_h / mask_h
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y

        return torch.from_numpy(pts)

    def get_mask(self, image: Tensor | str | Path) -> Tensor:
        """Run model and return raw probability mask.

        Args:
            image: Input image

        Returns:
            (1, 1, H/4, W/4) probability mask in [0, 1]
        """
        img_tensor, _ = self._preprocess(image)
        with torch.no_grad():
            mask: Tensor = self.model(img_tensor.unsqueeze(0).to(self.device))
        return mask.cpu()

    def detect(self, image: Tensor | str | Path) -> Detection | None:
        """Detect watermark region in image.

        Args:
            image: Input image (path, tensor)

        Returns:
            Detection with quadrilateral corners if a region is found, else None.
        """
        img_tensor, original_size = self._preprocess(image)

        with torch.no_grad():
            mask = self.model(img_tensor.unsqueeze(0).to(self.device))

        mask_2d = mask[0, 0].cpu()
        corners = self._extract_corners(mask_2d, original_size)
        if corners is None:
            return None

        # Confidence: mean probability in the thresholded region
        confidence = float(mask_2d[mask_2d >= self.threshold].mean())

        quad = Quadrilateral(
            top_left=Point(float(corners[0, 0]), float(corners[0, 1])),
            top_right=Point(float(corners[1, 0]), float(corners[1, 1])),
            bottom_right=Point(float(corners[2, 0]), float(corners[2, 1])),
            bottom_left=Point(float(corners[3, 0]), float(corners[3, 1])),
        )

        return Detection(
            corners=quad,
            confidence=confidence,
            detector_type="fast",
        )
