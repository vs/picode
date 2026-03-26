# picode/detection/rectifier.py
"""Perspective rectification for detected watermark regions."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import Tensor

from picode.detection.types import Quadrilateral


class Rectifier:
    """Extracts and rectifies watermarked region from detected corners.

    Uses OpenCV's perspective transform to warp a quadrilateral region
    to a square output suitable for decoder input.

    Args:
        output_size: Size of output square image (default 400 for decoder)

    Example:
        >>> rectifier = Rectifier(output_size=400)
        >>> detection = fast_detector.detect(image)
        >>> if detection:
        ...     rectified = rectifier.rectify(image, detection.corners)
        ...     message = decoder(rectified.unsqueeze(0))
    """

    def __init__(self, output_size: int = 400) -> None:
        self.output_size = output_size
        self.dst_corners = np.array(
            [
                [0, 0],
                [output_size - 1, 0],
                [output_size - 1, output_size - 1],
                [0, output_size - 1],
            ],
            dtype=np.float32,
        )

    def rectify(
        self,
        image: Tensor,
        corners: Quadrilateral | Tensor,
    ) -> Tensor:
        """Extract and rectify region defined by corners.

        Args:
            image: Source image (C, H, W) in [0, 1]
            corners: Quadrilateral or (8,) tensor of pixel coordinates

        Returns:
            Rectified (C, output_size, output_size) tensor in [0, 1]
        """
        # Convert corners to numpy array
        if isinstance(corners, Tensor):
            src_corners = corners.detach().cpu().numpy().reshape(4, 2).astype(np.float32)
        else:
            src_corners = corners.to_numpy()

        # Convert image to numpy HWC format
        img_np = image.permute(1, 2, 0).cpu().numpy()

        # Ensure proper range for OpenCV
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_corners, self.dst_corners)

        # Apply warp
        rectified = cv2.warpPerspective(
            img_np,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Convert back to tensor CHW format in [0, 1]
        rectified_tensor = torch.from_numpy(rectified).float() / 255.0
        rectified_tensor = rectified_tensor.permute(2, 0, 1)

        return rectified_tensor

    def rectify_batch(
        self,
        images: list[Tensor],
        corners: list[Quadrilateral | Tensor],
    ) -> Tensor:
        """Batch rectification.

        Args:
            images: List of (C, H, W) tensors
            corners: List of Quadrilateral or (8,) tensors

        Returns:
            (B, C, output_size, output_size) tensor
        """
        return torch.stack(
            [self.rectify(img, corn) for img, corn in zip(images, corners)]
        )
