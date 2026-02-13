"""Geometric distortions using Kornia.

Uses kornia.geometry.transform for geometric operations. API matches native geometric.py exactly.
"""

from typing import Any

import kornia.geometry.transform
import torch
from torch import Tensor

from picode.distortions.base import Distortion


class Rotation(Distortion):
    """Rotation around image center using Kornia.

    API-compatible with native.Rotation.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        max_angle: Maximum rotation angle in degrees.
    """

    name = "rotation"

    def __init__(self, intensity: float = 0.5, max_angle: float = 30.0):
        super().__init__(intensity)
        self.max_angle = max_angle
        self._current_angle: float | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply rotation using kornia.geometry.transform.rotate.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Rotated tensor clamped to [0, 1].
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Effective angle based on intensity
        effective_max = self.max_angle * self.intensity

        # Random angles for each batch item (in degrees)
        angles = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * effective_max

        self._current_angle = angles[0].item() if B > 0 else 0.0

        # Center of rotation: (B, 2) tensor with (x, y) coordinates
        center = torch.tensor([[W / 2.0, H / 2.0]], device=device, dtype=dtype).expand(B, -1)

        # Apply rotation using Kornia
        output = kornia.geometry.transform.rotate(x, angles, center=center)

        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample rotation parameters."""
        effective_max = self.max_angle * self.intensity
        angle = (torch.rand(1).item() * 2 - 1) * effective_max
        return {"angle": angle}


class Scale(Distortion):
    """Random zoom in/out transformation using Kornia.

    API-compatible with native.Scale.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        min_scale: Minimum scale factor.
        max_scale: Maximum scale factor.
    """

    name = "scale"

    def __init__(
        self, intensity: float = 0.5, min_scale: float = 0.8, max_scale: float = 1.2
    ):
        super().__init__(intensity)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._current_scale: float | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply scale transformation using kornia.geometry.transform.resize.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Scaled tensor clamped to [0, 1].
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Interpolate between 1.0 (identity) and actual scale range based on intensity
        effective_min = 1.0 + (self.min_scale - 1.0) * self.intensity
        effective_max = 1.0 + (self.max_scale - 1.0) * self.intensity

        # Random scale factors (one per batch)
        scales = (
            torch.rand(B, device=device, dtype=dtype) * (effective_max - effective_min)
            + effective_min
        )

        self._current_scale = scales[0].item() if B > 0 else 1.0

        # Process each batch item with its own scale factor
        outputs = []
        for b in range(B):
            scale = scales[b].item()
            img = x[b : b + 1]  # (1, C, H, W)

            # Compute scaled size
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))

            # Resize to scaled size
            scaled = kornia.geometry.transform.resize(
                img, (new_h, new_w), interpolation="bilinear"
            )

            if scale > 1.0:
                # Scale > 1 means zoom in: center crop to original size
                start_h = (new_h - H) // 2
                start_w = (new_w - W) // 2
                result = scaled[:, :, start_h : start_h + H, start_w : start_w + W]
            elif scale < 1.0:
                # Scale < 1 means zoom out: pad to original size
                pad_h = H - new_h
                pad_w = W - new_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                result = torch.nn.functional.pad(
                    scaled, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )
            else:
                result = scaled

            outputs.append(result)

        output = torch.cat(outputs, dim=0)
        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample scale parameters."""
        effective_min = 1.0 + (self.min_scale - 1.0) * self.intensity
        effective_max = 1.0 + (self.max_scale - 1.0) * self.intensity
        scale = torch.rand(1).item() * (effective_max - effective_min) + effective_min
        return {"scale": scale}


class Crop(Distortion):
    """Random crop and resize transformation using Kornia.

    API-compatible with native.Crop.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        min_ratio: Minimum crop ratio (fraction of original size).
    """

    name = "crop"

    def __init__(self, intensity: float = 0.5, min_ratio: float = 0.7):
        super().__init__(intensity)
        self.min_ratio = min_ratio
        self._current_crop: dict[str, float] | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply crop and resize using kornia.geometry.transform.crop_and_resize.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Cropped and resized tensor clamped to [0, 1].
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Effective min ratio based on intensity (1.0 means no crop)
        effective_min = 1.0 - (1.0 - self.min_ratio) * self.intensity

        # Random crop ratios
        crop_ratios = (
            torch.rand(B, device=device, dtype=dtype) * (1.0 - effective_min) + effective_min
        )

        # Random crop positions (where the crop window starts)
        # Max offset is (1 - crop_ratio) as fraction of image size
        max_offsets = 1.0 - crop_ratios
        offset_x = torch.rand(B, device=device, dtype=dtype) * max_offsets
        offset_y = torch.rand(B, device=device, dtype=dtype) * max_offsets

        self._current_crop = {
            "ratio": crop_ratios[0].item() if B > 0 else 1.0,
            "offset_x": offset_x[0].item() if B > 0 else 0.0,
            "offset_y": offset_y[0].item() if B > 0 else 0.0,
        }

        # Build boxes tensor: (B, 4, 2) with corners
        # [top-left, top-right, bottom-right, bottom-left]
        boxes = torch.zeros(B, 4, 2, device=device, dtype=dtype)

        for b in range(B):
            # Compute crop box in pixel coordinates
            x1 = offset_x[b] * W
            y1 = offset_y[b] * H
            crop_w = crop_ratios[b] * W
            crop_h = crop_ratios[b] * H
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            # Corners: [top-left, top-right, bottom-right, bottom-left]
            boxes[b, 0] = torch.tensor([x1, y1], device=device, dtype=dtype)  # top-left
            boxes[b, 1] = torch.tensor([x2, y1], device=device, dtype=dtype)  # top-right
            boxes[b, 2] = torch.tensor([x2, y2], device=device, dtype=dtype)  # bottom-right
            boxes[b, 3] = torch.tensor([x1, y2], device=device, dtype=dtype)  # bottom-left

        # Crop and resize to original dimensions
        output = kornia.geometry.transform.crop_and_resize(
            x, boxes, (H, W), mode="bilinear"
        )

        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample crop parameters."""
        effective_min = 1.0 - (1.0 - self.min_ratio) * self.intensity
        ratio = torch.rand(1).item() * (1.0 - effective_min) + effective_min
        max_offset = 1.0 - ratio
        offset_x = torch.rand(1).item() * max_offset
        offset_y = torch.rand(1).item() * max_offset
        return {"ratio": ratio, "offset_x": offset_x, "offset_y": offset_y}


class PerspectiveWarp(Distortion):
    """Random perspective transformation using Kornia.

    API-compatible with native.PerspectiveWarp.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        scale: Maximum corner displacement as fraction of image size.
    """

    name = "perspective_warp"

    def __init__(self, intensity: float = 0.5, scale: float = 0.1):
        super().__init__(intensity)
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """Apply perspective warp using kornia.geometry.transform functions.

        Uses get_perspective_transform and warp_perspective.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Warped tensor clamped to [0, 1].
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Effective scale based on intensity
        effective_scale = self.scale * self.intensity

        # Source corners in pixel coordinates: (B, 4, 2)
        # Order: top-left, top-right, bottom-right, bottom-left
        src_points = torch.tensor(
            [[0.0, 0.0], [W - 1, 0.0], [W - 1, H - 1], [0.0, H - 1]],
            device=device,
            dtype=dtype,
        ).unsqueeze(0).expand(B, -1, -1).clone()

        # Random displacements for destination corners
        # Scale displacements relative to image dimensions
        displacement_x = (
            (torch.rand(B, 4, device=device, dtype=dtype) * 2 - 1) * effective_scale * W
        )
        displacement_y = (
            (torch.rand(B, 4, device=device, dtype=dtype) * 2 - 1) * effective_scale * H
        )
        displacements = torch.stack([displacement_x, displacement_y], dim=-1)

        # Destination corners
        dst_points = src_points + displacements

        # Compute perspective transform matrix
        M = kornia.geometry.transform.get_perspective_transform(src_points, dst_points)

        # Apply warp
        output = kornia.geometry.transform.warp_perspective(
            x, M, (H, W), mode="bilinear", padding_mode="zeros"
        )

        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample perspective warp parameters."""
        return {"scale": self.scale}
