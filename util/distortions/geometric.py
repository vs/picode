"""Geometric distortions using PyTorch native operations.

Implements perspective warp, rotation, scale, and crop transformations
using torch.nn.functional.affine_grid and grid_sample for differentiability.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from distortions.base import Distortion


class PerspectiveWarp(Distortion):
    """Random perspective transformation.

    Applies a perspective warp by displacing image corners and computing
    a homography transformation, applied via grid_sample.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        scale: Maximum corner displacement as fraction of image size.
    """

    name = "perspective_warp"

    def __init__(self, intensity: float = 0.5, scale: float = 0.1):
        super().__init__(intensity)
        self.scale = scale

    def _compute_perspective_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Compute perspective transformation grid from corner displacements.

        Uses a homography matrix computed from 4 corner correspondences.
        """
        # Effective scale based on intensity
        effective_scale = self.scale * self.intensity

        # Source corners in normalized coordinates [-1, 1]
        # Order: top-left, top-right, bottom-right, bottom-left
        src_corners = torch.tensor(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            device=device,
            dtype=dtype,
        )

        # Random displacements for destination corners
        displacements = (
            torch.rand(batch_size, 4, 2, device=device, dtype=dtype) * 2 - 1
        ) * effective_scale

        # Destination corners
        dst_corners = src_corners.unsqueeze(0) + displacements

        # Compute homography for each batch item and create grids
        grids = []
        for b in range(batch_size):
            H = self._compute_homography(src_corners, dst_corners[b])
            grid = self._apply_homography(H, height, width, device, dtype)
            grids.append(grid)

        return torch.stack(grids, dim=0)

    def _compute_homography(self, src: Tensor, dst: Tensor) -> Tensor:
        """Compute 3x3 homography matrix from 4 point correspondences.

        Uses Direct Linear Transform (DLT) algorithm.

        Args:
            src: Source points (4, 2)
            dst: Destination points (4, 2)

        Returns:
            Homography matrix (3, 3) that maps dst -> src
            (we want to sample from source given destination coordinates)
        """
        # We need H such that src = H @ dst (to sample src given dst coords)
        # So we swap src and dst in the DLT computation
        A = []
        for i in range(4):
            x, y = dst[i, 0].item(), dst[i, 1].item()
            u, v = src[i, 0].item(), src[i, 1].item()

            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

        A = torch.tensor(A, device=src.device, dtype=src.dtype)

        # Solve using SVD
        _, _, Vh = torch.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)

        # Normalize so H[2,2] = 1
        H = H / H[2, 2]

        return H

    def _apply_homography(
        self,
        H: Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Apply homography to create sampling grid.

        Args:
            H: Homography matrix (3, 3)
            height: Output height
            width: Output width

        Returns:
            Sampling grid (H, W, 2) in normalized coordinates
        """
        # Create grid of destination coordinates
        y_coords = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Homogeneous coordinates
        ones = torch.ones_like(grid_x)
        coords = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)

        # Apply homography
        coords_flat = coords.reshape(-1, 3)  # (H*W, 3)
        transformed = coords_flat @ H.T  # (H*W, 3)

        # Convert from homogeneous coordinates
        transformed = transformed.reshape(height, width, 3)
        grid = transformed[..., :2] / transformed[..., 2:3].clamp(min=1e-8)

        return grid

    def forward(self, x: Tensor) -> Tensor:
        """Apply perspective warp.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Warped tensor.
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape

        grid = self._compute_perspective_grid(B, H, W, x.device, x.dtype)

        output = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict:
        """Sample perspective warp parameters."""
        return {"scale": self.scale}


class Rotation(Distortion):
    """Rotation around image center.

    Applies rotation transformation using affine_grid and grid_sample.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        max_angle: Maximum rotation angle in degrees.
    """

    name = "rotation"

    def __init__(self, intensity: float = 0.5, max_angle: float = 30.0):
        super().__init__(intensity)
        self.max_angle = max_angle
        self._current_angle: float | None = None

    def _create_rotation_matrix(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Create 2x3 affine rotation matrix.

        Returns:
            Affine transformation matrix (B, 2, 3)
        """
        # Effective angle based on intensity
        effective_max = self.max_angle * self.intensity

        # Random angles for each batch item
        angles_deg = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * effective_max
        angles_rad = angles_deg * math.pi / 180.0

        self._current_angle = angles_deg[0].item() if batch_size > 0 else 0.0

        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)

        # Rotation matrix (no translation)
        # [cos, -sin, 0]
        # [sin,  cos, 0]
        theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        return theta

    def forward(self, x: Tensor) -> Tensor:
        """Apply rotation.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Rotated tensor.
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape

        theta = self._create_rotation_matrix(B, x.device, x.dtype)
        grid = F.affine_grid(theta, [B, C, H, W], align_corners=True)
        output = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict:
        """Sample rotation parameters."""
        effective_max = self.max_angle * self.intensity
        angle = (torch.rand(1).item() * 2 - 1) * effective_max
        return {"angle": angle}


class Scale(Distortion):
    """Random zoom in/out transformation.

    Applies scale transformation using affine_grid and grid_sample.

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

    def _create_scale_matrix(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Create 2x3 affine scale matrix.

        Returns:
            Affine transformation matrix (B, 2, 3)
        """
        # Interpolate between 1.0 (identity) and actual scale range based on intensity
        effective_min = 1.0 + (self.min_scale - 1.0) * self.intensity
        effective_max = 1.0 + (self.max_scale - 1.0) * self.intensity

        # Random scale factors
        scales = torch.rand(batch_size, device=device, dtype=dtype) * (
            effective_max - effective_min
        ) + effective_min

        self._current_scale = scales[0].item() if batch_size > 0 else 1.0

        # Scale matrix (divide by scale to zoom in, multiply to zoom out)
        # For grid_sample, we need the inverse: to zoom in (scale > 1),
        # we sample from a smaller region, so we divide coordinates
        inv_scales = 1.0 / scales

        theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = inv_scales
        theta[:, 1, 1] = inv_scales

        return theta

    def forward(self, x: Tensor) -> Tensor:
        """Apply scale transformation.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Scaled tensor.
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape

        theta = self._create_scale_matrix(B, x.device, x.dtype)
        grid = F.affine_grid(theta, [B, C, H, W], align_corners=True)
        output = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict:
        """Sample scale parameters."""
        effective_min = 1.0 + (self.min_scale - 1.0) * self.intensity
        effective_max = 1.0 + (self.max_scale - 1.0) * self.intensity
        scale = torch.rand(1).item() * (effective_max - effective_min) + effective_min
        return {"scale": scale}


class Crop(Distortion):
    """Random crop and resize transformation.

    Applies a random crop and resizes back to original dimensions
    using grid_sample for differentiability.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        min_ratio: Minimum crop ratio (fraction of original size).
    """

    name = "crop"

    def __init__(self, intensity: float = 0.5, min_ratio: float = 0.7):
        super().__init__(intensity)
        self.min_ratio = min_ratio
        self._current_crop: dict | None = None

    def _create_crop_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Create sampling grid for random crop.

        Returns:
            Sampling grid (B, H, W, 2)
        """
        # Effective min ratio based on intensity (1.0 means no crop)
        effective_min = 1.0 - (1.0 - self.min_ratio) * self.intensity

        # Random crop ratios
        crop_ratios = torch.rand(batch_size, device=device, dtype=dtype) * (
            1.0 - effective_min
        ) + effective_min

        # Random crop positions (where the crop window starts)
        # Max offset is (1 - crop_ratio) in normalized coords
        max_offsets = 1.0 - crop_ratios
        offset_x = torch.rand(batch_size, device=device, dtype=dtype) * max_offsets
        offset_y = torch.rand(batch_size, device=device, dtype=dtype) * max_offsets

        self._current_crop = {
            "ratio": crop_ratios[0].item() if batch_size > 0 else 1.0,
            "offset_x": offset_x[0].item() if batch_size > 0 else 0.0,
            "offset_y": offset_y[0].item() if batch_size > 0 else 0.0,
        }

        # Create grids for each batch item
        grids = []
        for b in range(batch_size):
            # Grid coordinates for the crop region in [-1, 1] space
            # Crop starts at (offset_x, offset_y) and spans crop_ratio
            x_start = -1.0 + 2.0 * offset_x[b]
            x_end = x_start + 2.0 * crop_ratios[b]
            y_start = -1.0 + 2.0 * offset_y[b]
            y_end = y_start + 2.0 * crop_ratios[b]

            x_coords = torch.linspace(
                x_start.item(), x_end.item(), width, device=device, dtype=dtype
            )
            y_coords = torch.linspace(
                y_start.item(), y_end.item(), height, device=device, dtype=dtype
            )

            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grids.append(grid)

        return torch.stack(grids, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply crop and resize.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Cropped and resized tensor.
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape

        grid = self._create_crop_grid(B, H, W, x.device, x.dtype)
        output = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return output.clamp(0.0, 1.0)

    def sample_parameters(self) -> dict:
        """Sample crop parameters."""
        effective_min = 1.0 - (1.0 - self.min_ratio) * self.intensity
        ratio = torch.rand(1).item() * (1.0 - effective_min) + effective_min
        max_offset = 1.0 - ratio
        offset_x = torch.rand(1).item() * max_offset
        offset_y = torch.rand(1).item() * max_offset
        return {"ratio": ratio, "offset_x": offset_x, "offset_y": offset_y}
