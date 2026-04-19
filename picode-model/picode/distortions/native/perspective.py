"""Perspective transform utilities for pre-encode warp / post-encode unwarp.

Implements the StegaStamp training strategy where:
1. Input image is warped using forward transform M_forward
2. Encoding happens in warped (canonical) space
3. Residual is unwarped using inverse transform M_inverse
4. Border handling is applied based on mode

This creates a stable canonical encoding space, allowing the encoder to learn
patterns that are inherently robust to geometric transformations.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor


def get_rand_transform_matrix(
    batch_size: int,
    image_size: int,
    max_translation: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Generate random perspective transform matrices and their inverses.

    Uses Direct Linear Transform (DLT) algorithm to compute homography
    from 4 corner point correspondences.

    Args:
        batch_size: Number of images in batch.
        image_size: Size of square images (height = width).
        max_translation: Maximum corner displacement as fraction of image size.
            For example, 0.1 means corners can move up to 10% of image size.
        device: Target device.

    Returns:
        Tuple of (M_forward, M_inverse):
        - M_forward: (B, 3, 3) homography matrices for warping input
        - M_inverse: (B, 3, 3) inverse homography matrices for unwarping
    """
    # Source corners in pixel coordinates
    # Order: top-left, top-right, bottom-right, bottom-left
    src_corners = torch.tensor(
        [
            [0.0, 0.0],
            [image_size - 1, 0.0],
            [image_size - 1, image_size - 1],
            [0.0, image_size - 1],
        ],
        dtype=torch.float32,
        device=device,
    )  # (4, 2)

    # Random displacements for each corner in each batch
    # Displacement in pixel units
    max_disp = max_translation * image_size
    displacements = (
        torch.rand(batch_size, 4, 2, device=device) * 2 - 1
    ) * max_disp  # (B, 4, 2)

    # Destination corners (displaced source corners)
    dst_corners = src_corners.unsqueeze(0) + displacements  # (B, 4, 2)

    # Compute homography for each batch item
    M_forward_list = []
    M_inverse_list = []

    for b in range(batch_size):
        # Forward: maps original coords to warped coords (src -> dst)
        H_forward = _compute_homography_dlt(src_corners, dst_corners[b])
        # Inverse: maps warped coords back to original (dst -> src)
        H_inverse = _compute_homography_dlt(dst_corners[b], src_corners)

        M_forward_list.append(H_forward)
        M_inverse_list.append(H_inverse)

    M_forward = torch.stack(M_forward_list, dim=0)  # (B, 3, 3)
    M_inverse = torch.stack(M_inverse_list, dim=0)  # (B, 3, 3)

    return M_forward, M_inverse


def _compute_homography_dlt(src: Tensor, dst: Tensor) -> Tensor:
    """Compute 3x3 homography matrix using Direct Linear Transform (DLT).

    Computes H such that dst = H @ src (in homogeneous coordinates).
    For grid_sample, we need the inverse mapping (where to sample from),
    so we compute H that maps destination coords to source coords.

    Args:
        src: Source points (4, 2) in pixel coordinates
        dst: Destination points (4, 2) in pixel coordinates

    Returns:
        Homography matrix (3, 3) that maps src -> dst
    """
    # Build the DLT matrix A for solving Ah = 0
    # For each point correspondence (x, y) -> (x', y'):
    # [-x, -y, -1,  0,  0,  0, x*x', y*x', x']
    # [ 0,  0,  0, -x, -y, -1, x*y', y*y', y']
    rows: list[Tensor] = []
    for i in range(4):
        x, y = src[i, 0], src[i, 1]
        xp, yp = dst[i, 0], dst[i, 1]

        row1 = torch.tensor(
            [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp],
            dtype=src.dtype,
            device=src.device,
        )
        row2 = torch.tensor(
            [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp],
            dtype=src.dtype,
            device=src.device,
        )
        rows.append(row1)
        rows.append(row2)

    A = torch.stack(rows, dim=0)  # (8, 9)

    # Solve using SVD: h is the last row of V (or column of Vh)
    _, _, Vh = torch.linalg.svd(A)
    h = Vh[-1]  # (9,)
    H = h.reshape(3, 3)

    # Normalize so H[2, 2] = 1
    H = H / (H[2, 2] + 1e-8)

    return cast(Tensor, H)


def perspective_transform(image: Tensor, M: Tensor, padding_mode: str = "zeros") -> Tensor:
    """Apply perspective transform to image using grid_sample.

    Args:
        image: Input tensor (B, C, H, W) in [0, 1].
        M: Homography matrices (B, 3, 3) - maps output coords to input coords.
            For warping with M_forward, pass the inverse (M_inverse) here
            because grid_sample needs to know where to sample FROM.
        padding_mode: Padding mode for out-of-bound pixels.
            "zeros", "border", or "reflection".
            Note: "border" mode is not supported on MPS; will fall back to "reflection".

    Returns:
        Transformed image (B, C, H, W).
    """
    # Handle MPS limitation: border padding mode not supported
    if image.device.type == "mps" and padding_mode == "border":
        padding_mode = "reflection"
    B, C, H, W = image.shape

    # Create normalized coordinate grid for output image
    # Grid in [-1, 1] range as expected by grid_sample
    y_coords = torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype)
    x_coords = torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Convert normalized coords to pixel coords for homography transform
    # [-1, 1] -> [0, W-1] and [0, H-1]
    grid_x_px = (grid_x + 1) * (W - 1) / 2
    grid_y_px = (grid_y + 1) * (H - 1) / 2

    # Homogeneous coordinates (H, W, 3)
    ones = torch.ones_like(grid_x_px)
    coords_hom = torch.stack([grid_x_px, grid_y_px, ones], dim=-1)  # (H, W, 3)
    coords_flat = coords_hom.reshape(-1, 3)  # (H*W, 3)

    # Apply homography for each batch item
    grids = []
    for b in range(B):
        # M[b] maps output pixel coords to input pixel coords
        transformed = coords_flat @ M[b].T  # (H*W, 3)

        # Convert from homogeneous coordinates
        transformed = transformed.reshape(H, W, 3)
        xy = transformed[..., :2] / transformed[..., 2:3].clamp(min=1e-8)

        # Convert pixel coords back to normalized coords
        # [0, W-1] -> [-1, 1] and [0, H-1] -> [-1, 1]
        xy_norm = xy.clone()
        xy_norm[..., 0] = xy[..., 0] * 2 / (W - 1) - 1
        xy_norm[..., 1] = xy[..., 1] * 2 / (H - 1) - 1

        grids.append(xy_norm)

    grid = torch.stack(grids, dim=0)  # (B, H, W, 2)

    # Apply grid_sample
    output = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    return output


def get_identity_transform(batch_size: int, device: torch.device) -> Tensor:
    """Get identity homography matrices.

    Args:
        batch_size: Number of matrices to generate.
        device: Target device.

    Returns:
        Identity matrices (B, 3, 3).
    """
    eye = torch.eye(3, device=device, dtype=torch.float32)
    return eye.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
