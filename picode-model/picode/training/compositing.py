"""Compositing module for placing encoded images into background scenes.

During training, encoded images are composited into random backgrounds at
random scale, position, and perspective. This teaches the decoder to handle
real-world scenarios where the encoded image appears within a larger photo.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import Tensor


def composite_into_background(
    encoded: Tensor,
    background: Tensor,
    scale_min: float = 0.50,
    scale_max: float = 0.95,
    perspective_strength: float = 0.05,
) -> dict[str, Tensor]:
    """Composite encoded images into background images.

    Places each encoded image into the corresponding background at a random
    scale, position, and perspective. Returns the composited scene, a binary
    mask of the encoded region, and the corner coordinates.

    Args:
        encoded: Encoded images (B, 3, H, W) in [0, 1].
        background: Background images (B, 3, H, W) in [0, 1]. Same size as encoded.
        scale_min: Minimum scale of encoded image relative to background.
        scale_max: Maximum scale of encoded image relative to background.
        perspective_strength: Max corner displacement as fraction of region size.

    Returns:
        Dict with:
            - composited: (B, 3, H, W) composited scene
            - mask: (B, 1, H, W) binary mask of encoded region
            - corners: (B, 4, 2) pixel coordinates of encoded region corners
              (top-left, top-right, bottom-right, bottom-left)
    """
    B, C, H, W = encoded.shape
    device = encoded.device

    composited = background.clone()
    masks = torch.zeros(B, 1, H, W, device=device)
    all_corners = torch.zeros(B, 4, 2, device=device)

    for i in range(B):
        scale = torch.empty(1).uniform_(scale_min, scale_max).item()
        region_h = int(H * scale)
        region_w = int(W * scale)

        max_y = H - region_h
        max_x = W - region_w
        offset_y = int(torch.randint(0, max(max_y, 1), (1,)).item())
        offset_x = int(torch.randint(0, max(max_x, 1), (1,)).item())

        dst_corners = np.array([
            [offset_x, offset_y],
            [offset_x + region_w, offset_y],
            [offset_x + region_w, offset_y + region_h],
            [offset_x, offset_y + region_h],
        ], dtype=np.float32)

        if perspective_strength > 0:
            max_disp = perspective_strength * min(region_h, region_w)
            noise = np.random.uniform(-max_disp, max_disp, (4, 2)).astype(np.float32)
            dst_corners = dst_corners + noise
            dst_corners[:, 0] = np.clip(dst_corners[:, 0], 0, W - 1)
            dst_corners[:, 1] = np.clip(dst_corners[:, 1], 0, H - 1)

        src_corners = np.array([
            [0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_corners, dst_corners)

        enc_np = (encoded[i].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        warped = cv2.warpPerspective(enc_np, M, (W, H))
        warped_t = torch.from_numpy(warped).float().permute(2, 0, 1) / 255.0

        mask_np = np.ones((H, W), dtype=np.uint8) * 255
        mask_warped = cv2.warpPerspective(mask_np, M, (W, H))
        mask_t = torch.from_numpy(mask_warped).float() / 255.0

        warped_t = warped_t.to(device)
        mask_t = mask_t.to(device)
        mask_3ch = mask_t.unsqueeze(0).expand(3, -1, -1)
        composited[i] = composited[i] * (1 - mask_3ch) + warped_t * mask_3ch
        masks[i, 0] = mask_t
        all_corners[i] = torch.from_numpy(dst_corners).to(device)

    return {
        "composited": composited.clamp(0, 1),
        "mask": masks,
        "corners": all_corners,
    }


def extract_with_jitter(
    composited: Tensor,
    corners: Tensor,
    output_size: int = 512,
    jitter: float = 0.0,
) -> Tensor:
    """Extract and rectify the encoded region from composited scene.

    Applies perspective correction to extract the region defined by corners,
    with optional jitter to simulate imprecise cropping.

    Args:
        composited: Composited scene (B, 3, H, W) in [0, 1].
        corners: Corner coordinates (B, 4, 2) — TL, TR, BR, BL in pixels.
        output_size: Size of output square image.
        jitter: Max jitter as fraction of region size (0 = no jitter).

    Returns:
        Extracted images (B, 3, output_size, output_size).
    """
    B = composited.shape[0]
    H, W = composited.shape[2], composited.shape[3]
    results = []

    dst = np.array([
        [0, 0], [output_size - 1, 0],
        [output_size - 1, output_size - 1], [0, output_size - 1],
    ], dtype=np.float32)

    for i in range(B):
        src = corners[i].detach().cpu().numpy().astype(np.float32)

        if jitter > 0:
            region_w = np.linalg.norm(src[1] - src[0])
            region_h = np.linalg.norm(src[3] - src[0])
            max_disp = jitter * min(region_w, region_h)
            noise = np.random.uniform(-max_disp, max_disp, (4, 2)).astype(np.float32)
            src = src + noise
            src[:, 0] = np.clip(src[:, 0], 0, W - 1)
            src[:, 1] = np.clip(src[:, 1], 0, H - 1)

        M = cv2.getPerspectiveTransform(src, dst)
        img_np = (composited[i].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        warped = cv2.warpPerspective(img_np, M, (output_size, output_size))
        results.append(torch.from_numpy(warped).float().permute(2, 0, 1) / 255.0)

    return torch.stack(results).to(composited.device)
