"""Texture-based residual masking for encode-time quality improvement.

Computes a spatial mask based on local image texture (variance), used to
attenuate the steganographic residual in smooth regions where it's
perceptually visible.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_texture_mask(image: Tensor, floor: float = 0.3) -> Tensor:
    """Compute a texture-based spatial mask for residual attenuation.

    Args:
        image: (B, 3, H, W) input image in [0, 1].
        floor: Minimum mask value. Smooth regions get this value,
               textured regions get values up to 1.0. Default: 0.3.

    Returns:
        (B, 1, H, W) mask in [floor, 1.0].
    """
    B, _, H, W = image.shape

    # 1. Convert to grayscale (luminance)
    # ITU-R BT.601 weights
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    # gray: (B, 1, H, W)

    # 2. Build Gaussian kernel for local variance
    kernel_size = 15
    sigma = 3.0
    # Clamp kernel for small images
    max_k = min(H, W)
    if max_k % 2 == 0:
        max_k -= 1
    if kernel_size > max_k:
        kernel_size = max(max_k, 1)
        sigma = kernel_size / 5.0  # Scale sigma proportionally

    pad = kernel_size // 2
    ax = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - pad
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)

    # 3. Compute local variance: Var(X) = E[X^2] - E[X]^2
    # Use replicate padding to avoid boundary artifacts (zero-padding creates
    # false variance at edges of constant images)
    gray_padded = F.pad(gray, [pad, pad, pad, pad], mode="replicate")
    gray_sq_padded = F.pad(gray**2, [pad, pad, pad, pad], mode="replicate")
    mean = F.conv2d(gray_padded, kernel)
    mean_sq = F.conv2d(gray_sq_padded, kernel)
    variance = (mean_sq - mean**2).clamp(min=0)  # Clamp for numerical stability

    # 4. Normalize variance to [0, 1] using a sigmoid with adaptive midpoint
    # Use the per-image 10th percentile as the transition point: regions below
    # this are the smoothest ~10% and get attenuated; regions above are kept.
    # This ensures uniformly noisy images get high mask values everywhere
    # (since even the 10th-percentile variance is well above zero).
    var_flat = variance.view(B, -1)
    var_max = var_flat.max(dim=1).values.view(B, 1, 1, 1)

    # Edge case: constant/near-constant image (no meaningful variance)
    # Threshold accounts for floating-point error in E[X^2] - E[X]^2
    eps = 1e-6
    if (var_max < eps).all():
        return torch.full((B, 1, H, W), floor, device=image.device, dtype=image.dtype)

    # Use a steep sigmoid: gain = 1 / (small_fraction_of_max_variance)
    # Midpoint at 10th percentile so most textured pixels map to ~1.0
    p10 = var_flat.quantile(0.10, dim=1).view(B, 1, 1, 1)
    # Scale factor: ensures transition is sharp around the midpoint
    # Use a fraction of the max variance to set the sigmoid width
    scale = var_max * 0.05  # 5% of max variance as sigmoid width
    scale = scale.clamp(min=1e-10)

    # For constant images in a mixed batch, handle separately
    constant_mask = (var_max < eps).expand_as(variance)
    raw_mask = torch.sigmoid((variance - p10) / scale)
    raw_mask = torch.where(constant_mask, torch.zeros_like(raw_mask), raw_mask)

    # 5. Apply floor
    mask = floor + (1.0 - floor) * raw_mask

    return mask
