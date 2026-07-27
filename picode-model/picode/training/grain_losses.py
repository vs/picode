"""PicoGrain-specific loss functions.

These losses are designed to preserve image structure while permitting
grain texture. Standard L2/LPIPS would fight the grain, so we compute
losses on blurred versions (removing grain, keeping structure).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Apply Gaussian blur to a tensor.

    Args:
        x: (B, C, H, W) tensor.
        sigma: Gaussian standard deviation.

    Returns:
        Blurred tensor, same shape.
    """
    k = 2 * math.ceil(3 * sigma) + 1
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, k, k)
    kernel = kernel.expand(x.shape[1], -1, -1, -1)
    return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])


def blurred_l2_loss(original: Tensor, encoded: Tensor, blur_sigma: float = 2.0) -> Tensor:
    """L2 loss computed on Gaussian-blurred images.

    Removes grain texture before comparing, so the loss preserves
    color and structure without penalizing high-frequency grain.

    Args:
        original: (B, 3, H, W) original images.
        encoded: (B, 3, H, W) encoded images.
        blur_sigma: Gaussian blur sigma.

    Returns:
        Scalar L2 loss.
    """
    orig_blur = _gaussian_blur(original, blur_sigma)
    enc_blur = _gaussian_blur(encoded, blur_sigma)
    return F.mse_loss(enc_blur, orig_blur)


def blurred_lpips_loss(
    original: Tensor,
    encoded: Tensor,
    lpips_fn: torch.nn.Module,
    blur_sigma: float = 2.0,
) -> Tensor:
    """LPIPS loss computed on Gaussian-blurred images.

    Args:
        original: (B, 3, H, W) original images.
        encoded: (B, 3, H, W) encoded images.
        lpips_fn: LPIPS network (expects input in [-1, 1]).
        blur_sigma: Gaussian blur sigma.

    Returns:
        Scalar LPIPS loss.
    """
    orig_blur = _gaussian_blur(original, blur_sigma)
    enc_blur = _gaussian_blur(encoded, blur_sigma)
    return lpips_fn(orig_blur * 2 - 1, enc_blur * 2 - 1).mean()  # type: ignore[no-any-return]


def luminance_fidelity_loss(
    image: Tensor, residual: Tensor, lum_floor: float = 0.1,
) -> Tensor:
    """L2 residual loss weighted by inverse luminance.

    Penalizes grain in dark areas where it shouldn't appear.

    Args:
        image: (B, 3, H, W) original cover image in [0, 1].
        residual: (B, 1, H, W) encoding residual.
        lum_floor: Minimum luminance mask value.

    Returns:
        Scalar loss.
    """
    lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    inv_weight = 1.0 - (lum_floor + (1.0 - lum_floor) * lum)
    return (inv_weight * residual**2).mean()


def envelope_smoothness_loss(envelope: Tensor) -> Tensor:
    """Total variation loss on the envelope to keep it spatially smooth.

    Args:
        envelope: (B, 1, H, W) smooth envelope from encoder.

    Returns:
        Scalar TV loss.
    """
    diff_h = (envelope[:, :, 1:, :] - envelope[:, :, :-1, :]).pow(2).mean()
    diff_w = (envelope[:, :, :, 1:] - envelope[:, :, :, :-1]).pow(2).mean()
    return diff_h + diff_w
