"""Loss functions for PicodeFrame training.

Frame-specific losses that only penalize the generated border region
while preserving the original center pixels.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from picode.models.picodeframe.decoder import Decoder


def message_loss(decoded_logits: Tensor, message: Tensor) -> Tensor:
    """Binary cross-entropy loss for message recovery.

    Args:
        decoded_logits: Decoder output logits (B, num_bits) - NOT probabilities.
        message: Target message (B, num_bits) binary tensor.

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(decoded_logits, message)


def message_loss_mse(decoded_logits: Tensor, message: Tensor) -> Tensor:
    """MSE loss for message recovery.

    Applies sigmoid to logits then computes MSE against binary targets.
    Unlike BCE, MSE has no stable trivial equilibrium at 0.5, making
    training collapse less likely and recovery possible.

    Args:
        decoded_logits: Decoder output logits (B, num_bits) - NOT probabilities.
        message: Target message (B, num_bits) binary tensor.

    Returns:
        Scalar MSE loss.
    """
    decoded_probs = torch.sigmoid(decoded_logits)
    return F.mse_loss(decoded_probs, message)


def frame_l2_loss(generated: Tensor, ground_truth: Tensor, mask: Tensor) -> Tensor:
    """L2 loss on frame pixels only (where mask == 0).

    Args:
        generated: Generated framed image (B, C, H, W).
        ground_truth: Ground truth image with real borders (B, C, H, W).
        mask: Binary mask (B, 1, H, W), 1 in center, 0 in border.

    Returns:
        Scalar L2 loss on frame region.
    """
    # Invert mask to select frame pixels
    frame_mask = 1 - mask  # (B, 1, H, W)
    # Count frame pixels for proper normalization
    num_frame_pixels = frame_mask.sum().clamp(min=1)
    # Compute squared error only on frame pixels
    sq_error = ((generated - ground_truth) ** 2) * frame_mask
    return sq_error.sum() / num_frame_pixels


def frame_lpips_loss(
    generated: Tensor,
    ground_truth: Tensor,
    mask: Tensor,
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """LPIPS loss on frame region via full-image comparison.

    Computes LPIPS on the full generated vs ground truth images. Since
    the center pixels are identical (enforced by hard mask), the perceptual
    loss naturally comes from the frame region only. This avoids extracting
    thin border strips that are too small for LPIPS network pooling layers.

    Args:
        generated: Generated framed image (B, C, H, W) in [0, 1].
        ground_truth: Ground truth image (B, C, H, W) in [0, 1].
        mask: Binary mask (B, 1, H, W), 1 in center, 0 in border. (unused,
            kept for API consistency)
        lpips_fn: LPIPS loss function (expects inputs in [-1, 1]).

    Returns:
        Scalar LPIPS loss.
    """
    # Scale to [-1, 1] for LPIPS
    gen_scaled = generated * 2 - 1
    gt_scaled = ground_truth * 2 - 1

    # Full-image LPIPS. Center pixels are identical so loss comes from frame.
    return lpips_fn(gen_scaled, gt_scaled).mean()


def frame_color_loss(generated: Tensor, ground_truth: Tensor, mask: Tensor) -> Tensor:
    """Penalize color shifts in the frame residual.

    Computes the variance of the residual across RGB channels at each frame pixel,
    then averages. This pushes the encoder toward grayscale-only residuals (equal
    change across R, G, B) so the frame has no visible color artifacts.

    Args:
        generated: Generated framed image (B, C, H, W).
        ground_truth: Ground truth image (B, C, H, W).
        mask: Binary mask (B, 1, H, W), 1 in center, 0 in border.

    Returns:
        Scalar color variance loss on frame region.
    """
    residual = generated - ground_truth  # (B, 3, H, W)
    frame_mask = 1 - mask  # (B, 1, H, W)
    # Variance across RGB channels at each spatial position
    channel_var = residual.var(dim=1, keepdim=True)  # (B, 1, H, W)
    num_frame_pixels = frame_mask.sum().clamp(min=1)
    return (channel_var * frame_mask).sum() / num_frame_pixels


def stn_scale_loss(decoder: Decoder) -> Tensor:
    """STN scale regularization loss.

    Args:
        decoder: PicodeFrame decoder with STN.

    Returns:
        Scalar regularization loss.
    """
    return decoder.stn_scale_reg()


def compute_picodeframe_loss(
    generated: Tensor,
    ground_truth: Tensor,
    mask: Tensor,
    messages: Tensor,
    decoded_logits: Tensor,
    decoder: Decoder,
    lpips_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    weight_msg: float = 1.0,
    weight_frame_l2: float = 2.0,
    weight_frame_lpips: float = 1.5,
    weight_stn_reg: float = 0.1,
    use_lpips: bool = True,
) -> dict[str, Tensor]:
    """Compute combined PicodeFrame training loss.

    Args:
        generated: Generated framed image (B, C, H, W).
        ground_truth: Ground truth image with real borders (B, C, H, W).
        mask: Binary mask (B, 1, H, W), 1 in center, 0 in border.
        messages: Target messages (B, num_bits).
        decoded_logits: Decoder output logits (B, num_bits).
        decoder: PicodeFrame decoder (for STN regularization).
        lpips_fn: Optional LPIPS loss function.
        weight_msg: Weight for message loss.
        weight_frame_l2: Weight for frame L2 loss.
        weight_frame_lpips: Weight for frame LPIPS loss.
        weight_stn_reg: Weight for STN regularization.
        use_lpips: Whether to include LPIPS loss.

    Returns:
        Dictionary with loss components and total loss.
    """
    loss_msg = message_loss(decoded_logits, messages)
    loss_fl2 = frame_l2_loss(generated, ground_truth, mask)
    loss_stn = stn_scale_loss(decoder)

    total = weight_msg * loss_msg + weight_frame_l2 * loss_fl2 + weight_stn_reg * loss_stn

    result: dict[str, Tensor] = {
        "loss_msg": loss_msg,
        "loss_frame_l2": loss_fl2,
        "loss_stn_reg": loss_stn,
    }

    if use_lpips and lpips_fn is not None:
        loss_flpips = frame_lpips_loss(generated, ground_truth, mask, lpips_fn)
        total = total + weight_frame_lpips * loss_flpips
        result["loss_frame_lpips"] = loss_flpips

    result["loss"] = total
    return result
