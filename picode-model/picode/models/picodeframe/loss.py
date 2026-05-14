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
    """LPIPS loss focused on frame region.

    Extracts the bounding box around the frame area and computes LPIPS
    on the border strips. Since the frame is a thin border, we compute
    LPIPS on each side strip separately and average.

    Args:
        generated: Generated framed image (B, C, H, W) in [0, 1].
        ground_truth: Ground truth image (B, C, H, W) in [0, 1].
        mask: Binary mask (B, 1, H, W), 1 in center, 0 in border.
        lpips_fn: LPIPS loss function (expects inputs in [-1, 1]).

    Returns:
        Scalar LPIPS loss on frame region.
    """
    _, _, H, W = generated.shape

    # Find frame width from the mask (distance from edge to first 1)
    # Use the first element's mask to determine frame width
    mask_2d = mask[0, 0]  # (H, W)
    # Find first row that has a 1 in the center
    row_sums = mask_2d.sum(dim=1)
    frame_rows = (row_sums == 0).sum().item()
    fw = max(int(frame_rows), 1)

    # Scale to [-1, 1] for LPIPS
    gen_scaled = generated * 2 - 1
    gt_scaled = ground_truth * 2 - 1

    # Compute LPIPS on each border strip
    losses = []

    # Top strip
    if fw > 0:
        top_gen = gen_scaled[:, :, :fw, :]
        top_gt = gt_scaled[:, :, :fw, :]
        # LPIPS needs reasonable spatial size, pad if too thin
        if fw < 16:
            top_gen = F.interpolate(top_gen, size=(16, W), mode="bilinear", align_corners=False)
            top_gt = F.interpolate(top_gt, size=(16, W), mode="bilinear", align_corners=False)
        losses.append(lpips_fn(top_gen, top_gt).mean())

    # Bottom strip
    if fw > 0:
        bot_gen = gen_scaled[:, :, H - fw:, :]
        bot_gt = gt_scaled[:, :, H - fw:, :]
        if fw < 16:
            bot_gen = F.interpolate(bot_gen, size=(16, W), mode="bilinear", align_corners=False)
            bot_gt = F.interpolate(bot_gt, size=(16, W), mode="bilinear", align_corners=False)
        losses.append(lpips_fn(bot_gen, bot_gt).mean())

    # Left strip (excluding corners already counted)
    if fw > 0:
        left_gen = gen_scaled[:, :, fw:H - fw, :fw]
        left_gt = gt_scaled[:, :, fw:H - fw, :fw]
        if fw < 16:
            left_gen = F.interpolate(
                left_gen, size=(H - 2 * fw, 16), mode="bilinear", align_corners=False
            )
            left_gt = F.interpolate(
                left_gt, size=(H - 2 * fw, 16), mode="bilinear", align_corners=False
            )
        losses.append(lpips_fn(left_gen, left_gt).mean())

    # Right strip
    if fw > 0:
        right_gen = gen_scaled[:, :, fw:H - fw, W - fw:]
        right_gt = gt_scaled[:, :, fw:H - fw, W - fw:]
        if fw < 16:
            right_gen = F.interpolate(
                right_gen, size=(H - 2 * fw, 16), mode="bilinear", align_corners=False
            )
            right_gt = F.interpolate(
                right_gt, size=(H - 2 * fw, 16), mode="bilinear", align_corners=False
            )
        losses.append(lpips_fn(right_gen, right_gt).mean())

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=generated.device)


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
