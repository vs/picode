"""Picodeine loss functions.

Standard steganography losses: message BCE, image L2, LPIPS, STN regularization.
No frame-specific losses (unlike PicodeFrame) — Picodeine encodes across the full image.
"""

from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor

from picode.models.picodeine.decoder import Decoder


def message_loss(decoded_logits: Tensor, message: Tensor) -> Tensor:
    """Binary cross-entropy loss between decoded logits and target message.

    Args:
        decoded_logits: (B, num_bits) raw logits from decoder.
        message: (B, num_bits) binary target {0, 1}.

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(decoded_logits, message)


def image_loss(encoded: Tensor, original: Tensor) -> Tensor:
    """L2 loss between encoded and original images.

    Args:
        encoded: (B, 3, H, W) encoded image.
        original: (B, 3, H, W) original image.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(encoded, original)


def stn_scale_loss(decoder: Decoder) -> Tensor:
    """STN regularization loss toward identity transform.

    Args:
        decoder: Decoder with STN parameters.

    Returns:
        Scalar regularization loss.
    """
    return decoder.stn_scale_reg()


def compute_picodeine_loss(
    original: Tensor,
    encoded: Tensor,
    messages: Tensor,
    decoded_logits: Tensor,
    decoder: Decoder,
    lpips_fn: Callable[[Tensor, Tensor], Tensor] | None,
    use_lpips: bool = True,
    weight_msg: float = 7.0,
    weight_l2: float = 1.0,
    weight_lpips: float = 1.5,
    weight_stn_reg: float = 0.1,
) -> dict[str, Tensor]:
    """Compute all Picodeine losses.

    Args:
        original: (B, 3, H, W) original images.
        encoded: (B, 3, H, W) encoded images.
        messages: (B, num_bits) target messages.
        decoded_logits: (B, num_bits) decoder output logits.
        decoder: Decoder instance (for STN reg).
        lpips_fn: LPIPS function, or None.
        use_lpips: Whether to include LPIPS loss.
        weight_msg: Message loss weight.
        weight_l2: L2 loss weight.
        weight_lpips: LPIPS loss weight.
        weight_stn_reg: STN regularization weight.

    Returns:
        Dict with loss components and total 'loss'.
    """
    loss_msg = message_loss(decoded_logits, messages)
    loss_l2 = image_loss(encoded, original)
    loss_stn = stn_scale_loss(decoder)

    total = weight_msg * loss_msg + weight_l2 * loss_l2 + weight_stn_reg * loss_stn

    result: dict[str, Tensor] = {
        "loss_msg": loss_msg,
        "loss_l2": loss_l2,
        "loss_stn_reg": loss_stn,
    }

    if use_lpips and lpips_fn is not None:
        orig_scaled = original * 2 - 1
        enc_scaled = encoded * 2 - 1
        loss_lpips = lpips_fn(enc_scaled, orig_scaled).mean()
        total = total + weight_lpips * loss_lpips
        result["loss_lpips"] = loss_lpips

    result["loss"] = total
    return result
