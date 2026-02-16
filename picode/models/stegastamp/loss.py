"""Loss functions for StegaStamp training."""

from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor


def message_loss(decoded_logits: Tensor, message: Tensor) -> Tensor:
    """Binary cross-entropy loss for message recovery.

    Args:
        decoded_logits: Decoder output logits (B, num_bits) - NOT probabilities
        message: Target message (B, num_bits) binary tensor

    Returns:
        Scalar BCE loss
    """
    return F.binary_cross_entropy_with_logits(decoded_logits, message)


def image_loss(encoded: Tensor, original: Tensor) -> Tensor:
    """L2 (MSE) loss between encoded and original image."""
    return F.mse_loss(encoded, original)


def compute_loss(
    original: Tensor,
    encoded: Tensor,
    message: Tensor,
    decoded_logits: Tensor,
    lpips_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    use_lpips: bool = True,
    weight_msg: float = 1.0,
    weight_l2: float = 2.0,
    weight_lpips: float = 1.0,
) -> dict[str, Tensor]:
    """Compute combined training loss.

    Args:
        original: Original image (B, C, H, W) in [0, 1]
        encoded: Encoded image (B, C, H, W) in [0, 1]
        message: Target message (B, num_bits) binary tensor
        decoded_logits: Decoder output logits (B, num_bits) - NOT probabilities
        lpips_fn: Optional LPIPS loss function
        use_lpips: Whether to include LPIPS loss
        weight_msg: Weight for message loss
        weight_l2: Weight for L2 image loss
        weight_lpips: Weight for LPIPS loss

    Returns:
        Dictionary with loss components and total loss
    """
    loss_msg = message_loss(decoded_logits, message)
    loss_l2 = image_loss(encoded, original)

    total = weight_msg * loss_msg + weight_l2 * loss_l2

    result = {
        "loss_msg": loss_msg,
        "loss_l2": loss_l2,
    }

    if use_lpips and lpips_fn is not None:
        orig_scaled = original * 2 - 1
        enc_scaled = encoded * 2 - 1
        loss_lpips = lpips_fn(enc_scaled, orig_scaled).mean()
        total = total + weight_lpips * loss_lpips
        result["loss_lpips"] = loss_lpips

    result["loss"] = total
    return result
