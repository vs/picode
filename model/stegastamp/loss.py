"""Loss functions for StegaStamp training."""

from collections.abc import Callable
from typing import Any

import torch.nn.functional as F
from torch import Tensor


def message_loss(decoded: Tensor, message: Tensor) -> Tensor:
    """Binary cross-entropy loss for message recovery."""
    return F.binary_cross_entropy(decoded, message)


def image_loss(encoded: Tensor, original: Tensor) -> Tensor:
    """L2 (MSE) loss between encoded and original image."""
    return F.mse_loss(encoded, original)


def compute_loss(
    original: Tensor,
    encoded: Tensor,
    message: Tensor,
    decoded: Tensor,
    lpips_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    use_lpips: bool = True,
    weight_msg: float = 1.0,
    weight_l2: float = 2.0,
    weight_lpips: float = 1.0,
) -> dict[str, Tensor]:
    """Compute combined training loss."""
    loss_msg = message_loss(decoded, message)
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
