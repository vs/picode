"""Picodeine training step.

Single training step: encode -> distort -> decode -> loss -> backward.
Follows the same pattern as StegaStamp's train_step.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from picode.models.picodeine.decoder import Decoder
from picode.models.picodeine.encoder import Encoder
from picode.models.picodeine.loss import compute_picodeine_loss


def train_step(
    encoder: Encoder,
    decoder: Decoder,
    images: Tensor,
    distortion: Callable[[Tensor], Tensor] | None = None,
    optimizer: Optimizer | None = None,
    num_bits: int = 127,
    use_lpips: bool = True,
    lpips_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    weight_msg: float = 7.0,
    weight_l2: float = 1.0,
    weight_lpips: float = 1.5,
    weight_stn_reg: float = 0.1,
) -> dict[str, float]:
    """Execute single training step.

    Args:
        encoder: Picodeine encoder.
        decoder: Picodeine decoder.
        images: (B, 3, H, W) training images in [0, 1].
        distortion: Optional distortion function.
        optimizer: Optional optimizer (if None, no backward pass).
        num_bits: Number of message bits.
        use_lpips: Whether to use LPIPS loss.
        lpips_fn: LPIPS function.
        weight_msg: Message loss weight.
        weight_l2: L2 loss weight.
        weight_lpips: LPIPS loss weight.
        weight_stn_reg: STN regularization weight.

    Returns:
        Dict of loss values (detached floats).
    """
    device = images.device
    batch_size = images.shape[0]

    # Generate random messages
    messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

    # Encode
    encoded = encoder(images, messages)

    # Apply distortions
    if distortion is not None:
        distorted = distortion(encoded)
    else:
        distorted = encoded

    # Decode
    decoded_logits = decoder(distorted)

    # Compute loss
    losses = compute_picodeine_loss(
        original=images,
        encoded=encoded,
        messages=messages,
        decoded_logits=decoded_logits,
        decoder=decoder,
        lpips_fn=lpips_fn,
        use_lpips=use_lpips,
        weight_msg=weight_msg,
        weight_l2=weight_l2,
        weight_lpips=weight_lpips,
        weight_stn_reg=weight_stn_reg,
    )

    # Optional backward pass
    if optimizer is not None:
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

    # Compute metrics
    with torch.no_grad():
        probs = torch.sigmoid(decoded_logits)
        predicted = (probs > 0.5).float()
        bit_accuracy = (predicted == messages).float().mean()

    result = {k: v.detach().item() for k, v in losses.items()}
    result["bit_accuracy"] = bit_accuracy.item()
    return result
