"""Training utilities for StegaStamp."""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from picode.models.stegastamp.decoder import Decoder
from picode.models.stegastamp.encoder import Encoder
from picode.models.stegastamp.loss import compute_loss


def train_step(
    encoder: Encoder,
    decoder: Decoder,
    images: Tensor,
    distortion: Callable[[Tensor], Tensor] | None = None,
    optimizer: Optimizer | None = None,
    num_bits: int = 100,
    use_lpips: bool = True,
    lpips_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, float]:
    """Execute a single training step.

    The decoder outputs raw logits, and the loss function uses BCE with logits
    for numerical stability.

    Args:
        encoder: StegaStamp encoder model.
        decoder: StegaStamp decoder model (outputs logits).
        images: Batch of images (B, 3, H, W) in [0, 1].
        distortion: Optional distortion to apply to encoded images.
        optimizer: Optional optimizer. If None, no backward pass is performed.
        num_bits: Number of message bits.
        use_lpips: Whether to include LPIPS loss.
        lpips_fn: LPIPS function. Required if use_lpips=True.

    Returns:
        Dictionary of loss values (detached floats).
    """
    batch_size = images.shape[0]
    device = images.device

    messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()
    encoded = encoder(images, messages)

    if distortion is not None:
        distorted = distortion(encoded)
    else:
        distorted = encoded

    decoded_logits = decoder(distorted)

    losses = compute_loss(
        original=images,
        encoded=encoded,
        message=messages,
        decoded_logits=decoded_logits,
        lpips_fn=lpips_fn,
        use_lpips=use_lpips,
    )

    if optimizer is not None:
        optimizer.zero_grad()
        losses["loss"].backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    return {k: v.detach().item() for k, v in losses.items()}


class StegaStampTrainer:
    """Convenience wrapper for training StegaStamp models."""

    def __init__(
        self,
        num_bits: int = 100,
        device: torch.device | None = None,
    ) -> None:
        self.num_bits = num_bits
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = Encoder(num_bits=num_bits).to(self.device)
        self.decoder = Decoder(num_bits=num_bits).to(self.device)

    def encode(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode a message into an image."""
        self.encoder.eval()
        with torch.no_grad():
            result: Tensor = self.encoder(image.to(self.device), message.to(self.device))
            return result

    def decode(self, image: Tensor) -> Tensor:
        """Decode message probabilities from an image.

        Args:
            image: Input image tensor (B, 3, H, W) in [0, 1].

        Returns:
            Message probabilities (B, num_bits) in [0, 1].
        """
        self.decoder.eval()
        with torch.no_grad():
            logits = self.decoder(image.to(self.device))
            return torch.sigmoid(logits)

    def decode_binary(self, image: Tensor) -> Tensor:
        """Decode binary message from an image.

        Args:
            image: Input image tensor (B, 3, H, W) in [0, 1].

        Returns:
            Binary message (B, num_bits) with values {0, 1}.
        """
        probs = self.decode(image)
        return (probs > 0.5).float()
