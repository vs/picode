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
    """Execute a single training step."""
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
        """Decode a message from an image."""
        self.decoder.eval()
        with torch.no_grad():
            result: Tensor = self.decoder(image.to(self.device))
            return result

    def decode_binary(self, image: Tensor) -> Tensor:
        """Decode binary message from an image."""
        logits = self.decode(image)
        return (logits > 0).float()
