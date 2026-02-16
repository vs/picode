"""Base classes for steganography encoder/decoder models."""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module, ABC):
    """Abstract base class for steganography encoders.

    An encoder embeds a binary message into an image imperceptibly.
    """

    @abstractmethod
    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode a message into an image.

        Args:
            image: Input image tensor (B, C, H, W) in [0, 1].
            message: Binary message tensor (B, num_bits).

        Returns:
            Encoded image tensor (B, C, H, W) in [0, 1].
        """
        pass


class Decoder(nn.Module, ABC):
    """Abstract base class for steganography decoders.

    A decoder extracts a binary message from an encoded (and possibly distorted) image.

    Note: Implementations may return either logits or probabilities from forward().
    The decode() method always returns binary bits.
    """

    @abstractmethod
    def forward(self, image: Tensor) -> Tensor:
        """Extract message representation from an image.

        Args:
            image: Input image tensor (B, C, H, W) in [0, 1].

        Returns:
            Message tensor (B, num_bits). May be logits or probabilities
            depending on implementation.
        """
        pass

    def decode(self, image: Tensor) -> Tensor:
        """Extract binary message from an image.

        Args:
            image: Input image tensor (B, C, H, W) in [0, 1].

        Returns:
            Binary message tensor (B, num_bits).
        """
        import torch
        output = self.forward(image)
        probs = torch.sigmoid(output) if output.min() < 0 or output.max() > 1 else output
        return (probs > 0.5).float()
