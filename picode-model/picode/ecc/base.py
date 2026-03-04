"""Base class for error correction codes."""

from abc import ABC, abstractmethod

from torch import Tensor


class ECC(ABC):
    """Abstract base class for error correction codes.

    An ECC encodes messages with redundancy to allow error detection
    and correction after transmission through a noisy channel.
    """

    @abstractmethod
    def encode(self, message: Tensor) -> Tensor:
        """Encode a message with error correction redundancy.

        Args:
            message: Binary message tensor (B, k) where k is the message length.

        Returns:
            Encoded tensor (B, n) where n > k includes redundancy bits.
        """
        pass

    @abstractmethod
    def decode(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        """Decode and correct errors in an encoded message.

        Args:
            encoded: Encoded tensor (B, n) possibly with errors.

        Returns:
            Tuple of:
                - Corrected message tensor (B, k).
                - Success boolean tensor (B,) indicating successful decodes.
        """
        pass

    @property
    @abstractmethod
    def rate(self) -> float:
        """Code rate (k/n) - ratio of data bits to total bits.

        Higher rate means less redundancy but lower error correction capability.
        """
        pass

    @property
    @abstractmethod
    def message_length(self) -> int:
        """Number of data bits (k) in a message."""
        pass

    @property
    @abstractmethod
    def codeword_length(self) -> int:
        """Number of bits (n) in an encoded codeword."""
        pass
