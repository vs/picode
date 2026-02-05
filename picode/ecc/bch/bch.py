"""BCH error correction code implementation."""

import galois
import numpy as np
import torch
from torch import Tensor

from picode.ecc.base import ECC


class BCH(ECC):
    """BCH error correction code using galois library.

    BCH codes add redundancy to messages, allowing recovery from bit errors.
    Default configuration BCH(127, 64) corrects up to 10 bit errors,
    suitable for print-and-scan robustness.

    Args:
        n: Codeword length (must be 2^m - 1, e.g., 7, 15, 31, 63, 127, 255).
            Default: 127.
        k: Message length (valid values depend on n). Default: 64.

    Example:
        >>> bch = BCH(127, 64)
        >>> message = torch.randint(0, 2, (4, 64)).float()
        >>> codeword = bch.encode(message)
        >>> decoded, success = bch.decode(codeword)
        >>> assert (decoded == message).all()
    """

    def __init__(self, n: int = 127, k: int = 64) -> None:
        """Initialize BCH code with given parameters."""
        self._bch = galois.BCH(n, k)

    def encode(self, message: Tensor) -> Tensor:
        """Encode messages with BCH redundancy.

        Args:
            message: Binary tensor (B, k) with values in {0, 1}.

        Returns:
            Codeword tensor (B, n) with BCH parity bits appended.
        """
        device = message.device
        msg_np = message.cpu().numpy().astype(int)
        codewords = self._bch.encode(msg_np)
        return torch.from_numpy(np.asarray(codewords)).float().to(device)

    def decode(self, codeword: Tensor) -> tuple[Tensor, Tensor]:
        """Decode and correct errors in codewords.

        Args:
            codeword: Binary tensor (B, n), possibly with bit errors.

        Returns:
            Tuple of:
                - messages: Corrected message tensor (B, k).
                  Failed decodes contain original bits (uncorrected).
                - success: Boolean tensor (B,) indicating successful decodes.
        """
        device = codeword.device
        hard = (codeword > 0.5).cpu().numpy().astype(int)

        batch_size = hard.shape[0]
        messages = np.zeros((batch_size, self._bch.k), dtype=int)
        success = np.ones(batch_size, dtype=bool)

        for i in range(batch_size):
            decoded, num_errors = self._bch.decode(hard[i : i + 1], errors=True)
            num_errors_arr = np.atleast_1d(num_errors)
            if num_errors_arr[0] == -1:
                # Decoding failed - return original message bits
                messages[i] = hard[i, : self._bch.k]
                success[i] = False
            else:
                messages[i] = np.asarray(decoded)[0, : self._bch.k]

        return (
            torch.from_numpy(messages).float().to(device),
            torch.from_numpy(success).to(device),
        )

    @property
    def rate(self) -> float:
        """Code rate (k/n) - ratio of data bits to total bits."""
        return float(self._bch.k) / float(self._bch.n)

    @property
    def message_length(self) -> int:
        """Number of data bits (k) in a message."""
        return int(self._bch.k)

    @property
    def codeword_length(self) -> int:
        """Number of bits (n) in an encoded codeword."""
        return int(self._bch.n)

    @property
    def t(self) -> int:
        """Number of bit errors that can be corrected."""
        return int(self._bch.t)
