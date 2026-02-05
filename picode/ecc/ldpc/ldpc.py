"""LDPC error correction code using pyldpc library."""

import numpy as np
from numpy.typing import NDArray
from pyldpc import make_ldpc
from torch import Tensor

from picode.ecc.base import ECC


class LDPC(ECC):
    """LDPC error correction using belief propagation decoding.

    Uses pyldpc library for matrix generation and decoding.

    Args:
        n: Codeword length (total bits after encoding).
        d_v: Variable node degree. Default 3.
        d_c: Check node degree. Default 6.
        snr: Signal-to-noise ratio for belief propagation decoder. Default 10.0.
        seed: Random seed for reproducible matrix generation.

    Example:
        >>> ldpc = LDPC(n=200, d_v=3, d_c=6)
        >>> message = torch.randint(0, 2, (4, ldpc.message_length))
        >>> encoded = ldpc.encode(message)
        >>> decoded = ldpc.decode(encoded.float())
    """

    def __init__(
        self,
        n: int,
        d_v: int = 3,
        d_c: int = 6,
        snr: float = 10.0,
        seed: int | None = None,
    ) -> None:
        """Initialize LDPC code with given parameters."""
        self._n = n
        self._d_v = d_v
        self._d_c = d_c
        self._snr = snr

        # Generate LDPC matrices
        self._H, self._G = make_ldpc(
            n,
            d_v,
            d_c,
            systematic=True,
            sparse=True,
            seed=seed,
        )

        # k is derived from generator matrix shape
        self._k = self._G.shape[1]

    @property
    def rate(self) -> float:
        """Code rate (k/n)."""
        return self._k / self._n

    @property
    def message_length(self) -> int:
        """Number of data bits (k) in a message."""
        return self._k

    @property
    def codeword_length(self) -> int:
        """Number of bits (n) in an encoded codeword."""
        return self._n

    @property
    def H(self) -> NDArray[np.int_]:
        """Parity check matrix (n-k, n)."""
        return self._H

    @property
    def G(self) -> NDArray[np.int_]:
        """Generator matrix (n, k)."""
        return self._G

    def encode(self, message: Tensor) -> Tensor:
        """Encode binary messages with LDPC redundancy.

        Args:
            message: Binary tensor (B, k) with values in {0, 1}.

        Returns:
            Encoded tensor (B, n) with values in {0, 1}.
        """
        raise NotImplementedError("encode not yet implemented")

    def decode(self, received: Tensor) -> Tensor:
        """Decode received soft values using belief propagation.

        Args:
            received: Soft probabilities (B, n) with values in [0, 1]
                representing P(bit=1).

        Returns:
            Decoded binary tensor (B, k) with values in {0, 1}.
        """
        raise NotImplementedError("decode not yet implemented")
