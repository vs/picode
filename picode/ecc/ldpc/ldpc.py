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
        import torch

        batch_size = message.shape[0]
        device = message.device
        dtype = message.dtype

        # Convert to numpy for pyldpc
        msg_np = message.detach().cpu().numpy().astype(np.int_)

        # Encode each message in batch
        encoded_list = []
        for i in range(batch_size):
            # G is (n, k), message is (k,), result is (n,)
            codeword = (self._G @ msg_np[i]) % 2
            encoded_list.append(codeword)

        # Stack and convert back to tensor
        encoded_np = np.stack(encoded_list, axis=0)
        return torch.from_numpy(encoded_np).to(device=device, dtype=dtype)

    def decode(self, received: Tensor) -> Tensor:
        """Decode received soft values using belief propagation.

        Args:
            received: Soft probabilities (B, n) with values in [0, 1]
                representing P(bit=1).

        Returns:
            Decoded binary tensor (B, k) with values in {0, 1}.
        """
        import torch
        from pyldpc import decode as ldpc_decode
        from pyldpc import get_message

        batch_size = received.shape[0]
        device = received.device
        dtype = received.dtype

        # Convert to numpy with float64 (required by pyldpc's numba-jit decoder)
        recv_np = received.detach().cpu().numpy().astype(np.float64)

        # Convert [0, 1] probabilities to BPSK-like signal [-1, 1]
        # P(bit=1) = 0 -> y = -1 (strong 0)
        # P(bit=1) = 1 -> y = +1 (strong 1)
        # P(bit=1) = 0.5 -> y = 0 (uncertain)
        y = 2.0 * recv_np - 1.0

        # Decode each codeword in batch
        decoded_list = []
        for i in range(batch_size):
            # Decode using belief propagation
            codeword = ldpc_decode(self._H, y[i], self._snr)
            # Extract original message from systematic codeword
            message = get_message(self._G, codeword)
            decoded_list.append(message)

        # Stack and convert back to tensor
        decoded_np = np.stack(decoded_list, axis=0)
        return torch.from_numpy(decoded_np).to(device=device, dtype=dtype)
