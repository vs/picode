"""Picode v3 encoder/decoder implementation.

Based on StegaStamp architecture, adapted for 127-bit BCH codewords.
Designed for imperceptible steganography with ECC error recovery.

Key differences from StegaStamp:
- Default num_bits=127 (for BCH(127, 50) encoding)
- Trained with lighter distortions
- Lower message loss weight (0.7 vs 1.0)
"""

from picode.models.picode_v3.decoder import Decoder
from picode.models.picode_v3.encoder import Encoder

__all__ = [
    "Encoder",
    "Decoder",
]
