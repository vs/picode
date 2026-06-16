"""PicodeTier: variable-bit steganography with tier-coupled strength.

Supports 4 tiers (16/32/64/96 bits) with per-tier residual strength.
Encoder is conditioned on the tier; decoder auto-detects the tier.
"""

from picode.models.picodetier.decoder import Decoder
from picode.models.picodetier.encoder import Encoder
from picode.models.picodetier.tiers import EMBED_DIM, MAX_BITS, NUM_TIERS, TIERS

__all__ = ["TIERS", "NUM_TIERS", "MAX_BITS", "EMBED_DIM", "Encoder", "Decoder"]
