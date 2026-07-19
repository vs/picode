"""PicoTier: variable-bit steganography with tier-coupled strength and Sobel masking.

Supports 4 tiers (UHQ/HQ/MQ/LQ) with per-tier residual strength and Sobel mask floor.
Encoder is conditioned on the tier; decoder auto-detects the tier.
"""

from picode.models.picotier.decoder import Decoder
from picode.models.picotier.encoder import Encoder
from picode.models.picotier.tiers import (
    EMBED_DIM,
    MAX_BITS,
    NUM_TIERS,
    TIER_NAMES,
    TIERS,
)

__all__ = [
    "TIERS", "TIER_NAMES", "NUM_TIERS", "MAX_BITS", "EMBED_DIM",
    "Encoder", "Decoder",
]
