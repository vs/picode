"""PicodeLite steganography model.

Optimized for:
- Reduced visual artifacts (800x800 encoder with learned upsampling)
- Fast mobile inference (320x320 decoder, no STN, ~640K params)
- 63-bit messages with BCH(63,36) error correction
"""

from picode.models.picodelite.decoder import Decoder
from picode.models.picodelite.encoder import Encoder

__all__ = ["Encoder", "Decoder"]
