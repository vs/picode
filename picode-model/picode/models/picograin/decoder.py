"""PicoGrain decoder: re-exports PicoTrust decoder.

PicoGrain reuses the PicoTrust CNN+STN decoder unchanged. The grain
texture is designed to be decoded by the same architecture — only the
encoder differs.
"""

from picode.models.picotrust.decoder import Decoder

__all__ = ["Decoder"]
