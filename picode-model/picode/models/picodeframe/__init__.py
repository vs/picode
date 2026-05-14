"""PicodeFrame steganography model.

Frame-based encoding that preserves the original image exactly
by hiding messages in a generated border frame.
"""

from picode.models.picodeframe.decoder import Decoder
from picode.models.picodeframe.encoder import Encoder

__all__ = ["Encoder", "Decoder"]
