"""Model factory for creating encoder/decoder pairs.

This module provides factory functions to create encoder and decoder instances
based on model configuration. Supported model types:

- **stegastamp**: Original StegaStamp architecture (400x400, 100 bits)
- **picodelite**: Optimized mobile architecture (800x800 encoder, 320x320 decoder, 63 bits)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from picode.models.base import Decoder as BaseDecoder
from picode.models.base import Encoder as BaseEncoder

if TYPE_CHECKING:
    from picode.training.config import ModelConfig


def create_encoder(config: ModelConfig, num_bits: int) -> BaseEncoder:
    """Create an encoder based on model configuration.

    Factory function that instantiates the appropriate encoder class
    based on the model type specified in the configuration.

    Args:
        config: Model configuration containing the model type.
        num_bits: Number of bits in the message to encode.

    Returns:
        An encoder instance (StegaStamp or PicodeLite).

    Raises:
        ValueError: If the model type is not recognized.

    Examples:
        >>> config = ModelConfig(type="stegastamp")
        >>> encoder = create_encoder(config, num_bits=100)

        >>> config = ModelConfig(type="picodelite")
        >>> encoder = create_encoder(config, num_bits=63)
    """
    if config.type == "stegastamp":
        from picode.models.stegastamp import Encoder as StegaEncoder

        return StegaEncoder(num_bits=num_bits)
    elif config.type == "picodelite":
        from picode.models.picodelite import Encoder as LiteEncoder

        return LiteEncoder(num_bits=num_bits)
    else:
        raise ValueError(f"Unknown model type: {config.type}")


def create_decoder(config: ModelConfig, num_bits: int) -> BaseDecoder:
    """Create a decoder based on model configuration.

    Factory function that instantiates the appropriate decoder class
    based on the model type specified in the configuration.

    Args:
        config: Model configuration containing the model type.
        num_bits: Number of bits in the message to decode.

    Returns:
        A decoder instance (StegaStamp or PicodeLite).

    Raises:
        ValueError: If the model type is not recognized.

    Examples:
        >>> config = ModelConfig(type="stegastamp")
        >>> decoder = create_decoder(config, num_bits=100)

        >>> config = ModelConfig(type="picodelite")
        >>> decoder = create_decoder(config, num_bits=63)
    """
    if config.type == "stegastamp":
        from picode.models.stegastamp import Decoder as StegaDecoder

        return StegaDecoder(num_bits=num_bits)
    elif config.type == "picodelite":
        from picode.models.picodelite import Decoder as LiteDecoder

        return LiteDecoder(num_bits=num_bits)
    else:
        raise ValueError(f"Unknown model type: {config.type}")
