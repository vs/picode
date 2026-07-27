"""Model factory for creating encoder/decoder pairs.

This module provides factory functions to create encoder and decoder instances
based on model configuration. Supported model types:

- **stegastamp**: Original StegaStamp architecture (400x400, 100 bits)
- **picodelite**: Optimized mobile architecture (800x800 encoder, 320x320 decoder, 63 bits)
- **picodeframe**: Frame-based encoding (400x400, 127 bits BCH(127,64), preserves original image)
- **picotrust**: StegaStamp U-Net + TrustMark enhancements (256x256, 100 bits, CNN decoder)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from picode.models.base import Decoder as BaseDecoder
from picode.models.base import Encoder as BaseEncoder

if TYPE_CHECKING:
    from picode.training.config import ModelConfig


def create_encoder(
    config: ModelConfig,
    num_bits: int,
    strength: float | None = None,
    use_mask: bool = False,
    max_residual_amplitude: float = 0.0,
    residual_blur_sigma: float = 0.0,
    strength_conditioned: bool = False,
) -> BaseEncoder:
    """Create an encoder based on model configuration.

    Factory function that instantiates the appropriate encoder class
    based on the model type specified in the configuration.

    Args:
        config: Model configuration containing the model type.
        num_bits: Number of bits in the message to encode.
        strength: Residual amplitude bound for PicoTrust v2 (tanh scaling).
        use_mask: Whether to enable learned spatial mask (PicoTrust v2).

    Returns:
        An encoder instance (StegaStamp, PicodeLite, PicodeFrame, or PicoTrust).

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
    elif config.type == "picodeframe":
        from picode.models.picodeframe import Encoder as FrameEncoder

        return FrameEncoder(
            num_bits=num_bits, max_residual_amplitude=max_residual_amplitude,
        )
    elif config.type == "picotrust":
        from picode.models.picotrust import Encoder as PicoTrustEncoder

        return PicoTrustEncoder(
            num_bits=num_bits, image_size=config.encoder_size,
            strength=strength, use_mask=use_mask,
            residual_blur_sigma=residual_blur_sigma,
            strength_conditioned=strength_conditioned,
        )
    elif config.type == "picotier":
        from picode.models.picotier import Encoder as TierEncoder

        return TierEncoder(image_size=config.encoder_size)
    elif config.type == "picograin":
        from picode.models.picograin import Encoder as PicoGrainEncoder

        return PicoGrainEncoder(
            num_bits=num_bits, image_size=config.encoder_size,
            strength=strength,
        )
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

        return LiteDecoder(num_bits=num_bits, input_size=config.decoder_size)
    elif config.type == "picodeframe":
        from picode.models.picodeframe import Decoder as FrameDecoder

        return FrameDecoder(num_bits=num_bits)
    elif config.type == "picotrust":
        from picode.models.picotrust import Decoder as PicoTrustDecoder

        return PicoTrustDecoder(num_bits=num_bits, image_size=config.decoder_size)
    elif config.type == "picotier":
        from picode.models.picotier import Decoder as TierDecoder

        return TierDecoder(image_size=config.decoder_size)
    elif config.type == "picograin":
        from picode.models.picograin import Decoder as PicoGrainDecoder

        return PicoGrainDecoder(num_bits=num_bits, image_size=config.decoder_size)
    else:
        raise ValueError(f"Unknown model type: {config.type}")
