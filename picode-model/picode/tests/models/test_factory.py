"""Tests for the model factory module."""

import pytest
import torch

from picode.models.base import Decoder as BaseDecoder
from picode.models.base import Encoder as BaseEncoder
from picode.training.config import ModelConfig


class TestCreateEncoder:
    """Tests for the create_encoder factory function."""

    def test_create_stegastamp_encoder(self) -> None:
        """Factory returns StegaEncoder instance for stegastamp type."""
        from picode.models.factory import create_encoder
        from picode.models.stegastamp import Encoder as StegaEncoder

        config = ModelConfig(type="stegastamp")
        encoder = create_encoder(config, num_bits=100)

        assert isinstance(encoder, StegaEncoder)
        assert isinstance(encoder, BaseEncoder)
        assert encoder.num_bits == 100

    def test_create_picodelite_encoder(self) -> None:
        """Factory returns LiteEncoder instance for picodelite type."""
        from picode.models.factory import create_encoder
        from picode.models.picodelite import Encoder as LiteEncoder

        config = ModelConfig(type="picodelite")
        encoder = create_encoder(config, num_bits=63)

        assert isinstance(encoder, LiteEncoder)
        assert isinstance(encoder, BaseEncoder)
        assert encoder.num_bits == 63

    def test_unknown_model_type_raises(self) -> None:
        """Factory raises ValueError for unknown model type."""
        from picode.models.factory import create_encoder

        config = ModelConfig(type="unknown_model")

        with pytest.raises(ValueError, match="Unknown model type: unknown_model"):
            create_encoder(config, num_bits=100)


class TestCreateDecoder:
    """Tests for the create_decoder factory function."""

    def test_create_stegastamp_decoder(self) -> None:
        """Factory returns StegaDecoder instance for stegastamp type."""
        from picode.models.factory import create_decoder
        from picode.models.stegastamp import Decoder as StegaDecoder

        config = ModelConfig(type="stegastamp")
        decoder = create_decoder(config, num_bits=100)

        assert isinstance(decoder, StegaDecoder)
        assert isinstance(decoder, BaseDecoder)
        assert decoder.num_bits == 100

    def test_create_picodelite_decoder(self) -> None:
        """Factory returns LiteDecoder instance for picodelite type."""
        from picode.models.factory import create_decoder
        from picode.models.picodelite import Decoder as LiteDecoder

        config = ModelConfig(type="picodelite")
        decoder = create_decoder(config, num_bits=63)

        assert isinstance(decoder, LiteDecoder)
        assert isinstance(decoder, BaseDecoder)
        assert decoder.num_bits == 63

    def test_unknown_model_type_raises(self) -> None:
        """Factory raises ValueError for unknown model type."""
        from picode.models.factory import create_decoder

        config = ModelConfig(type="invalid_type")

        with pytest.raises(ValueError, match="Unknown model type: invalid_type"):
            create_decoder(config, num_bits=100)


class TestEncoderOutputShape:
    """Tests for encoder output shapes."""

    def test_stegastamp_encoder_output_shape(self) -> None:
        """StegaStamp encoder outputs 400x400 images."""
        from picode.models.factory import create_encoder

        config = ModelConfig(type="stegastamp", encoder_size=400)
        encoder = create_encoder(config, num_bits=100)

        image = torch.rand(1, 3, 400, 400)
        message = torch.randint(0, 2, (1, 100)).float()
        output = encoder(image, message)

        assert output.shape == (1, 3, 400, 400)

    def test_picodelite_encoder_output_shape(self) -> None:
        """PicodeLite encoder outputs 800x800 images."""
        from picode.models.factory import create_encoder

        config = ModelConfig(type="picodelite", encoder_size=800)
        encoder = create_encoder(config, num_bits=63)

        image = torch.rand(1, 3, 800, 800)
        message = torch.randint(0, 2, (1, 63)).float()
        output = encoder(image, message)

        assert output.shape == (1, 3, 800, 800)


class TestDecoderOutputShape:
    """Tests for decoder output shapes."""

    def test_stegastamp_decoder_output_shape(self) -> None:
        """StegaStamp decoder outputs 100 bits."""
        from picode.models.factory import create_decoder

        config = ModelConfig(type="stegastamp", decoder_size=400)
        decoder = create_decoder(config, num_bits=100)

        image = torch.rand(1, 3, 400, 400)
        output = decoder(image)

        assert output.shape == (1, 100)

    def test_picodelite_decoder_output_shape(self) -> None:
        """PicodeLite decoder outputs 63 bits."""
        from picode.models.factory import create_decoder

        config = ModelConfig(type="picodelite", decoder_size=320)
        decoder = create_decoder(config, num_bits=63)

        image = torch.rand(1, 3, 320, 320)
        output = decoder(image)

        assert output.shape == (1, 63)
