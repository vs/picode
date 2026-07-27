"""Tests for PicoGrain factory registration."""

from picode.models.factory import create_decoder, create_encoder
from picode.models.picograin.decoder import Decoder
from picode.models.picograin.encoder import Encoder
from picode.training.config import ModelConfig


def test_create_picograin_encoder():
    config = ModelConfig(type="picograin", encoder_size=256, decoder_size=256)
    enc = create_encoder(config, num_bits=127, strength=0.10)
    assert isinstance(enc, Encoder)
    assert enc.num_bits == 127


def test_create_picograin_decoder():
    config = ModelConfig(type="picograin", encoder_size=256, decoder_size=256)
    dec = create_decoder(config, num_bits=127)
    assert isinstance(dec, Decoder)
    assert dec.num_bits == 127
