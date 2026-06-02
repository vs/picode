"""Tests for model factory PicoTrust v2 support."""

from picode.models.factory import create_decoder, create_encoder
from picode.training.config import ModelConfig


def test_factory_creates_512_encoder():
    config = ModelConfig(type="picotrust", encoder_size=512, decoder_size=512)
    enc = create_encoder(config, num_bits=100)
    assert enc.image_size == 512


def test_factory_creates_512_decoder():
    config = ModelConfig(type="picotrust", encoder_size=512, decoder_size=512)
    dec = create_decoder(config, num_bits=100)
    assert dec.image_size == 512


def test_factory_passes_strength_to_encoder():
    config = ModelConfig(type="picotrust", encoder_size=512, decoder_size=512)
    enc = create_encoder(config, num_bits=100, strength=0.03, use_mask=True)
    assert enc.strength == 0.03
    assert enc.mask_head is not None


def test_factory_default_no_strength():
    config = ModelConfig(type="picotrust", encoder_size=256, decoder_size=256)
    enc = create_encoder(config, num_bits=100)
    assert enc.strength is None


def test_factory_other_models_unaffected():
    """StegaStamp factory call should still work without new params."""
    config = ModelConfig(type="stegastamp")
    enc = create_encoder(config, num_bits=100)
    assert not hasattr(enc, 'strength') or enc.strength is None
