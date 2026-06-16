"""Tests for PicodeTier factory integration."""

import torch

from picode.models.factory import create_decoder, create_encoder
from picode.models.picodetier.encoder import Encoder
from picode.models.picodetier.decoder import Decoder
from picode.models.picodetier.tiers import MAX_BITS
from picode.training.config import ModelConfig


class TestFactoryCreatesPicodetier:

    def test_creates_encoder(self):
        mc = ModelConfig(type="picodetier", encoder_size=512)
        enc = create_encoder(mc, num_bits=MAX_BITS)
        assert isinstance(enc, Encoder)
        assert enc.image_size == 512

    def test_creates_decoder(self):
        mc = ModelConfig(type="picodetier", decoder_size=256)
        dec = create_decoder(mc, num_bits=MAX_BITS)
        assert isinstance(dec, Decoder)
        assert dec.image_size == 256

    def test_encoder_forward_via_factory(self):
        mc = ModelConfig(type="picodetier", encoder_size=256)
        enc = create_encoder(mc, num_bits=MAX_BITS)
        img = torch.rand(1, 3, 256, 256)
        msg = torch.randint(0, 2, (1, MAX_BITS)).float()
        tier = torch.tensor([2])
        result = enc(img, msg, tier)
        assert result["encoded"].shape == (1, 3, 256, 256)

    def test_decoder_forward_via_factory(self):
        mc = ModelConfig(type="picodetier", decoder_size=256)
        dec = create_decoder(mc, num_bits=MAX_BITS)
        img = torch.rand(1, 3, 256, 256)
        logits, tier_logits = dec(img)
        assert logits.shape == (1, MAX_BITS)

    def test_other_models_unaffected(self):
        """Existing model types still work."""
        mc = ModelConfig(type="stegastamp")
        enc = create_encoder(mc, num_bits=100)
        assert not isinstance(enc, Encoder)
