"""Tests for PicoGrain decoder (reuses PicoTrust architecture at 127 bits)."""

import torch

from picode.models.picograin.decoder import Decoder


def test_decoder_127_bits():
    dec = Decoder(num_bits=127, image_size=512)
    assert dec.num_bits == 127


def test_decoder_output_shape():
    dec = Decoder(num_bits=127, image_size=512)
    image = torch.rand(2, 3, 512, 512)
    logits = dec(image)
    assert logits.shape == (2, 127)


def test_decoder_gradient_flow():
    dec = Decoder(num_bits=127, image_size=512)
    image = torch.rand(2, 3, 512, 512, requires_grad=True)
    logits = dec(image)
    logits.sum().backward()
    assert image.grad is not None


def test_decoder_stn_reg():
    dec = Decoder(num_bits=127, image_size=512)
    reg = dec.stn_scale_reg()
    assert reg.shape == ()
    assert reg.item() >= 0.0
