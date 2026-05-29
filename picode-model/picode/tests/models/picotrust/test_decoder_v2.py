"""Tests for PicoTrust decoder at 512x512."""

import torch

from picode.models.picotrust.decoder import Decoder


def test_decoder_accepts_image_size():
    dec = Decoder(num_bits=100, image_size=512)
    assert dec.image_size == 512


def test_decoder_512_output_shape():
    dec = Decoder(num_bits=100, image_size=512)
    image = torch.rand(2, 3, 512, 512)
    logits = dec(image)
    assert logits.shape == (2, 100)


def test_decoder_256_backward_compat():
    dec = Decoder(num_bits=100)
    image = torch.rand(2, 3, 256, 256)
    logits = dec(image)
    assert logits.shape == (2, 100)


def test_decoder_512_gradient_flow():
    dec = Decoder(num_bits=100, image_size=512)
    image = torch.rand(2, 3, 512, 512, requires_grad=True)
    logits = dec(image)
    logits.sum().backward()
    assert image.grad is not None
