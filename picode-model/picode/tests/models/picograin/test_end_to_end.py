"""End-to-end smoke test: encode -> distort -> decode."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from picode.models.picograin.decoder import Decoder
from picode.models.picograin.encoder import Encoder


def test_encode_decode_round_trip():
    """Encoder + decoder produce 127 logits from encoded image."""
    enc = Encoder(num_bits=127, image_size=256, strength=0.10)
    dec = Decoder(num_bits=127, image_size=256)

    image = torch.rand(2, 3, 256, 256)
    message = torch.randint(0, 2, (2, 127)).float()

    encoded = enc(image, message)["encoded"]
    logits = dec(encoded)

    assert logits.shape == (2, 127)
    loss = F.binary_cross_entropy_with_logits(logits, message)
    loss.backward()
    assert cast(nn.Conv2d, enc.e_post[-1]).weight.grad is not None
    assert dec.decoder[0].weight.grad is not None


def test_encode_produces_visible_grain():
    """Encoded image should differ from original in a grain-like pattern."""
    enc = Encoder(num_bits=127, image_size=256, strength=0.10)
    with torch.no_grad():
        torch.manual_seed(42)
        import torch.nn as nn

        nn.init.kaiming_normal_(cast(nn.Conv2d, enc.e_post[-1]).weight)

    image = torch.rand(1, 3, 256, 256)
    message = torch.randint(0, 2, (1, 127)).float()
    encoded = enc(image, message)["encoded"]

    residual = encoded - image
    assert residual.abs().mean() > 1e-4
    assert residual.abs().max() < 0.10 + 1e-6
    assert torch.allclose(residual[:, 0], residual[:, 1], atol=1e-6)


def test_bch_integration():
    """BCH(127, 71, 9) encodes and decodes through the model."""
    from picode.ecc.bch import BCH

    bch = BCH(n=127, k=71)
    assert bch.t == 9
    assert bch.codeword_length == 127
    assert bch.message_length == 71

    payload = torch.randint(0, 2, (2, 71)).float()
    codeword = bch.encode(payload)
    assert codeword.shape == (2, 127)

    enc = Encoder(num_bits=127, image_size=256, strength=0.10)
    dec = Decoder(num_bits=127, image_size=256)
    image = torch.rand(2, 3, 256, 256)

    encoded = enc(image, codeword)["encoded"]
    logits = dec(encoded)
    hard_bits = (torch.sigmoid(logits) > 0.5).float()
    decoded_msg, success = bch.decode(hard_bits)

    assert decoded_msg.shape == (2, 71)
    assert success.shape == (2,)
