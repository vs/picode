"""Tests for Picodeine train step."""

import torch

from picode.models.picodeine.decoder import Decoder
from picode.models.picodeine.encoder import Encoder
from picode.models.picodeine.train import train_step


class TestTrainStep:
    """Test the train_step function."""

    def test_returns_expected_keys(self) -> None:
        encoder = Encoder(num_bits=127)
        decoder = Decoder(num_bits=127)
        images = torch.rand(2, 3, 512, 512)
        result = train_step(encoder, decoder, images, use_lpips=False)
        assert "loss" in result
        assert "loss_msg" in result
        assert "loss_l2" in result
        assert "loss_stn_reg" in result
        assert "bit_accuracy" in result

    def test_values_are_floats(self) -> None:
        encoder = Encoder(num_bits=127)
        decoder = Decoder(num_bits=127)
        images = torch.rand(2, 3, 512, 512)
        result = train_step(encoder, decoder, images, use_lpips=False)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"

    def test_optimizer_updates_params(self) -> None:
        encoder = Encoder(num_bits=127)
        decoder = Decoder(num_bits=127)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        images = torch.rand(2, 3, 512, 512)
        # Snapshot a parameter before
        param_before = encoder.conv1.weight.data.clone()
        train_step(encoder, decoder, images, optimizer=optimizer, use_lpips=False)
        # Parameter should have changed
        assert not torch.equal(param_before, encoder.conv1.weight.data)

    def test_no_optimizer_no_update(self) -> None:
        encoder = Encoder(num_bits=127)
        decoder = Decoder(num_bits=127)
        images = torch.rand(2, 3, 512, 512)
        param_before = encoder.conv1.weight.data.clone()
        train_step(encoder, decoder, images, optimizer=None, use_lpips=False)
        assert torch.equal(param_before, encoder.conv1.weight.data)

    def test_bit_accuracy_in_range(self) -> None:
        encoder = Encoder(num_bits=127)
        decoder = Decoder(num_bits=127)
        images = torch.rand(2, 3, 512, 512)
        result = train_step(encoder, decoder, images, use_lpips=False)
        assert 0.0 <= result["bit_accuracy"] <= 1.0
