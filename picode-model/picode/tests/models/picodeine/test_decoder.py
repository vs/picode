"""Tests for Picodeine decoder with STN."""

import pytest
import torch

from picode.models.picodeine.decoder import Decoder


class TestDecoderArchitecture:
    """Test decoder structure and initialization."""

    def test_no_batchnorm(self) -> None:
        decoder = Decoder(num_bits=127)
        for module in decoder.modules():
            assert not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))

    def test_has_stn(self) -> None:
        decoder = Decoder(num_bits=127)
        assert hasattr(decoder, "stn_params")
        assert hasattr(decoder, "stn_fc_weight")
        assert hasattr(decoder, "stn_fc_bias")

    def test_stn_identity_init(self) -> None:
        decoder = Decoder(num_bits=127)
        expected_bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert torch.allclose(decoder.stn_fc_bias.data, expected_bias)
        assert torch.allclose(decoder.stn_fc_weight.data, torch.zeros(128, 6))

    def test_rejects_non_divisible_input_size(self) -> None:
        """input_size must be divisible by 32 for stride-2 conv stack."""
        with pytest.raises(ValueError, match="divisible by 32"):
            Decoder(num_bits=127, input_size=400)

    def test_num_bits_attribute(self) -> None:
        decoder = Decoder(num_bits=127)
        assert decoder.num_bits == 127

    def test_fc_head_outputs_127(self) -> None:
        decoder = Decoder(num_bits=127)
        # Find the last linear layer
        last_linear = None
        for m in decoder.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        assert last_linear is not None
        assert last_linear.out_features == 127


class TestDecoderForward:
    """Test decoder forward pass."""

    def test_output_shape(self, sample_image: torch.Tensor) -> None:
        decoder = Decoder(num_bits=127)
        logits = decoder(sample_image)
        assert logits.shape == (2, 127)

    def test_output_is_logits(self, sample_image: torch.Tensor) -> None:
        decoder = Decoder(num_bits=127)
        logits = decoder(sample_image)
        # Logits are unbounded
        assert logits.min() < 0.5 or logits.max() > 0.5

    def test_gradient_flow(self, sample_image: torch.Tensor) -> None:
        decoder = Decoder(num_bits=127)
        sample_image.requires_grad_(True)
        logits = decoder(sample_image)
        logits.sum().backward()
        assert sample_image.grad is not None
        assert sample_image.grad.abs().sum() > 0

    def test_decode_method(self, sample_image: torch.Tensor) -> None:
        decoder = Decoder(num_bits=127)
        bits = decoder.decode(sample_image)
        assert bits.shape == (2, 127)
        assert set(bits.unique().tolist()).issubset({0.0, 1.0})


class TestSTNRegularization:
    """Test STN scale regularization."""

    def test_stn_reg_near_zero_at_init(self) -> None:
        decoder = Decoder(num_bits=127)
        reg = decoder.stn_scale_reg()
        assert reg.item() < 1e-6

    def test_stn_reg_increases_with_deviation(self) -> None:
        decoder = Decoder(num_bits=127)
        decoder.stn_fc_bias.data = torch.tensor([2.0, 1.0, 1.0, 1.0, 2.0, 1.0])
        reg = decoder.stn_scale_reg()
        assert reg.item() > 0.01


class TestSTNTrainability:
    """Test STN parameter freezing/unfreezing."""

    def test_stn_trainable_by_default(self) -> None:
        decoder = Decoder(num_bits=127)
        assert decoder.stn_fc_weight.requires_grad
        assert decoder.stn_fc_bias.requires_grad

    def test_freeze_stn_linear_option(self) -> None:
        decoder = Decoder(num_bits=127, freeze_stn_linear=True)
        assert not decoder.stn_fc_weight.requires_grad
        assert not decoder.stn_fc_bias.requires_grad

    def test_unfreeze_stn_linear(self) -> None:
        decoder = Decoder(num_bits=127, freeze_stn_linear=True)
        decoder.unfreeze_stn_linear()
        assert decoder.stn_fc_weight.requires_grad
        assert decoder.stn_fc_bias.requires_grad

    def test_freeze_stn_linear_method(self) -> None:
        decoder = Decoder(num_bits=127)
        decoder.freeze_stn_linear()
        assert not decoder.stn_fc_weight.requires_grad
        assert not decoder.stn_fc_bias.requires_grad
