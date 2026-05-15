"""Tests for the PicodeFrame decoder module."""

import torch

from picode.models.picodeframe.decoder import Decoder


class TestDecoderArchitecture:
    """Test decoder architecture."""

    def test_no_batchnorm(self) -> None:
        """Decoder has no BatchNorm layers."""
        import torch.nn as nn
        decoder = Decoder(num_bits=96)
        for module in decoder.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), \
                f"Found BatchNorm: {module}"

    def test_has_stn(self) -> None:
        """Decoder has Spatial Transformer Network."""
        decoder = Decoder(num_bits=96)
        assert hasattr(decoder, "stn_params")
        assert hasattr(decoder, "stn_fc_weight")
        assert hasattr(decoder, "stn_fc_bias")

    def test_stn_identity_init(self) -> None:
        """STN is initialized to identity transform."""
        decoder = Decoder(num_bits=96)
        expected_bias = torch.tensor([1., 0., 0., 0., 1., 0.])
        assert torch.allclose(decoder.stn_fc_bias, expected_bias)
        assert torch.allclose(decoder.stn_fc_weight, torch.zeros(128, 6))

    def test_num_bits_attribute(self) -> None:
        """Decoder stores num_bits as 96."""
        decoder = Decoder(num_bits=96)
        assert decoder.num_bits == 96


class TestDecoderForward:
    """Test decoder forward pass."""

    def test_output_shape(self, sample_image: torch.Tensor) -> None:
        """Decoder outputs (B, 96) tensor."""
        decoder = Decoder(num_bits=96)
        result = decoder(sample_image)
        assert result.shape == (2, 96)

    def test_output_is_logits(self, sample_image: torch.Tensor) -> None:
        """Output is logits (unbounded)."""
        decoder = Decoder(num_bits=96)
        result = decoder(sample_image)
        assert result.min() < 0.5 or result.max() > 0.5

    def test_gradient_flow(self, sample_image: torch.Tensor) -> None:
        """Gradients flow through decoder."""
        decoder = Decoder(num_bits=96)
        sample_image.requires_grad_(True)
        result = decoder(sample_image)
        loss = result.sum()
        loss.backward()
        assert sample_image.grad is not None

    def test_decode_method(self, sample_image: torch.Tensor) -> None:
        """Decode method returns binary bits."""
        decoder = Decoder(num_bits=96)
        result = decoder.decode(sample_image)
        assert result.shape == (2, 96)
        assert torch.all((result == 0) | (result == 1))


class TestDecoderWithMask:
    """Test decoder with center-masking."""

    def test_mask_zeros_center(self, sample_image: torch.Tensor) -> None:
        """Passing mask zeros out center pixels before CNN."""
        decoder = Decoder(num_bits=96)
        fw = 16
        B, _, H, W = sample_image.shape
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, fw:H - fw, fw:W - fw] = 1.0

        # With mask, output should differ from without
        out_no_mask = decoder(sample_image)
        out_with_mask = decoder(sample_image, mask=mask)
        assert not torch.allclose(out_no_mask, out_with_mask, atol=1e-4)

    def test_mask_gradient_flow(self, sample_image: torch.Tensor) -> None:
        """Gradients flow through masked decoder."""
        decoder = Decoder(num_bits=96)
        fw = 16
        B, _, H, W = sample_image.shape
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, fw:H - fw, fw:W - fw] = 1.0

        sample_image.requires_grad_(True)
        result = decoder(sample_image, mask=mask)
        loss = result.sum()
        loss.backward()
        assert sample_image.grad is not None

    def test_output_shape_with_mask(self, sample_image: torch.Tensor) -> None:
        """Output shape unchanged with mask."""
        decoder = Decoder(num_bits=96)
        mask = torch.zeros(2, 1, 400, 400)
        mask[:, :, 16:384, 16:384] = 1.0
        result = decoder(sample_image, mask=mask)
        assert result.shape == (2, 96)


class TestSTNRegularization:
    """Test STN scale regularization."""

    def test_stn_scale_reg_at_identity(self) -> None:
        """STN reg loss is zero at identity initialization."""
        decoder = Decoder(num_bits=96)
        reg = decoder.stn_scale_reg()
        assert reg.item() < 1e-6, "STN reg should be ~0 at identity init"

    def test_stn_scale_reg_increases_with_deviation(self) -> None:
        """STN reg loss increases when params deviate from identity."""
        decoder = Decoder(num_bits=96)
        with torch.no_grad():
            decoder.stn_fc_bias.copy_(torch.tensor([1.5, 0.2, 0.1, -0.1, 1.3, 0.05]))
        reg = decoder.stn_scale_reg()
        assert reg.item() > 0.01, "STN reg should increase with deviation"


class TestSTNTrainability:
    """Test STN parameter trainability."""

    def test_stn_parameters_trainable_by_default(self) -> None:
        """STN parameters are trainable by default."""
        decoder = Decoder(num_bits=96)
        assert decoder.stn_fc_weight.requires_grad is True
        assert decoder.stn_fc_bias.requires_grad is True

    def test_freeze_stn_linear_option(self) -> None:
        """Can freeze STN parameters at init."""
        decoder = Decoder(num_bits=96, freeze_stn_linear=True)
        assert decoder.stn_fc_weight.requires_grad is False
        assert decoder.stn_fc_bias.requires_grad is False

    def test_unfreeze_stn_linear_method(self) -> None:
        """Can unfreeze STN parameters after init."""
        decoder = Decoder(num_bits=96, freeze_stn_linear=True)
        decoder.unfreeze_stn_linear()
        assert decoder.stn_fc_weight.requires_grad is True
        assert decoder.stn_fc_bias.requires_grad is True

    def test_freeze_stn_linear_method(self) -> None:
        """Can freeze STN parameters after init."""
        decoder = Decoder(num_bits=96)
        decoder.freeze_stn_linear()
        assert decoder.stn_fc_weight.requires_grad is False
        assert decoder.stn_fc_bias.requires_grad is False
