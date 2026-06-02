"""Tests for the PicodeFrame decoder module."""

import torch

from picode.models.picodeframe.decoder import Decoder


class TestDecoderArchitecture:
    """Test decoder architecture."""

    def test_no_batchnorm(self) -> None:
        """Decoder has no BatchNorm layers."""
        import torch.nn as nn
        decoder = Decoder(num_bits=127)
        for module in decoder.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), \
                f"Found BatchNorm: {module}"

    def test_has_stn(self) -> None:
        """Decoder has Spatial Transformer Network."""
        decoder = Decoder(num_bits=127)
        assert hasattr(decoder, "stn_params")
        assert hasattr(decoder, "stn_fc_weight")
        assert hasattr(decoder, "stn_fc_bias")

    def test_stn_identity_init(self) -> None:
        """STN is initialized to identity transform."""
        decoder = Decoder(num_bits=127)
        expected_bias = torch.tensor([1., 0., 0., 0., 1., 0.])
        assert torch.allclose(decoder.stn_fc_bias, expected_bias)
        assert torch.allclose(decoder.stn_fc_weight, torch.zeros(128, 6))

    def test_num_bits_attribute(self) -> None:
        """Decoder stores num_bits as 127."""
        decoder = Decoder(num_bits=127)
        assert decoder.num_bits == 127

    def test_deep_cnn_goes_to_13x13(self) -> None:
        """Decoder CNN goes to 13x13 (5 stride-2 convolutions)."""
        decoder = Decoder(num_bits=127)
        # Feed a 400x400 image through just the CNN (4 channels: RGB + border mask)
        x = torch.randn(1, 4, 400, 400)
        features = decoder.decoder_cnn(x)
        assert features.shape == (1, 128, 13, 13)

    def test_fc_head_dimensions(self) -> None:
        """FC head: 21888 (21632 CNN + 256 border) → 512 → 127."""
        decoder = Decoder(num_bits=127)
        assert decoder.fc1.in_features == 128 * 13 * 13 + 256  # 21888
        assert decoder.fc1.out_features == 512
        assert decoder.fc2.in_features == 512
        assert decoder.fc2.out_features == 127

    def test_has_border_pooling_branch(self) -> None:
        """Decoder has border-pooling FC branch."""
        decoder = Decoder(num_bits=127)
        assert hasattr(decoder, "border_fc")
        # border_fc input: 3 channels × 4 strips × 400 pixels = 4800
        assert decoder.border_fc[0].in_features == 3 * 4 * 400
        assert decoder.border_fc[0].out_features == 256


class TestDecoderForward:
    """Test decoder forward pass."""

    def test_output_shape(self, sample_image: torch.Tensor) -> None:
        """Decoder outputs (B, 127) tensor."""
        decoder = Decoder(num_bits=127)
        result = decoder(sample_image, frame_width=32)
        assert result.shape == (2, 127)

    def test_output_shape_no_frame_width(self, sample_image: torch.Tensor) -> None:
        """Decoder works with default frame_width."""
        decoder = Decoder(num_bits=127)
        result = decoder(sample_image)
        assert result.shape == (2, 127)

    def test_output_is_logits(self, sample_image: torch.Tensor) -> None:
        """Output is logits (unbounded)."""
        decoder = Decoder(num_bits=127)
        result = decoder(sample_image)
        assert result.min() < 0.5 or result.max() > 0.5

    def test_gradient_flow(self, sample_image: torch.Tensor) -> None:
        """Gradients flow through decoder."""
        decoder = Decoder(num_bits=127)
        sample_image.requires_grad_(True)
        result = decoder(sample_image, frame_width=32)
        loss = result.sum()
        loss.backward()
        assert sample_image.grad is not None

    def test_decode_method(self, sample_image: torch.Tensor) -> None:
        """Decode method returns binary bits."""
        decoder = Decoder(num_bits=127)
        result = decoder.decode(sample_image, frame_width=32)
        assert result.shape == (2, 127)
        assert torch.all((result == 0) | (result == 1))


class TestBorderPooling:
    """Test the border-pooling branch."""

    def test_different_frame_widths_produce_different_output(self) -> None:
        """Different frame widths change border-pooling features."""
        decoder = Decoder(num_bits=127)
        decoder.eval()
        image = torch.rand(1, 3, 400, 400)

        out_narrow = decoder(image, frame_width=16)
        out_wide = decoder(image, frame_width=64)
        assert not torch.allclose(out_narrow, out_wide, atol=1e-4)

    def test_border_strip_pooling_shape(self) -> None:
        """Strip pooling produces correct feature dimensions."""
        decoder = Decoder(num_bits=127, height=400)
        image = torch.rand(2, 3, 400, 400)
        features = decoder._pool_border_strips(image, frame_width=32)
        # 3 channels × 4 strips × 400 pixels = 4800
        assert features.shape == (2, 4800)


class TestDecoderWithMask:
    """Test decoder with masked average pooling."""

    def test_mask_changes_output(self, sample_image: torch.Tensor) -> None:
        """Passing mask produces different output than no mask."""
        decoder = Decoder(num_bits=127)
        fw = 16
        B, _, H, W = sample_image.shape
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, fw:H - fw, fw:W - fw] = 1.0

        out_no_mask = decoder(sample_image, frame_width=fw)
        out_with_mask = decoder(sample_image, mask=mask, frame_width=fw)
        assert not torch.allclose(out_no_mask, out_with_mask, atol=1e-4)

    def test_mask_gradient_flow(self, sample_image: torch.Tensor) -> None:
        """Gradients flow through masked decoder."""
        decoder = Decoder(num_bits=127)
        fw = 16
        B, _, H, W = sample_image.shape
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, fw:H - fw, fw:W - fw] = 1.0

        sample_image.requires_grad_(True)
        result = decoder(sample_image, mask=mask, frame_width=fw)
        loss = result.sum()
        loss.backward()
        assert sample_image.grad is not None

    def test_output_shape_with_mask(self, sample_image: torch.Tensor) -> None:
        """Output shape unchanged with mask."""
        decoder = Decoder(num_bits=127)
        mask = torch.zeros(2, 1, 400, 400)
        mask[:, :, 16:384, 16:384] = 1.0
        result = decoder(sample_image, mask=mask, frame_width=16)
        assert result.shape == (2, 127)

    def test_mask_channel_affects_output(self) -> None:
        """Different masks produce different outputs (CNN uses border indicator)."""
        decoder = Decoder(num_bits=127)
        decoder.eval()

        B = 1
        image = torch.rand(B, 3, 400, 400)

        # Narrow border mask
        mask_narrow = torch.zeros(B, 1, 400, 400)
        mask_narrow[:, :, 10:390, 10:390] = 1.0

        # Wide border mask
        mask_wide = torch.zeros(B, 1, 400, 400)
        mask_wide[:, :, 80:320, 80:320] = 1.0

        out_narrow = decoder(image, mask=mask_narrow, frame_width=10)
        out_wide = decoder(image, mask=mask_wide, frame_width=80)

        # Different masks → different border indicator channels → different outputs
        assert not torch.allclose(out_narrow, out_wide, atol=1e-4)


class TestSTNRegularization:
    """Test STN scale regularization."""

    def test_stn_scale_reg_at_identity(self) -> None:
        """STN reg loss is zero at identity initialization."""
        decoder = Decoder(num_bits=127)
        reg = decoder.stn_scale_reg()
        assert reg.item() < 1e-6, "STN reg should be ~0 at identity init"

    def test_stn_scale_reg_increases_with_deviation(self) -> None:
        """STN reg loss increases when params deviate from identity."""
        decoder = Decoder(num_bits=127)
        with torch.no_grad():
            decoder.stn_fc_bias.copy_(torch.tensor([1.5, 0.2, 0.1, -0.1, 1.3, 0.05]))
        reg = decoder.stn_scale_reg()
        assert reg.item() > 0.01, "STN reg should increase with deviation"


class TestSTNTrainability:
    """Test STN parameter trainability."""

    def test_stn_parameters_trainable_by_default(self) -> None:
        """STN parameters are trainable by default."""
        decoder = Decoder(num_bits=127)
        assert decoder.stn_fc_weight.requires_grad is True
        assert decoder.stn_fc_bias.requires_grad is True

    def test_freeze_stn_linear_option(self) -> None:
        """Can freeze STN parameters at init."""
        decoder = Decoder(num_bits=127, freeze_stn_linear=True)
        assert decoder.stn_fc_weight.requires_grad is False
        assert decoder.stn_fc_bias.requires_grad is False

    def test_unfreeze_stn_linear_method(self) -> None:
        """Can unfreeze STN parameters after init."""
        decoder = Decoder(num_bits=127, freeze_stn_linear=True)
        decoder.unfreeze_stn_linear()
        assert decoder.stn_fc_weight.requires_grad is True
        assert decoder.stn_fc_bias.requires_grad is True

    def test_freeze_stn_linear_method(self) -> None:
        """Can freeze STN parameters after init."""
        decoder = Decoder(num_bits=127)
        decoder.freeze_stn_linear()
        assert decoder.stn_fc_weight.requires_grad is False
        assert decoder.stn_fc_bias.requires_grad is False
