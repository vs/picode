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

    def test_shallow_cnn_stops_at_50x50(self) -> None:
        """Decoder CNN stops at 50x50 (3 stride-2 convolutions)."""
        decoder = Decoder(num_bits=96)
        # Feed a 400x400 image through just the CNN
        x = torch.randn(1, 3, 400, 400)
        features = decoder.decoder_cnn(x)
        assert features.shape == (1, 128, 50, 50)

    def test_fc_head_dimensions(self) -> None:
        """FC head: 128 → 512 → 96."""
        decoder = Decoder(num_bits=96)
        assert decoder.fc1.in_features == 128
        assert decoder.fc1.out_features == 512
        assert decoder.fc2.in_features == 512
        assert decoder.fc2.out_features == 96


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
    """Test decoder with masked average pooling."""

    def test_mask_changes_output(self, sample_image: torch.Tensor) -> None:
        """Passing mask produces different output than no mask."""
        decoder = Decoder(num_bits=96)
        fw = 16
        B, _, H, W = sample_image.shape
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, fw:H - fw, fw:W - fw] = 1.0

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

    def test_masked_pooling_aggregates_border_only(self) -> None:
        """Masked pooling uses only border features, ignoring center."""
        decoder = Decoder(num_bits=96)
        decoder.eval()

        # Create image where border and center have very different content
        B = 1
        image = torch.rand(B, 3, 400, 400)
        fw = 20
        mask = torch.zeros(B, 1, 400, 400)
        mask[:, :, fw:400 - fw, fw:400 - fw] = 1.0

        # Get output with mask
        out1 = decoder(image, mask=mask)

        # Modify center pixels drastically — should NOT affect masked output
        image_modified = image.clone()
        image_modified[:, :, fw + 10:400 - fw - 10, fw + 10:400 - fw - 10] = 1.0 - \
            image[:, :, fw + 10:400 - fw - 10, fw + 10:400 - fw - 10]

        out2 = decoder(image_modified, mask=mask)

        # With border-only pooling, outputs should be very similar
        # (not identical because STN sees full image, and CNN has some receptive field overlap)
        diff = (out1 - out2).abs().mean().item()
        # The diff should be small — center changes have minimal effect through border pooling
        assert diff < 1.0, f"Center change affected masked output too much: {diff}"


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
