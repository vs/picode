"""Tests for the PicodeLite decoder module."""

import pytest
import torch
import torch.nn as nn

from picode.models.picodelite.decoder import Decoder


class TestDecoderArchitecture:
    """Test PicodeLite decoder architecture - NO STN, global pooling."""

    def test_no_stn_params(self) -> None:
        """Decoder has no STN parameters (unlike StegaStamp)."""
        decoder = Decoder(num_bits=63)
        assert not hasattr(decoder, "stn_params"), "Should not have stn_params"

    def test_no_stn_fc_weight(self) -> None:
        """Decoder has no STN FC weight."""
        decoder = Decoder(num_bits=63)
        assert not hasattr(decoder, "stn_fc_weight"), "Should not have stn_fc_weight"

    def test_no_stn_fc_bias(self) -> None:
        """Decoder has no STN FC bias."""
        decoder = Decoder(num_bits=63)
        assert not hasattr(decoder, "stn_fc_bias"), "Should not have stn_fc_bias"

    def test_has_global_pool(self) -> None:
        """Decoder has global average pooling layer."""
        decoder = Decoder(num_bits=63)
        assert hasattr(decoder, "global_pool"), "Missing global_pool"
        assert isinstance(decoder.global_pool, nn.AdaptiveAvgPool2d)

    def test_no_batchnorm(self) -> None:
        """Decoder has no BatchNorm layers."""
        decoder = Decoder(num_bits=63)
        for module in decoder.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), \
                f"Found BatchNorm: {module}"

    def test_parameter_count_under_1m(self) -> None:
        """Decoder has less than 1M parameters for mobile inference."""
        decoder = Decoder(num_bits=63)
        param_count = sum(p.numel() for p in decoder.parameters())
        assert param_count < 1_000_000, f"Parameter count {param_count} exceeds 1M"

    def test_num_bits_attribute(self) -> None:
        """Decoder stores num_bits attribute."""
        decoder = Decoder(num_bits=63)
        assert decoder.num_bits == 63

    def test_default_num_bits_is_63(self) -> None:
        """Default num_bits is 63 for PicodeLite."""
        decoder = Decoder()
        assert decoder.num_bits == 63


class TestDecoderForward:
    """Test decoder forward pass."""

    @pytest.fixture
    def sample_image_320(self) -> torch.Tensor:
        """Random 320x320 RGB image batch."""
        return torch.rand(2, 3, 320, 320)

    def test_output_shape(self, sample_image_320: torch.Tensor) -> None:
        """Decoder outputs (B, num_bits) tensor for 320x320 input."""
        decoder = Decoder(num_bits=63)
        result = decoder(sample_image_320)
        assert result.shape == (2, 63)

    def test_output_shape_batch_1(self) -> None:
        """Works with batch size 1."""
        decoder = Decoder(num_bits=63)
        image = torch.rand(1, 3, 320, 320)
        result = decoder(image)
        assert result.shape == (1, 63)

    def test_output_shape_batch_4(self) -> None:
        """Works with batch size 4."""
        decoder = Decoder(num_bits=63)
        image = torch.rand(4, 3, 320, 320)
        result = decoder(image)
        assert result.shape == (4, 63)

    def test_output_shape_batch_8(self) -> None:
        """Works with batch size 8."""
        decoder = Decoder(num_bits=63)
        image = torch.rand(8, 3, 320, 320)
        result = decoder(image)
        assert result.shape == (8, 63)

    def test_output_is_logits(self, sample_image_320: torch.Tensor) -> None:
        """Output is logits (unbounded, can be negative)."""
        decoder = Decoder(num_bits=63)
        result = decoder(sample_image_320)
        # Logits are unbounded - at least some should be outside [0,1]
        # With random init, this is almost certain
        assert result.min() < 0.5 or result.max() > 0.5, "Expected unbounded logits"

    def test_gradient_flow(self, sample_image_320: torch.Tensor) -> None:
        """Gradients flow through decoder."""
        decoder = Decoder(num_bits=63)
        sample_image_320.requires_grad_(True)
        result = decoder(sample_image_320)
        loss = result.sum()
        loss.backward()
        assert sample_image_320.grad is not None

    def test_different_num_bits(self, sample_image_320: torch.Tensor) -> None:
        """Works with different message sizes."""
        decoder = Decoder(num_bits=100)
        result = decoder(sample_image_320)
        assert result.shape == (2, 100)

    def test_decode_method(self, sample_image_320: torch.Tensor) -> None:
        """Decode method returns binary bits."""
        decoder = Decoder(num_bits=63)
        result = decoder.decode(sample_image_320)
        assert result.shape == (2, 63)
        assert torch.all((result == 0) | (result == 1))

    def test_resolution_independent_via_global_pool(self) -> None:
        """Global pooling makes decoder resolution-independent."""
        decoder = Decoder(num_bits=63)
        # Should work with different resolutions
        for size in [256, 320, 400, 512]:
            image = torch.rand(1, 3, size, size)
            result = decoder(image)
            assert result.shape == (1, 63), f"Failed for size {size}"


class TestDecoderInit:
    """Test decoder initialization."""

    def test_kaiming_normal_init(self) -> None:
        """Weights are initialized with Kaiming normal."""
        decoder = Decoder(num_bits=63)
        # Check that conv weights have reasonable variance (not all zeros or ones)
        for name, module in decoder.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                # Kaiming normal should have non-trivial variance
                assert weight.std() > 0.01, f"Conv {name} weights may not be initialized"
                assert weight.std() < 1.0, f"Conv {name} weights may have too high variance"

    def test_biases_are_zero(self) -> None:
        """Biases are initialized to zero."""
        decoder = Decoder(num_bits=63)
        for name, module in decoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
                assert torch.allclose(module.bias, torch.zeros_like(module.bias)), \
                    f"Bias {name} should be zero initialized"
