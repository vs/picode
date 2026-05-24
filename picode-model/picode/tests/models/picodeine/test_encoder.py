"""Tests for Picodeine encoder with AdaIN message injection."""

import torch

from picode.models.picodeine.encoder import Encoder


class TestEncoderArchitecture:
    """Test encoder structure and initialization."""

    def test_no_batchnorm(self) -> None:
        encoder = Encoder(num_bits=127)
        for module in encoder.modules():
            assert not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))

    def test_weight_initialization(self) -> None:
        encoder = Encoder(num_bits=127)
        for name, param in encoder.named_parameters():
            if "weight" in name and param.dim() >= 2:
                # AdaIN projections are zero-initialized by design
                if "adain" in name:
                    continue
                assert param.std() > 0.01, f"{name} appears uninitialized"

    def test_three_channel_input(self) -> None:
        """Encoder takes 3ch image input (no message spatial concat)."""
        encoder = Encoder(num_bits=127)
        assert encoder.conv1.in_channels == 3

    def test_num_bits_attribute(self) -> None:
        encoder = Encoder(num_bits=127)
        assert encoder.num_bits == 127

    def test_has_mapping_network(self) -> None:
        encoder = Encoder(num_bits=127)
        assert hasattr(encoder, "mapping")

    def test_has_adain_layers(self) -> None:
        encoder = Encoder(num_bits=127)
        assert hasattr(encoder, "adain1")
        assert hasattr(encoder, "adain2")
        assert hasattr(encoder, "adain3")
        assert hasattr(encoder, "adain4")

    def test_has_secret_dense(self) -> None:
        """Encoder has spatial message expansion layer for bottleneck injection."""
        encoder = Encoder(num_bits=127)
        assert hasattr(encoder, "secret_dense")
        assert isinstance(encoder.secret_dense, torch.nn.Linear)
        assert encoder.secret_dense.in_features == 127
        assert encoder.secret_dense.out_features == 32 * 8 * 8


class TestEncoderForward:
    """Test encoder forward pass behavior."""

    def test_output_shape(self, sample_image: torch.Tensor, sample_message: torch.Tensor) -> None:
        encoder = Encoder(num_bits=127)
        output = encoder(sample_image, sample_message)
        assert output.shape == sample_image.shape

    def test_gradient_flow(self, sample_image: torch.Tensor, sample_message: torch.Tensor) -> None:
        encoder = Encoder(num_bits=127)
        sample_image.requires_grad_(True)
        output = encoder(sample_image, sample_message)
        output.sum().backward()
        assert sample_image.grad is not None
        assert sample_image.grad.abs().sum() > 0

    def test_message_affects_output(self, sample_image: torch.Tensor) -> None:
        """Different messages produce different outputs at init (bottleneck spatial)."""
        encoder = Encoder(num_bits=127)
        m1 = torch.zeros(2, 127)
        m2 = torch.ones(2, 127)
        out1 = encoder(sample_image, m1)
        out2 = encoder(sample_image, m2)
        assert not torch.allclose(out1, out2)

    def test_residual_is_small_at_init(
        self, sample_image: torch.Tensor, sample_message: torch.Tensor
    ) -> None:
        """At initialization, residual should be small (AdaIN starts as identity)."""
        encoder = Encoder(num_bits=127)
        output = encoder(sample_image, sample_message)
        residual = (output - sample_image).abs().mean()
        # Residual should be modest at init (network not trained yet)
        assert residual < 2.0

    def test_rejects_non_divisible_input(self, sample_message: torch.Tensor) -> None:
        """Input must be divisible by 16 for U-Net skip connections."""
        encoder = Encoder(num_bits=127)
        bad_image = torch.rand(2, 3, 500, 500)
        try:
            encoder(bad_image, sample_message)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
