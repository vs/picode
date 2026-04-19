"""Tests for RandomBlurKernel."""

import pytest
import torch

from picode.distortions.native.blur import RandomBlurKernel


class TestRandomBlurKernel:
    """Tests for random blur kernel matching StegaStamp."""

    @pytest.fixture
    def blur(self) -> RandomBlurKernel:
        """Create RandomBlurKernel with default StegaStamp parameters."""
        return RandomBlurKernel(probs=(0.25, 0.25), kernel_size=7)

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Create sample image tensor."""
        torch.manual_seed(42)
        return torch.rand(2, 3, 64, 64)

    def test_output_shape(self, blur: RandomBlurKernel, sample_image: torch.Tensor) -> None:
        """Output shape should match input."""
        output = blur(sample_image)
        assert output.shape == sample_image.shape

    def test_output_range(self, blur: RandomBlurKernel, sample_image: torch.Tensor) -> None:
        """Output should be in valid range [0, 1]."""
        output = blur(sample_image)
        assert output.min() >= 0
        assert output.max() <= 1

    def test_gradient_flow(self, blur: RandomBlurKernel, sample_image: torch.Tensor) -> None:
        """Gradients should flow through blur."""
        sample_image.requires_grad_(True)
        output = blur(sample_image)
        output.sum().backward()

        assert sample_image.grad is not None
        assert not torch.all(sample_image.grad == 0)

    def test_identity_probability(self) -> None:
        """With probs=(0, 0), should always return identity (no blur)."""
        blur = RandomBlurKernel(probs=(0.0, 0.0), kernel_size=7)
        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)

        output = blur(image)

        assert torch.allclose(output, image, atol=1e-5)

    def test_deterministic_gaussian(self) -> None:
        """With probs=(1, 0), should always apply Gaussian blur."""
        blur = RandomBlurKernel(probs=(1.0, 0.0), kernel_size=7)
        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)

        output = blur(image)

        # Gaussian blur reduces high-frequency content
        # Check that output is smoother (lower variance in gradients)
        input_grad = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
        output_grad = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]).mean()
        assert output_grad < input_grad

    def test_deterministic_line(self) -> None:
        """With probs=(0, 1), should always apply line/motion blur."""
        blur = RandomBlurKernel(probs=(0.0, 1.0), kernel_size=7)
        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)

        output = blur(image)

        # Line blur should change the image
        assert not torch.allclose(output, image, atol=1e-3)

    def test_batch_processing(self, blur: RandomBlurKernel) -> None:
        """Should handle batch of images correctly."""
        torch.manual_seed(42)
        batch = torch.rand(4, 3, 64, 64)
        output = blur(batch)

        assert output.shape == batch.shape
        assert output.min() >= 0
        assert output.max() <= 1

    def test_different_dtypes(self, blur: RandomBlurKernel) -> None:
        """Should work with different tensor dtypes."""
        torch.manual_seed(42)
        image_float32 = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        image_float64 = torch.rand(1, 3, 64, 64, dtype=torch.float64)

        output32 = blur(image_float32)
        output64 = blur(image_float64)

        assert output32.dtype == torch.float32
        assert output64.dtype == torch.float64

    def test_custom_parameters(self) -> None:
        """Should respect custom parameters."""
        blur = RandomBlurKernel(
            probs=(0.3, 0.4),  # 30% Gaussian, 40% line, 30% identity
            kernel_size=9,
            sigma_range_gauss=(0.5, 2.0),
            sigma_range_line=(0.1, 0.5),
            min_line_width=5,
        )
        assert blur.probs == (0.3, 0.4)
        assert blur.kernel_size == 9
        assert blur.sigma_range_gauss == (0.5, 2.0)
        assert blur.sigma_range_line == (0.1, 0.5)
        assert blur.min_line_width == 5

        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)
        output = blur(image)
        assert output.shape == image.shape

    def test_kernel_normalization(self) -> None:
        """Kernels should be normalized (sum to 1)."""
        blur = RandomBlurKernel(probs=(0.25, 0.25), kernel_size=7)

        # Test identity kernel
        identity = blur._identity_kernel(torch.device("cpu"), torch.float32)
        assert torch.allclose(identity.sum(), torch.tensor(1.0), atol=1e-6)

        # Test Gaussian kernel
        gaussian = blur._gaussian_kernel(torch.device("cpu"), torch.float32)
        assert torch.allclose(gaussian.sum(), torch.tensor(1.0), atol=1e-6)

        # Test line kernel
        line = blur._line_kernel(torch.device("cpu"), torch.float32)
        assert torch.allclose(line.sum(), torch.tensor(1.0), atol=1e-6)

    def test_stochastic_behavior(self) -> None:
        """Multiple calls should produce different results (stochastic)."""
        blur = RandomBlurKernel(probs=(0.33, 0.33), kernel_size=7)
        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)

        # Run multiple times and collect results
        outputs = []
        for _ in range(10):
            outputs.append(blur(image.clone()))

        # At least some outputs should be different
        # (with 33% identity, 33% Gaussian, 33% line, very unlikely all same)
        all_same = all(torch.allclose(outputs[0], o, atol=1e-6) for o in outputs[1:])
        assert not all_same, "Expected stochastic behavior but all outputs were identical"
