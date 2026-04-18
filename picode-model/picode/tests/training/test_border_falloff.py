"""Tests for border falloff mask implementation."""

import pytest
import torch

from picode.training.trainer import (
    compute_yuv_l2_loss,
    compute_yuv_l2_loss_with_falloff,
    create_border_falloff_mask,
)


class TestCreateBorderFalloffMask:
    """Tests for create_border_falloff_mask function."""

    def test_shape(self) -> None:
        """Mask should be (1, 1, H, W) for broadcasting."""
        mask = create_border_falloff_mask(400, 400)
        assert mask.shape == (1, 1, 400, 400)

    def test_shape_rectangular(self) -> None:
        """Mask should work with non-square dimensions."""
        mask = create_border_falloff_mask(300, 500)
        assert mask.shape == (1, 1, 300, 500)

    def test_center_near_zero(self) -> None:
        """Center of mask should be near 0 (no amplification)."""
        mask = create_border_falloff_mask(400, 400)
        center_val = mask[0, 0, 200, 200].item()
        assert center_val < 0.01, f"Center value {center_val} should be near 0"

    def test_border_near_one(self) -> None:
        """Border of mask should be near 1 (max amplification)."""
        mask = create_border_falloff_mask(400, 400)
        # Top edge center
        top_val = mask[0, 0, 0, 200].item()
        assert top_val > 0.9, f"Top border value {top_val} should be near 1"
        # Bottom edge center
        bottom_val = mask[0, 0, 399, 200].item()
        assert bottom_val > 0.9, f"Bottom border value {bottom_val} should be near 1"
        # Left edge center
        left_val = mask[0, 0, 200, 0].item()
        assert left_val > 0.9, f"Left border value {left_val} should be near 1"
        # Right edge center
        right_val = mask[0, 0, 200, 399].item()
        assert right_val > 0.9, f"Right border value {right_val} should be near 1"

    def test_corner_max(self) -> None:
        """Corners should have maximum values (both borders apply)."""
        mask = create_border_falloff_mask(400, 400)
        # Top-left corner
        corner = mask[0, 0, 0, 0].item()
        center = mask[0, 0, 200, 200].item()
        assert corner > center, "Corner should have higher weight than center"

    def test_device_placement(self) -> None:
        """Mask should be created on specified device."""
        device = torch.device("cpu")
        mask = create_border_falloff_mask(100, 100, device=device)
        assert mask.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self) -> None:
        """Mask should work on CUDA device."""
        device = torch.device("cuda")
        mask = create_border_falloff_mask(100, 100, device=device)
        assert mask.device.type == "cuda"

    def test_falloff_speed(self) -> None:
        """Different falloff speeds should affect border region size."""
        # Lower falloff_speed = larger border region
        mask_fast = create_border_falloff_mask(400, 400, falloff_speed=2)
        mask_slow = create_border_falloff_mask(400, 400, falloff_speed=8)

        # Check at 25% from edge (100 pixels in)
        fast_val = mask_fast[0, 0, 100, 200].item()
        slow_val = mask_slow[0, 0, 100, 200].item()

        # Fast falloff (speed=2 = 50% border) should have higher value at this point
        assert fast_val > slow_val, "Faster falloff should have larger border region"

    def test_symmetry(self) -> None:
        """Mask should be symmetric."""
        mask = create_border_falloff_mask(400, 400)
        # Top-bottom symmetry
        assert torch.allclose(mask[0, 0, 50, :], mask[0, 0, 349, :], atol=1e-5)
        # Left-right symmetry
        assert torch.allclose(mask[0, 0, :, 50], mask[0, 0, :, 349], atol=1e-5)

    def test_value_range(self) -> None:
        """Mask values should be in [0, 1]."""
        mask = create_border_falloff_mask(400, 400)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestComputeYuvL2LossWithFalloff:
    """Tests for compute_yuv_l2_loss_with_falloff function."""

    @pytest.fixture
    def sample_images(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample original and encoded images."""
        original = torch.rand(2, 3, 400, 400)
        # Create encoded with small perturbation
        encoded = original + 0.01 * torch.randn_like(original)
        encoded = encoded.clamp(0, 1)
        return original, encoded

    @pytest.fixture
    def falloff_mask(self) -> torch.Tensor:
        """Create falloff mask."""
        return create_border_falloff_mask(400, 400)

    def test_output_is_scalar(
        self, sample_images: tuple[torch.Tensor, torch.Tensor], falloff_mask: torch.Tensor
    ) -> None:
        """Loss should be a scalar tensor."""
        original, encoded = sample_images
        loss = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 1.0, 1.0), falloff_mask, edge_gain=10.0
        )
        assert loss.dim() == 0

    def test_border_perturbation_higher_loss(self, falloff_mask: torch.Tensor) -> None:
        """Perturbation at border should have higher loss than at center."""
        original = torch.zeros(1, 3, 400, 400) + 0.5

        # Perturbation at center
        encoded_center = original.clone()
        encoded_center[0, :, 190:210, 190:210] += 0.1

        # Same perturbation at border
        encoded_border = original.clone()
        encoded_border[0, :, 0:20, 190:210] += 0.1

        loss_center = compute_yuv_l2_loss_with_falloff(
            original, encoded_center, (1.0, 1.0, 1.0), falloff_mask, edge_gain=10.0
        )
        loss_border = compute_yuv_l2_loss_with_falloff(
            original, encoded_border, (1.0, 1.0, 1.0), falloff_mask, edge_gain=10.0
        )

        assert loss_border > loss_center, (
            f"Border loss {loss_border.item():.6f} should be > "
            f"center loss {loss_center.item():.6f}"
        )

    def test_edge_gain_amplification(
        self, sample_images: tuple[torch.Tensor, torch.Tensor], falloff_mask: torch.Tensor
    ) -> None:
        """Higher edge gain should increase loss."""
        original, encoded = sample_images

        loss_low = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 1.0, 1.0), falloff_mask, edge_gain=1.0
        )
        loss_high = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 1.0, 1.0), falloff_mask, edge_gain=10.0
        )

        assert loss_high > loss_low

    def test_yuv_weights_applied(self, falloff_mask: torch.Tensor) -> None:
        """YUV weights should affect loss calculation."""
        original = torch.zeros(1, 3, 400, 400) + 0.5
        # Add color shift (affects U/V channels more than Y)
        encoded = original.clone()
        encoded[0, 0, :, :] += 0.1  # Red channel change

        # With equal weights
        loss_equal = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 1.0, 1.0), falloff_mask, edge_gain=1.0
        )
        # With high chroma weights (U=100, V=100)
        loss_chroma = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 100.0, 100.0), falloff_mask, edge_gain=1.0
        )

        # High chroma weights should give higher loss for color changes
        assert loss_chroma > loss_equal

    def test_zero_perturbation_near_zero_loss(self, falloff_mask: torch.Tensor) -> None:
        """Zero perturbation should give near-zero loss."""
        original = torch.rand(1, 3, 400, 400)
        encoded = original.clone()

        loss = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 1.0, 1.0), falloff_mask, edge_gain=10.0
        )
        assert loss.item() < 1e-6

    def test_gradient_flow(
        self, sample_images: tuple[torch.Tensor, torch.Tensor], falloff_mask: torch.Tensor
    ) -> None:
        """Gradients should flow through the loss."""
        original, encoded = sample_images
        encoded.requires_grad_(True)

        loss = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 1.0, 1.0), falloff_mask, edge_gain=10.0
        )
        loss.backward()

        assert encoded.grad is not None
        assert not torch.all(encoded.grad == 0)


class TestCompareWithWithoutFalloff:
    """Compare loss with and without border falloff."""

    def test_falloff_increases_border_penalty(self) -> None:
        """Border falloff should increase penalty for border perturbations."""
        original = torch.zeros(1, 3, 400, 400) + 0.5
        falloff_mask = create_border_falloff_mask(400, 400)

        # Uniform perturbation across image
        encoded = original + 0.05

        # Loss without falloff (standard L2)
        loss_standard = compute_yuv_l2_loss(original, encoded, (1.0, 1.0, 1.0))

        # Loss with falloff
        loss_falloff = compute_yuv_l2_loss_with_falloff(
            original, encoded, (1.0, 1.0, 1.0), falloff_mask, edge_gain=10.0
        )

        # With uniform perturbation, falloff should increase total loss
        # because border regions get amplified
        assert loss_falloff > loss_standard
