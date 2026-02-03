"""Shared test fixtures."""

import pytest
import torch
from torch import Tensor


@pytest.fixture
def sample_image() -> Tensor:
    """Create a sample image tensor (1, 3, 64, 64) in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 64, 64)


@pytest.fixture
def batch_images() -> Tensor:
    """Create a batch of images (4, 3, 64, 64) in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(4, 3, 64, 64)


@pytest.fixture
def device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
