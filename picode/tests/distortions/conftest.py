"""Shared test fixtures for distortion tests."""

import pytest
import torch
from torch import Tensor


@pytest.fixture(params=["native", "kornia"])
def backend(request):
    """Parametrized fixture for testing both backends."""
    return request.param


@pytest.fixture
def distortion_module(backend):
    """Get the appropriate distortion module based on backend."""
    if backend == "native":
        from picode.distortions import native as mod
    else:
        from picode.distortions import kornia as mod
    return mod


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
