"""Fixtures for distortion backend benchmarks."""

import pytest
import torch


@pytest.fixture(params=["native", "kornia"])
def backend(request: pytest.FixtureRequest) -> str:
    """Parameterized backend fixture."""
    return request.param


@pytest.fixture(params=[1, 8])
def batch_size(request: pytest.FixtureRequest) -> int:
    """Parameterized batch size fixture."""
    return request.param


@pytest.fixture
def device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def benchmark_image(batch_size: int, device: torch.device) -> torch.Tensor:
    """Create benchmark image tensor at training size (400x400)."""
    return torch.rand(batch_size, 3, 400, 400, device=device)
