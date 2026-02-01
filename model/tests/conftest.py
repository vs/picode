"""Shared test fixtures."""

import pytest
import torch


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Random 400x400 RGB image batch."""
    return torch.rand(2, 3, 400, 400)


@pytest.fixture
def sample_message() -> torch.Tensor:
    """Random 100-bit message batch."""
    return torch.randint(0, 2, (2, 100)).float()


@pytest.fixture
def device() -> torch.device:
    """Use CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
