"""Shared test fixtures for ECC tests."""

import pytest
import torch
from torch import Tensor


@pytest.fixture
def sample_message() -> Tensor:
    """Create a sample binary message tensor (4, 100)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (4, 100), dtype=torch.float32)


@pytest.fixture
def batch_sizes() -> list[int]:
    """Common batch sizes to test."""
    return [1, 4, 16]
