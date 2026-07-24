"""PicoGrain test fixtures."""

import pytest
import torch
from torch import Tensor


@pytest.fixture
def sample_image() -> Tensor:
    return torch.rand(2, 3, 512, 512)


@pytest.fixture
def sample_image_256() -> Tensor:
    return torch.rand(2, 3, 256, 256)


@pytest.fixture
def sample_message() -> Tensor:
    """127-bit message (BCH codeword length)."""
    return torch.randint(0, 2, (2, 127)).float()
