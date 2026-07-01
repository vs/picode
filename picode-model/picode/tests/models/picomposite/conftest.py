"""PicoMposite test fixtures."""

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
def sample_message_96() -> Tensor:
    """Full 96-bit message (max tier)."""
    return torch.randint(0, 2, (2, 96)).float()


@pytest.fixture
def sample_message_16() -> Tensor:
    """16-bit message (padded to 96)."""
    msg = torch.zeros(2, 96)
    msg[:, :16] = torch.randint(0, 2, (2, 16)).float()
    return msg


@pytest.fixture
def tier_0() -> Tensor:
    """Tier 0 (16 bits) for both samples."""
    return torch.tensor([0, 0])


@pytest.fixture
def tier_3() -> Tensor:
    """Tier 3 (96 bits) for both samples."""
    return torch.tensor([3, 3])


@pytest.fixture
def mixed_tiers() -> Tensor:
    """Mixed tiers: sample 0 = T16, sample 1 = T96."""
    return torch.tensor([0, 3])
