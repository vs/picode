"""Shared fixtures for Picodeine model tests."""

import pytest
import torch


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Random 512x512 RGB image batch."""
    return torch.rand(2, 3, 512, 512)


@pytest.fixture
def sample_message() -> torch.Tensor:
    """Random 127-bit message batch (BCH(127,64) codeword)."""
    return torch.randint(0, 2, (2, 127)).float()


@pytest.fixture
def device() -> torch.device:
    """Use CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
