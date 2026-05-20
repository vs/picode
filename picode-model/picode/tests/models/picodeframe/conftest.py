"""Shared test fixtures for PicodeFrame tests."""

import pytest
import torch


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Random 400x400 RGB image batch."""
    return torch.rand(2, 3, 400, 400)


@pytest.fixture
def sample_message() -> torch.Tensor:
    """Random 127-bit message batch (BCH(127,64) codeword)."""
    return torch.randint(0, 2, (2, 127)).float()


@pytest.fixture
def frame_width() -> int:
    """Default frame width (4% of 400)."""
    return 16


@pytest.fixture
def padded_image(sample_image: torch.Tensor, frame_width: int) -> torch.Tensor:
    """Image with inner region extracted and reflection-padded back.

    Simulates the PicodeFrame input preparation: extract inner region,
    then reflection-pad back to original size.
    """
    fw = frame_width
    inner = sample_image[:, :, fw:-fw, fw:-fw]
    return torch.nn.functional.pad(inner, (fw, fw, fw, fw), mode="reflect")


@pytest.fixture
def device() -> torch.device:
    """Use CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
