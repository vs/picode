"""Shared test fixtures for training tests."""

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def test_images_dir(tmp_path: Path) -> Path:
    """Create a directory with test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        img = Image.new("RGB", (400, 400), color=(i * 20, i * 15, i * 10))
        img.save(img_dir / f"img_{i:03d}.jpg")
    return img_dir
