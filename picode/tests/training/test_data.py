"""Tests for data loading."""

from pathlib import Path

import torch
from PIL import Image

from picode.training.config import DataConfig
from picode.training.data import FolderDataset, create_dataloader


def _create_test_images(path: Path, count: int = 5) -> None:
    """Create test images in a directory."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 30, i * 20))
        img.save(path / f"img_{i}.jpg")


class TestFolderDataset:
    def test_finds_images(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path, count=5)
        dataset = FolderDataset(str(tmp_path), image_size=64)
        assert len(dataset) == 5

    def test_returns_tensor(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path, count=1)
        dataset = FolderDataset(str(tmp_path), image_size=64)
        img = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_image_size(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path, count=1)
        dataset = FolderDataset(str(tmp_path), image_size=128)
        assert dataset[0].shape == (3, 128, 128)


class TestCreateDataloader:
    def test_creates_dataloader(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path, count=8)
        cfg = DataConfig(source="folder", path=str(tmp_path), batch_size=4, num_workers=0)
        loader = create_dataloader(cfg, image_size=64)

        batch = next(iter(loader))
        assert batch.shape == (4, 3, 64, 64)
