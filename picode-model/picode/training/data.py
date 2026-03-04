"""Data loading utilities for training."""

from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from picode.training.config import DataConfig


class ImageDataset(Protocol):
    """Protocol for image datasets."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tensor:
        ...


class FolderDataset(Dataset[Tensor]):
    """Load images from a directory."""

    def __init__(
        self,
        path: str,
        image_size: int = 400,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        self.image_size = image_size
        self.files = self._find_images(Path(path), extensions)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def _find_images(self, path: Path, extensions: tuple[str, ...]) -> list[Path]:
        """Find all image files in directory."""
        files: list[Path] = []
        for ext in extensions:
            files.extend(path.glob(f"**/*{ext}"))
            files.extend(path.glob(f"**/*{ext.upper()}"))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        result: Tensor = self.transform(img)
        return result


def create_dataloader(config: DataConfig, image_size: int) -> DataLoader[Tensor]:
    """Factory function to create dataloader from config."""
    if config.source == "folder":
        dataset: Dataset[Tensor] = FolderDataset(config.path, image_size)
    else:
        raise ValueError(f"Unknown data source: {config.source}")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
