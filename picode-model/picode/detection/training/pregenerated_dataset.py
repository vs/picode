# picode/detection/training/pregenerated_dataset.py
"""Pre-generated dataset for FastDetector training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class PregeneratedDetectionDataset(Dataset):
    """Dataset that loads pre-generated detection training samples.

    Expects a directory with:
        - shard_00000.pt, shard_00001.pt, ... (list of sample dicts)
        - metadata.json (dataset info)

    Each sample dict contains:
        - image: (3, H, W) tensor
        - is_watermark: float (0 or 1)
        - corners: (8,) tensor
        - has_corners: float (0 or 1)

    Args:
        data_dir: Directory containing shard files and metadata.json
        transform: Optional transform to apply to samples

    Example:
        >>> dataset = PregeneratedDetectionDataset("data/detection_dataset")
        >>> sample = dataset[0]
        >>> sample["image"].shape
        torch.Size([3, 320, 320])
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: callable | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Index all samples across shards
        self._build_index()

    def _build_index(self) -> None:
        """Load all shards into a flat sample list for fast random access."""
        self.samples: list[dict] = []

        num_shards = self.metadata["num_shards"]
        for shard_idx in range(num_shards):
            shard_path = self.data_dir / f"shard_{shard_idx:05d}.pt"
            if shard_path.exists():
                shard_data = torch.load(shard_path, weights_only=False)
                self.samples.extend(shard_data)
                if (shard_idx + 1) % 50 == 0:
                    print(f"  Loaded {shard_idx + 1}/{num_shards} shards "
                          f"({len(self.samples)} samples)")
        print(f"  Loaded all {num_shards} shards ({len(self.samples)} samples)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]

        # Convert to tensors; uint8 images → float32 [0, 1]
        image = sample["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        result = {
            "image": image,
            "is_watermark": torch.tensor(sample["is_watermark"]),
            "corners": sample["corners"],
            "has_corners": torch.tensor(sample["has_corners"]),
        }

        if self.transform is not None:
            result = self.transform(result)

        return result

    @property
    def num_samples(self) -> int:
        return self.metadata.get("num_samples", len(self))

    @property
    def input_size(self) -> int:
        return self.metadata.get("input_size", 320)
