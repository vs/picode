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
        """Build index mapping global idx -> (shard_idx, local_idx)."""
        self.index: list[tuple[int, int]] = []
        self.shards: dict[int, list] = {}  # LRU cache for loaded shards
        self._max_cached_shards = 8  # Keep at most 8 shards in memory

        num_shards = self.metadata["num_shards"]
        shard_size = self.metadata.get("shard_size")

        if shard_size is None:
            # Legacy: no shard_size in metadata, measure from first shard
            first_shard = self.data_dir / "shard_00000.pt"
            if first_shard.exists():
                data = torch.load(first_shard, weights_only=False)
                shard_size = len(data)
                self.shards[0] = data
            else:
                shard_size = 500  # fallback

        for shard_idx in range(num_shards):
            shard_path = self.data_dir / f"shard_{shard_idx:05d}.pt"
            if shard_path.exists():
                if shard_idx < num_shards - 1:
                    count = shard_size
                else:
                    # Last shard may be smaller — load to check size
                    if shard_idx not in self.shards:
                        shard_data = torch.load(shard_path, weights_only=False)
                        self.shards[shard_idx] = shard_data
                    count = len(self.shards[shard_idx])
                for local_idx in range(count):
                    self.index.append((shard_idx, local_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        shard_idx, local_idx = self.index[idx]

        # Load shard if not cached
        if shard_idx not in self.shards:
            # Evict oldest shard if cache is full
            if len(self.shards) >= self._max_cached_shards:
                oldest = next(iter(self.shards))
                del self.shards[oldest]
            shard_path = self.data_dir / f"shard_{shard_idx:05d}.pt"
            self.shards[shard_idx] = torch.load(shard_path, weights_only=False)

        sample = self.shards[shard_idx][local_idx]

        # Convert to tensors if needed
        result = {
            "image": sample["image"],
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
