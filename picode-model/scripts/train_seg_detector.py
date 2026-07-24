#!/usr/bin/env python
"""Train SegDetectorModel on pre-generated segmentation shards.

Usage:
    python scripts/train_seg_detector.py \\
        --data-dir data/seg_detection_b72s20m85 \\
        --output-dir checkpoints/seg_detector \\
        --epochs 50 --batch-size 16

    # Multiple shard directories (concatenated):
    python scripts/train_seg_detector.py \\
        --data-dir data/seg_b72 data/seg_b96 \\
        --output-dir checkpoints/seg_detector
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from picode.detection.seg_detector import SegDetectorModel
from picode.detection.training.seg_loss import SegDetectionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegDetectorModel on segmentation shards")
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        required=True,
        help="Shard directory or directories (multiple are concatenated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/seg_detector",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save periodic checkpoint every N epochs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping max norm",
    )
    parser.add_argument(
        "--pos-weight", type=float, default=2.0,
        help="Positive class weight for BCE loss",
    )
    return parser.parse_args()


class SegDetectionDataset(Dataset):
    """Dataset that loads pre-generated segmentation training shards.

    Expects each shard directory to contain:
        - shard_00000.pt, shard_00001.pt, ... (list of sample dicts)
        - metadata.json (dataset info)

    Each sample dict contains:
        - image: (3, H, W) uint8 tensor
        - mask: (1, H, W) uint8 tensor — binary GT segmentation mask
        - is_watermark: float (0.0 or 1.0)
        - model_label: str (optional)

    Args:
        data_dir: Directory containing shard files and metadata.json.

    Example:
        >>> ds = SegDetectionDataset("data/seg_detection_b72s20m85")
        >>> sample = ds[0]
        >>> sample["image"].shape
        torch.Size([3, 320, 320])
        >>> sample["mask"].shape
        torch.Size([1, 320, 320])
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self._load_shards()

    def _load_shards(self) -> None:
        """Load all shards into a flat sample list."""
        self.samples: list[dict] = []
        num_shards = self.metadata["num_shards"]
        for shard_idx in range(num_shards):
            shard_path = self.data_dir / f"shard_{shard_idx:05d}.pt"
            if shard_path.exists():
                shard_data = torch.load(shard_path, weights_only=False)
                self.samples.extend(shard_data)
                if (shard_idx + 1) % 50 == 0:
                    print(
                        f"  Loaded {shard_idx + 1}/{num_shards} shards "
                        f"({len(self.samples)} samples)"
                    )
        print(f"  Loaded all {num_shards} shards ({len(self.samples)} samples)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str | float]:
        sample = self.samples[idx]

        # uint8 [0,255] → float32 [0,1]
        image = sample["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        mask = sample["mask"]
        if mask.dtype == torch.uint8:
            mask = mask.float() / 255.0

        result: dict[str, Tensor | str | float] = {
            "image": image,
            "mask": mask,
            "is_watermark": torch.tensor(float(sample["is_watermark"])),
        }
        if "model_label" in sample:
            result["model_label"] = sample["model_label"]
        return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SegDetectionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one training epoch.

    Args:
        model: SegDetectorModel.
        loader: Training DataLoader.
        loss_fn: SegDetectionLoss instance.
        optimizer: AdamW optimizer.
        device: Compute device.
        grad_clip: Max gradient norm.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images: Tensor = batch["image"].to(device)
        masks: Tensor = batch["mask"].to(device)

        optimizer.zero_grad()
        pred = model(images)
        losses = loss_fn(pred, masks)
        losses["total"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += losses["total"].item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SegDetectionLoss,
    device: torch.device,
) -> float:
    """Run one validation epoch.

    Args:
        model: SegDetectorModel.
        loader: Validation DataLoader.
        loss_fn: SegDetectionLoss instance.
        device: Compute device.

    Returns:
        Mean validation loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images: Tensor = batch["image"].to(device)
        masks: Tensor = batch["mask"].to(device)
        pred = model(images)
        losses = loss_fn(pred, masks)
        total_loss += losses["total"].item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: SegDetectorModel,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    """Save model checkpoint.

    Args:
        model: Trained SegDetectorModel.
        epoch: Current epoch number.
        val_loss: Validation loss at this epoch.
        path: Destination file path.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"input_size": model.input_size},
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset(s)
    sub_datasets: list[Dataset] = []
    for data_dir in args.data_dir:
        print(f"Loading shard dataset from {data_dir}...")
        ds = SegDetectionDataset(data_dir=data_dir)
        print(f"  {data_dir}: {len(ds)} samples")
        sub_datasets.append(ds)

    if len(sub_datasets) == 1:
        full_dataset: Dataset = sub_datasets[0]
    else:
        full_dataset = ConcatDataset(sub_datasets)
    print(f"Total samples: {len(full_dataset)}")

    # Train / val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Model, loss, optimizer, scheduler
    print("Creating SegDetectorModel (MobileNetV3-Small, input_size=320)...")
    model = SegDetectorModel(input_size=320, pretrained=True).to(device)

    loss_fn = SegDetectionLoss(pos_weight=args.pos_weight)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints will be saved to {output_dir}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, args.grad_clip)
        v_loss = val_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, "
            f"val_loss: {v_loss:.4f}, "
            f"lr: {lr:.2e}"
        )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(model, epoch, v_loss, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_path = output_dir / "best.pt"
            save_checkpoint(model, epoch, v_loss, best_path)
            print(f"  New best model (val_loss={v_loss:.4f})")

    # Final checkpoint
    final_path = output_dir / "final.pt"
    save_checkpoint(model, args.epochs, best_val_loss, final_path)
    print(f"\nTraining complete. Final checkpoint saved to {final_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
