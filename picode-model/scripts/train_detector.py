#!/usr/bin/env python
"""Train FastDetector using synthetic data from encoder.

Usage:
    python scripts/train_detector.py --encoder checkpoints/best.pt --data-dir data/coco/val2017
    python scripts/train_detector.py --encoder checkpoints/best.pt --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from picode.detection.fast_detector import FastDetectorModel
from picode.detection.training import (
    DetectionAugmentation,
    DetectionDataset,
    DetectionLoss,
    DetectionTrainer,
)
from picode.models.stegastamp import Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FastDetector")
    parser.add_argument(
        "--encoder",
        type=str,
        default="checkpoints/best.pt",
        help="Path to encoder checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/coco/val2017",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/detection",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_encoder(checkpoint_path: str, device: torch.device) -> Encoder:
    """Load trained encoder from checkpoint."""
    print(f"Loading encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get num_bits from config if available
    num_bits = 100
    if "config" in ckpt:
        config = ckpt["config"]
        if isinstance(config, dict) and "training" in config:
            num_bits = config["training"].get("num_bits", 100)

    encoder = Encoder(num_bits=num_bits)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device)
    encoder.eval()
    print(f"Encoder loaded (num_bits={num_bits})")
    return encoder


def create_datasets(
    encoder: Encoder,
    data_dir: str,
    val_split: float,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Create train and validation datasets."""
    print(f"Creating datasets from {data_dir}...")

    augmentation = DetectionAugmentation(
        photometric_p=0.5,
        geometric_p=0.5,
    )

    # Full dataset
    full_dataset = DetectionDataset(
        image_dir=data_dir,
        encoder=encoder,
        num_bits=encoder.num_bits,
        positive_ratio=0.5,
        input_size=320,
        perspective_strength=(0.05, 0.20),
        transform=augmentation,
    )

    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def main() -> None:
    args = parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load encoder
    encoder = load_encoder(args.encoder, device)

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        encoder=encoder,
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print("Creating FastDetectorModel...")
    model = FastDetectorModel(input_size=320, pretrained=True)

    # Create loss function
    loss_fn = DetectionLoss(
        cls_weight=1.0,
        corner_weight=5.0,
        conf_weight=0.5,
    )

    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=str(device),
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints will be saved to {args.output_dir}")

    history = trainer.fit(
        num_epochs=args.epochs,
        save_dir=args.output_dir,
        save_every=args.save_every,
    )

    # Save final checkpoint
    final_path = Path(args.output_dir) / "final.pt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete. Final checkpoint saved to {final_path}")

    # Print final metrics
    if history:
        final_metrics = history[-1]
        print(f"Final train loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Final val loss: {final_metrics.get('val_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
