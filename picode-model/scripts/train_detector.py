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
from picode.detection.training.augmentation import DomainRandomizedAugmentation
from picode.models.factory import create_encoder
from picode.training.config import ModelConfig


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
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Use domain randomization for sim-to-real transfer",
    )
    parser.add_argument(
        "--sobel-sigma",
        type=float,
        default=None,
        help="Sobel texture mask sigma (None = no mask)",
    )
    parser.add_argument(
        "--sobel-floor",
        type=float,
        default=0.85,
        help="Sobel mask floor value (default 0.85)",
    )
    parser.add_argument(
        "--strengths",
        type=str,
        default=None,
        help="Comma-separated residual strengths to sample from (e.g. '0.010,0.012,0.015,0.020')",
    )
    parser.add_argument(
        "--perspective-strength",
        type=float,
        nargs=2,
        default=[0.05, 0.20],
        metavar=("MIN", "MAX"),
        help="Perspective distortion range (default: 0.05 0.20)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping norm (None = no clipping)",
    )
    parser.add_argument(
        "--hard-negative-p",
        type=float,
        default=0.8,
        help="Probability of hard negative transform on negatives (default 0.8)",
    )
    return parser.parse_args()


def load_encoder(
    checkpoint_path: str, device: torch.device,
) -> tuple[torch.nn.Module, int, int]:
    """Load trained encoder from checkpoint (any model type).

    Returns:
        Tuple of (encoder, num_bits, encoder_input_size).
    """
    print(f"Loading encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    # Extract model config
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")
    encoder_size = model_cfg.get("encoder_size", 400)

    # Extract training config
    training_cfg = config.get("training", {})
    num_bits = training_cfg.get("num_bits", 100)

    # Compute strength for picotrust models
    residual_strength = training_cfg.get("residual_strength", 0)
    strength = None
    if residual_strength > 0:
        step = ckpt.get("step", 0)
        anneal_target = training_cfg.get("residual_strength_anneal_target", residual_strength)
        anneal_start = training_cfg.get("residual_strength_anneal_start", 0)
        anneal_steps = training_cfg.get("residual_strength_anneal_steps", 1)
        if step >= anneal_start and anneal_steps > 0:
            t = min((step - anneal_start) / anneal_steps, 1.0)
            strength = residual_strength + t * (anneal_target - residual_strength)
        else:
            strength = residual_strength

    # Detect mask usage
    loss_cfg = config.get("loss", {})
    use_mask = loss_cfg.get("mask_reg") is not None

    mc = ModelConfig(
        type=model_type,
        encoder_size=encoder_size,
        decoder_size=model_cfg.get("decoder_size", 400),
    )
    encoder = create_encoder(mc, num_bits=num_bits, strength=strength, use_mask=use_mask)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device)
    encoder.eval()
    print(f"Encoder loaded: type={model_type}, num_bits={num_bits}, size={encoder_size}")
    return encoder, num_bits, encoder_size


def create_datasets(
    encoder: torch.nn.Module,
    data_dir: str,
    num_bits: int,
    encoder_input_size: int,
    val_split: float,
    seed: int = 42,
    use_domain_randomization: bool = False,
    sobel_mask_sigma: float | None = None,
    sobel_mask_floor: float = 0.85,
    strength_values: list[float] | None = None,
    perspective_strength: tuple[float, float] = (0.05, 0.20),
    hard_negative_p: float = 0.8,
) -> tuple[Dataset, Dataset]:
    """Create train and validation datasets."""
    print(f"Creating datasets from {data_dir}...")
    if sobel_mask_sigma is not None:
        print(f"Sobel mask: sigma={sobel_mask_sigma}, floor={sobel_mask_floor}")
    if strength_values is not None:
        print(f"Strength sampling: {strength_values}")

    if use_domain_randomization:
        print("Using DomainRandomizedAugmentation for sim-to-real transfer")
        augmentation = DomainRandomizedAugmentation(
            photometric_p=0.7,
            geometric_p=0.5,
            domain_random_p=0.6,
            perspective_strength=(0.0, 0.25),
            rotation_degrees=(-15, 15),
        )
    else:
        augmentation = DetectionAugmentation(
            photometric_p=0.5,
            geometric_p=0.5,
        )

    # Full dataset
    full_dataset = DetectionDataset(
        image_dir=data_dir,
        encoder=encoder,
        num_bits=num_bits,
        positive_ratio=0.5,
        input_size=320,
        encoder_input_size=encoder_input_size,
        perspective_strength=perspective_strength,
        transform=augmentation,
        sobel_mask_sigma=sobel_mask_sigma,
        sobel_mask_floor=sobel_mask_floor,
        strength_values=strength_values,
        hard_negative_p=hard_negative_p,
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load encoder
    encoder, num_bits, encoder_input_size = load_encoder(args.encoder, device)

    # Parse strengths
    strength_values = None
    if args.strengths:
        strength_values = [float(s) for s in args.strengths.split(",")]

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        encoder=encoder,
        data_dir=args.data_dir,
        num_bits=num_bits,
        encoder_input_size=encoder_input_size,
        val_split=args.val_split,
        seed=args.seed,
        use_domain_randomization=args.domain_randomization,
        sobel_mask_sigma=args.sobel_sigma,
        sobel_mask_floor=args.sobel_floor,
        strength_values=strength_values,
        perspective_strength=tuple(args.perspective_strength),
        hard_negative_p=args.hard_negative_p,
    )

    # Create data loaders
    # Encoder lives on GPU/MPS in main process — workers can't access it
    num_workers = 0 if device.type in ("cuda", "mps") else args.num_workers
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
        grad_clip=args.grad_clip,
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
