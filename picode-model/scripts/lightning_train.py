#!/usr/bin/env python3
"""
Lightning.ai CLI training script for Picode.

Usage:
    # Fresh start
    python scripts/lightning_train.py

    # Resume from checkpoint
    python scripts/lightning_train.py --resume

    # Custom config
    python scripts/lightning_train.py --config configs/lightning_training.yaml

    # Override parameters
    python scripts/lightning_train.py --batch-size 8 --lr 0.0002
"""

import argparse
import glob
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint in the directory."""
    pattern = os.path.join(checkpoint_dir, "*", "checkpoint_*.pt")
    checkpoints = sorted(glob.glob(pattern))
    return checkpoints[-1] if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="Train Picode on Lightning.ai")
    parser.add_argument(
        "--config",
        default="configs/lightning_training.yaml",
        help="Path to training config (default: configs/lightning_training.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint to resume from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Override total training steps",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override data path",
    )

    args = parser.parse_args()

    # Build overrides dict
    overrides = {}
    if args.batch_size:
        overrides["data.batch_size"] = args.batch_size
    if args.lr:
        overrides["training.lr"] = args.lr
    if args.num_steps:
        overrides["training.num_steps"] = args.num_steps
    if args.data_path:
        overrides["data.path"] = args.data_path

    # Import trainer
    from picode.training.trainer import Trainer

    # Check GPU
    import torch
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be very slow.")
        print("On Lightning.ai, switch to GPU machine in the UI or via SDK.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine checkpoint to use
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.resume:
        # Load config to get checkpoint dir
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        checkpoint_dir = config.get("checkpoint", {}).get("dir", "/teamspace/studios/this_studio/checkpoints")
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        
        if checkpoint_path:
            print(f"Resuming from: {checkpoint_path}")
        else:
            print("No checkpoint found, starting fresh.")

    # Create trainer
    if checkpoint_path:
        trainer = Trainer.from_checkpoint(checkpoint_path, overrides=overrides if overrides else None)
    else:
        trainer = Trainer.from_config(args.config, overrides=overrides if overrides else None)

    # Start training
    print(f"\nStarting training...")
    print(f"Config: {args.config}")
    print(f"Overrides: {overrides if overrides else 'none'}")
    print()

    trainer.fit()


if __name__ == "__main__":
    main()
