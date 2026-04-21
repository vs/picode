#!/usr/bin/env python
"""Evaluate trained FastDetector.

Usage:
    python scripts/evaluate_detector.py checkpoints/detection/best.pt --data-dir data/coco/val2017
    python scripts/evaluate_detector.py checkpoints/detection/best.pt --encoder checkpoints/best.pt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from picode.detection.fast_detector import FastDetector
from picode.detection.training import (
    DetectionDataset,
    DetectionEvaluator,
)
from picode.models.stegastamp import Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FastDetector")
    parser.add_argument("checkpoint", type=str, help="Path to detector checkpoint")
    parser.add_argument(
        "--encoder",
        type=str,
        default="checkpoints/best.pt",
        help="Path to encoder checkpoint (for synthetic eval)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/coco/val2017",
        help="Directory containing evaluation images",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Number of samples to evaluate"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load detector
    print(f"Loading detector: {args.checkpoint}")
    detector = FastDetector.from_checkpoint(args.checkpoint, device=str(device))

    # Load encoder
    print(f"Loading encoder: {args.encoder}")
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    num_bits = 100
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        num_bits = ckpt["config"].get("training", {}).get("num_bits", 100)

    encoder = Encoder(num_bits=num_bits)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device).eval()

    # Create evaluation dataset
    print(f"Creating evaluation dataset from {args.data_dir}...")
    dataset = DetectionDataset(
        image_dir=args.data_dir,
        encoder=encoder,
        positive_ratio=0.5,
    )

    # Limit samples
    if args.num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"Evaluating on {len(dataset)} samples...")

    # Evaluate
    evaluator = DetectionEvaluator(detector.model, device=str(device))
    metrics = evaluator.evaluate_dataset(loader)

    print("\n=== Evaluation Results ===")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1 Score:  {metrics.f1:.4f}")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Mean IoU:  {metrics.mean_iou:.4f}")


if __name__ == "__main__":
    main()
