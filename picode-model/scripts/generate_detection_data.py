#!/usr/bin/env python
"""Pre-generate detection training data using encoder + Sobel mask.

Encodes all images once with random messages, applies Sobel mask and strength
sampling, saves sharded .pt files for fast training with PregeneratedDetectionDataset.

Usage:
    python scripts/generate_detection_data.py \
        --encoder checkpoints/best.pt \
        --data-dir data/train \
        --output-dir data/detection_b72s20m85 \
        --sobel-sigma 5.0 --sobel-floor 0.85 \
        --strengths "0.010,0.012,0.015,0.020"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms

from picode.detection.training.hard_negative import HardNegativeTransform
from picode.models.factory import create_encoder
from picode.training.config import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detection training data")
    parser.add_argument("--encoder", type=str, required=True, help="Encoder checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sobel-sigma", type=float, default=5.0, help="Sobel sigma")
    parser.add_argument("--sobel-floor", type=float, default=0.85, help="Sobel floor")
    parser.add_argument(
        "--strengths", type=str, default="0.010,0.012,0.015,0.020",
        help="Comma-separated strengths",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard")
    parser.add_argument("--input-size", type=int, default=320, help="Detector input size")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Positive ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_encoder(
    checkpoint_path: str, device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Load trained encoder from checkpoint."""
    print(f"Loading encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")
    encoder_size = model_cfg.get("encoder_size", 400)

    training_cfg = config.get("training", {})
    num_bits = training_cfg.get("num_bits", 100)

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

    loss_cfg = config.get("loss", {})
    use_mask = loss_cfg.get("mask_reg") is not None

    mc = ModelConfig(
        type=model_type, encoder_size=encoder_size,
        decoder_size=model_cfg.get("decoder_size", 400),
    )
    encoder = create_encoder(mc, num_bits=num_bits, strength=strength, use_mask=use_mask)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device)
    encoder.eval()
    print(f"Encoder loaded: type={model_type}, num_bits={num_bits}, size={encoder_size}")
    return encoder, num_bits, encoder_size


def sobel_texture_mask(
    image: Tensor, sigma: float, floor: float,
) -> Tensor:
    """Compute Sobel gradient texture mask. Returns (1,1,H,W) in [floor, 1.0]."""
    gray = image.mean(dim=1, keepdim=True)
    sx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=image.device,
    ).view(1, 1, 3, 3)
    sy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=image.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sx, padding=1)
    gy = F.conv2d(gray, sy, padding=1)
    grad_mag = (gx ** 2 + gy ** 2).sqrt()

    k = 2 * math.ceil(3 * sigma) + 1
    ax = torch.arange(k, dtype=torch.float32, device=image.device) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    gk = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    gk = (gk / gk.sum()).view(1, 1, k, k)
    grad_smooth = F.conv2d(grad_mag, gk, padding=k // 2)

    grad_max = grad_smooth.amax(dim=(-2, -1), keepdim=True) + 1e-8
    mask = floor + (1.0 - floor) * (grad_smooth / grad_max)
    return mask


def random_perspective_corners(
    perspective_strength: tuple[float, float] = (0.05, 0.20),
) -> Tensor:
    """Generate random quadrilateral corners (normalized [0, 1])."""
    cx, cy = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
    size_x = random.uniform(0.5, 0.8)
    size_y = random.uniform(0.5, 0.8)

    corners = torch.tensor([
        [cx - size_x / 2, cy - size_y / 2],
        [cx + size_x / 2, cy - size_y / 2],
        [cx + size_x / 2, cy + size_y / 2],
        [cx - size_x / 2, cy + size_y / 2],
    ])

    strength = random.uniform(*perspective_strength)
    for i in range(4):
        corners[i, 0] += random.uniform(-strength, strength)
        corners[i, 1] += random.uniform(-strength, strength)

    return corners.clamp(0.02, 0.98)


def generate_positive(
    image: Tensor,
    encoder: nn.Module,
    num_bits: int,
    device: torch.device,
    input_size: int,
    sobel_sigma: float,
    sobel_floor: float,
    strength_values: list[float],
) -> dict[str, Tensor]:
    """Generate one positive (encoded) sample."""
    message = torch.randint(0, 2, (1, num_bits)).float().to(device)
    image_on_device = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = encoder(image_on_device, message)
        if isinstance(output, dict):
            raw_encoded = output["encoded"]
        else:
            raw_encoded = output

        residual = raw_encoded - image_on_device

        # Apply Sobel texture mask
        mask = sobel_texture_mask(image_on_device, sobel_sigma, sobel_floor)
        residual = residual * mask

        # Sample strength
        strength = random.choice(strength_values)
        residual = residual * strength

        watermarked = (image_on_device + residual).clamp(0, 1)

    watermarked = watermarked.squeeze(0)

    # Resize to detector input size
    output_img = F.interpolate(
        watermarked.unsqueeze(0), size=(input_size, input_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0).cpu()

    corners = random_perspective_corners()

    return {
        "image": (output_img * 255).to(torch.uint8),
        "is_watermark": 1.0,
        "corners": corners.flatten(),
        "has_corners": 1.0,
    }


def generate_negative(
    image: Tensor,
    input_size: int,
    hard_negative: HardNegativeTransform,
    hard_negative_p: float = 0.8,
) -> dict[str, Tensor]:
    """Generate one negative (clean) sample."""
    output_img = F.interpolate(
        image.unsqueeze(0), size=(input_size, input_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    if random.random() < hard_negative_p:
        output_img = hard_negative(output_img)

    return {
        "image": (output_img.clamp(0, 1).cpu() * 255).to(torch.uint8),
        "is_watermark": 0.0,
        "corners": torch.zeros(8),
        "has_corners": 0.0,
    }


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

    encoder, num_bits, encoder_size = load_encoder(args.encoder, device)
    strength_values = [float(s) for s in args.strengths.split(",")]
    hard_negative = HardNegativeTransform()

    # Collect images
    data_dir = Path(args.data_dir)
    image_paths = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))
    if args.max_images:
        random.shuffle(image_paths)
        image_paths = image_paths[:args.max_images]
    print(f"Using {len(image_paths)} images from {data_dir}")

    load_transform = transforms.Compose([
        transforms.Resize((encoder_size, encoder_size)),
        transforms.ToTensor(),
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard: list[dict] = []
    shard_idx = 0
    total_samples = 0
    start_time = time.time()

    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = load_transform(image)
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            continue

        # Generate positive sample
        if random.random() < args.positive_ratio:
            sample = generate_positive(
                image_tensor, encoder, num_bits, device, args.input_size,
                args.sobel_sigma, args.sobel_floor, strength_values,
            )
        else:
            sample = generate_negative(
                image_tensor, args.input_size, hard_negative,
            )

        shard.append(sample)
        total_samples += 1

        # Save shard when full
        if len(shard) >= args.shard_size:
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            torch.save(shard, shard_path)
            elapsed = time.time() - start_time
            rate = total_samples / elapsed
            print(
                f"  Shard {shard_idx:5d} saved ({len(shard)} samples) | "
                f"{total_samples:6d}/{len(image_paths)} images | "
                f"{rate:.1f} img/s"
            )
            shard = []
            shard_idx += 1

    # Save remaining samples
    if shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(shard, shard_path)
        print(f"  Shard {shard_idx:5d} saved ({len(shard)} samples)")
        shard_idx += 1

    # Save metadata
    metadata = {
        "num_shards": shard_idx,
        "num_samples": total_samples,
        "shard_size": args.shard_size,
        "input_size": args.input_size,
        "encoder_checkpoint": args.encoder,
        "sobel_sigma": args.sobel_sigma,
        "sobel_floor": args.sobel_floor,
        "strength_values": strength_values,
        "positive_ratio": args.positive_ratio,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone! {total_samples} samples in {shard_idx} shards ({elapsed:.0f}s)")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
