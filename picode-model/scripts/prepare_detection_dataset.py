#!/usr/bin/env python3
"""Prepare pre-encoded detection dataset for FastDetector training.

This script pre-generates watermarked images locally so Modal training
only needs to do the actual detector training (no encoder overhead).

Usage:
    # Generate dataset locally (CPU/MPS)
    python scripts/prepare_detection_dataset.py \
        --encoder-checkpoint checkpoints/best.pt \
        --image-dir data/coco/train2017 \
        --output-dir data/detection_dataset \
        --num-samples 50000 \
        --positive-ratio 0.5

    # Upload to Modal volume
    modal volume put picode-data data/detection_dataset /detection_dataset

    # Then run training on Modal with pre-generated data
    modal run scripts/modal_train_detector.py::train_detector_pregenerated --epochs 10
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_encoder(checkpoint_path: str, device: torch.device):
    """Load encoder from checkpoint (any model type).

    Returns:
        Tuple of (encoder, num_bits, encoder_input_size).
    """
    from picode.models.factory import create_encoder
    from picode.training.config import ModelConfig

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")
    encoder_size = model_cfg.get("encoder_size", 400)
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


def random_perspective_corners(strength_range: tuple[float, float] = (0.05, 0.20)) -> torch.Tensor:
    """Generate random quadrilateral corners (normalized [0, 1])."""
    cx, cy = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
    size_x = random.uniform(0.5, 0.8)
    size_y = random.uniform(0.5, 0.8)

    corners = torch.tensor([
        [cx - size_x / 2, cy - size_y / 2],  # TL
        [cx + size_x / 2, cy - size_y / 2],  # TR
        [cx + size_x / 2, cy + size_y / 2],  # BR
        [cx - size_x / 2, cy + size_y / 2],  # BL
    ])

    strength = random.uniform(*strength_range)
    for i in range(4):
        corners[i, 0] += random.uniform(-strength, strength)
        corners[i, 1] += random.uniform(-strength, strength)

    return corners.clamp(0.02, 0.98)


def generate_sample(
    image_path: Path,
    encoder: torch.nn.Module,
    num_bits: int,
    device: torch.device,
    is_positive: bool,
    input_size: int = 320,
    encoder_size: int = 400,
) -> dict:
    """Generate a single training sample."""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Resize to encoder input size
    transform = transforms.Compose([
        transforms.Resize((encoder_size, encoder_size)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)

    if is_positive:
        # Generate watermarked image
        message = torch.randint(0, 2, (1, num_bits)).float().to(device)
        image_input = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = encoder(image_input, message)
            # PicoTrust with strength returns dict {"encoded": tensor}
            if isinstance(output, dict):
                watermarked = output["encoded"]
            else:
                watermarked = output
            watermarked = torch.clamp(watermarked, 0, 1)

        # Resize to detector input size
        output = F.interpolate(
            watermarked,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).cpu()

        corners = random_perspective_corners()

        return {
            "image": output,
            "is_watermark": 1.0,
            "corners": corners.flatten(),
            "has_corners": 1.0,
        }
    else:
        # Clean image (no watermark)
        output = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return {
            "image": output,
            "is_watermark": 0.0,
            "corners": torch.zeros(8),
            "has_corners": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Prepare detection dataset")
    parser.add_argument("--encoder-checkpoint", required=True, help="Path to encoder checkpoint")
    parser.add_argument("--image-dir", required=True, help="Directory with source images")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Ratio of positive samples")
    parser.add_argument("--input-size", type=int, default=320, help="Detector input size")
    parser.add_argument("--batch-size", type=int, default=100, help="Samples per shard file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load encoder
    print(f"Loading encoder from {args.encoder_checkpoint}...")
    encoder, num_bits, encoder_size = load_encoder(args.encoder_checkpoint, device)

    # Collect image paths
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    print(f"Found {len(image_paths)} source images")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    num_positive = int(args.num_samples * args.positive_ratio)
    num_negative = args.num_samples - num_positive
    print(f"Generating {num_positive} positive + {num_negative} negative = {args.num_samples} samples")

    samples = []
    shard_idx = 0

    def save_shard():
        nonlocal samples, shard_idx
        if samples:
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            torch.save(samples, shard_path)
            shard_idx += 1
            samples = []

    # Generate positive samples
    print("Generating positive samples...")
    for i in tqdm(range(num_positive), desc="Positive"):
        image_path = random.choice(image_paths)
        try:
            sample = generate_sample(
                image_path, encoder, num_bits, device,
                is_positive=True, input_size=args.input_size, encoder_size=encoder_size
            )
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}")
            continue

        if len(samples) >= args.batch_size:
            save_shard()

    # Generate negative samples
    print("Generating negative samples...")
    for i in tqdm(range(num_negative), desc="Negative"):
        image_path = random.choice(image_paths)
        try:
            sample = generate_sample(
                image_path, encoder, num_bits, device,
                is_positive=False, input_size=args.input_size, encoder_size=encoder_size
            )
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}")
            continue

        if len(samples) >= args.batch_size:
            save_shard()

    # Save remaining samples
    save_shard()

    # Save metadata
    metadata = {
        "num_samples": args.num_samples,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "positive_ratio": args.positive_ratio,
        "input_size": args.input_size,
        "num_bits": num_bits,
        "num_shards": shard_idx,
        "samples_per_shard": args.batch_size,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  - {shard_idx} shard files")
    print(f"  - metadata.json")
    print(f"\nTo upload to Modal:")
    print(f"  modal volume put picode-data {output_dir} /detection_dataset")


if __name__ == "__main__":
    main()
