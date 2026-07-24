#!/usr/bin/env python
"""Pre-generate composited segmentation training data with ground-truth masks.

Generates three sample types:
  - positive (40%): encoded image composited into background, mask=1 in composited region
  - hard_negative (30%): clean image composited into background, mask=1 in composited region
  - easy_negative (30%): clean image resized only, mask=all zeros

The compositing uses OpenCV warpPerspective so the detector learns to distinguish
encoding artifacts from normal compositing boundaries.

Usage:
    python scripts/generate_seg_detection_data.py \\
        --encoder checkpoints/best.pt \\
        --data-dir data/train \\
        --output-dir data/seg_detection_b72s20m85 \\
        --sobel-sigma 5.0 --sobel-floor 0.85 \\
        --strengths "0.014,0.016,0.018,0.020"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms

from picode.models.factory import create_encoder
from picode.training.config import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate composited segmentation detection training data",
    )
    parser.add_argument("--encoder", type=str, required=True, help="Encoder checkpoint path")
    parser.add_argument("--data-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sobel-sigma", type=float, default=5.0, help="Sobel sigma")
    parser.add_argument("--sobel-floor", type=float, default=0.85, help="Sobel mask floor")
    parser.add_argument(
        "--strengths",
        type=str,
        default="0.014,0.016,0.018,0.020",
        help="Comma-separated residual strengths",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard")
    parser.add_argument("--input-size", type=int, default=320, help="Output image size")
    parser.add_argument(
        "--scale-min", type=float, default=0.50,
        help="Minimum foreground scale relative to background",
    )
    parser.add_argument(
        "--scale-max", type=float, default=0.95,
        help="Maximum foreground scale relative to background",
    )
    parser.add_argument(
        "--perspective-strength",
        type=float,
        default=0.05,
        help="Perspective distortion strength (corner jitter fraction)",
    )
    parser.add_argument(
        "--positive-ratio", type=float, default=0.4,
        help="Fraction of positive samples (encoded + composited)",
    )
    parser.add_argument(
        "--hard-negative-ratio", type=float, default=0.3,
        help="Fraction of hard negatives (clean + composited)",
    )
    parser.add_argument(
        "--model-label", type=str, default=None,
        help="Model label stored in each sample (e.g. 'b72')",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_encoder(
    checkpoint_path: str, device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Load trained encoder from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Target device.

    Returns:
        Tuple of (encoder, num_bits, encoder_input_size).
    """
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


def sobel_texture_mask(
    image: Tensor, sigma: float, floor: float,
) -> Tensor:
    """Compute Sobel gradient texture mask.

    Args:
        image: (1, 3, H, W) image tensor in [0, 1].
        sigma: Gaussian smoothing sigma.
        floor: Minimum mask value (0–1).

    Returns:
        (1, 1, H, W) mask in [floor, 1.0].
    """
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


def _build_perspective_transform(
    fg_h: int,
    fg_w: int,
    bg_size: int,
    scale_min: float,
    scale_max: float,
    perspective_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a perspective transform matrix placing fg into bg canvas.

    Args:
        fg_h: Foreground image height.
        fg_w: Foreground image width.
        bg_size: Background canvas size (square).
        scale_min: Minimum scale of foreground relative to background.
        scale_max: Maximum scale of foreground relative to background.
        perspective_strength: Corner jitter as fraction of bg_size.

    Returns:
        Tuple of (M, dst_corners) where M is a 3x3 homography and dst_corners
        are the 4 destination corner pixels in bg space (TL, TR, BR, BL).
    """
    scale = random.uniform(scale_min, scale_max)
    fg_target_w = int(bg_size * scale)
    fg_target_h = int(fg_target_w * fg_h / fg_w)

    # Center position with random offset
    max_offset_x = bg_size - fg_target_w
    max_offset_y = bg_size - fg_target_h
    ox = random.randint(0, max(0, max_offset_x))
    oy = random.randint(0, max(0, max_offset_y))

    # Base dst corners (TL, TR, BR, BL) in bg pixel space
    dst = np.array([
        [ox, oy],
        [ox + fg_target_w, oy],
        [ox + fg_target_w, oy + fg_target_h],
        [ox, oy + fg_target_h],
    ], dtype=np.float32)

    # Jitter corners for perspective effect
    jitter = perspective_strength * bg_size
    for i in range(4):
        dst[i, 0] += random.uniform(-jitter, jitter)
        dst[i, 1] += random.uniform(-jitter, jitter)
    dst = np.clip(dst, 0, bg_size - 1)

    # Source corners: full fg image
    src = np.array([
        [0, 0],
        [fg_w - 1, 0],
        [fg_w - 1, fg_h - 1],
        [0, fg_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return M, dst


def composite_image(
    fg: np.ndarray,
    bg: np.ndarray,
    M: np.ndarray,
    bg_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite fg onto bg using perspective transform M.

    Args:
        fg: Foreground image (H, W, 3) uint8.
        bg: Background image (bg_size, bg_size, 3) uint8.
        M: 3x3 homography mapping fg → bg coordinates.
        bg_size: Canvas size.

    Returns:
        Tuple of (composited_image, binary_mask) both (bg_size, bg_size).
    """
    # Warp foreground into bg canvas
    warped_fg = cv2.warpPerspective(fg, M, (bg_size, bg_size))

    # Warp white mask to get fg footprint
    white = np.ones((fg.shape[0], fg.shape[1]), dtype=np.float32)
    warped_mask = cv2.warpPerspective(white, M, (bg_size, bg_size))
    binary_mask = (warped_mask > 0.5).astype(np.uint8)

    # Composite: paste warped_fg over bg where mask=1
    composite = bg.copy()
    composite[binary_mask == 1] = warped_fg[binary_mask == 1]

    return composite, binary_mask


def tensor_to_cv2(t: Tensor) -> np.ndarray:
    """Convert (3, H, W) float32 [0,1] tensor to (H, W, 3) uint8 BGR numpy."""
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_tensor(img: np.ndarray) -> Tensor:
    """Convert (H, W, 3) uint8 BGR numpy to (3, H, W) float32 [0,1] tensor."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def generate_positive(
    fg_image: Tensor,
    bg_image: Tensor,
    encoder: nn.Module,
    num_bits: int,
    device: torch.device,
    input_size: int,
    sobel_sigma: float,
    sobel_floor: float,
    strength_values: list[float],
    scale_min: float,
    scale_max: float,
    perspective_strength: float,
    model_label: str | None,
) -> dict:
    """Generate a positive sample: encoded fg composited into bg.

    Args:
        fg_image: (3, H, W) float32 [0,1] foreground image.
        bg_image: (3, H, W) float32 [0,1] background image.
        encoder: Loaded encoder model.
        num_bits: Number of message bits.
        device: Compute device.
        input_size: Output canvas size.
        sobel_sigma: Sobel texture mask sigma.
        sobel_floor: Sobel mask floor.
        strength_values: Pool of residual strength values to sample.
        scale_min: Minimum fg scale.
        scale_max: Maximum fg scale.
        perspective_strength: Corner jitter fraction.
        model_label: Optional label string.

    Returns:
        Sample dict with image (uint8), mask (uint8), is_watermark, model_label.
    """
    message = torch.randint(0, 2, (1, num_bits)).float().to(device)
    fg_on_device = fg_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = encoder(fg_on_device, message)
        raw_encoded = output["encoded"] if isinstance(output, dict) else output
        residual = raw_encoded - fg_on_device
        mask = sobel_texture_mask(fg_on_device, sobel_sigma, sobel_floor)
        residual = residual * mask * random.choice(strength_values)
        watermarked = (fg_on_device + residual).clamp(0, 1)

    watermarked = watermarked.squeeze(0)

    # Resize background to canvas
    bg_resized = F.interpolate(
        bg_image.unsqueeze(0), size=(input_size, input_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    fg_cv2 = tensor_to_cv2(watermarked)
    bg_cv2 = tensor_to_cv2(bg_resized)

    fg_h, fg_w = fg_cv2.shape[:2]
    M, _ = _build_perspective_transform(
        fg_h, fg_w, input_size, scale_min, scale_max, perspective_strength,
    )
    composited, binary_mask = composite_image(fg_cv2, bg_cv2, M, input_size)

    out_tensor = cv2_to_tensor(composited)
    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # (1, H, W)

    result: dict = {
        "image": (out_tensor * 255).to(torch.uint8),
        "mask": mask_tensor.to(torch.uint8),
        "is_watermark": 1.0,
    }
    if model_label is not None:
        result["model_label"] = model_label
    return result


def generate_hard_negative(
    fg_image: Tensor,
    bg_image: Tensor,
    input_size: int,
    scale_min: float,
    scale_max: float,
    perspective_strength: float,
    model_label: str | None,
) -> dict:
    """Generate a hard negative: clean fg composited into bg.

    Args:
        fg_image: (3, H, W) float32 [0,1] clean foreground image.
        bg_image: (3, H, W) float32 [0,1] background image.
        input_size: Output canvas size.
        scale_min: Minimum fg scale.
        scale_max: Maximum fg scale.
        perspective_strength: Corner jitter fraction.
        model_label: Optional label string.

    Returns:
        Sample dict with mask=1 in composited region but is_watermark=0.
    """
    bg_resized = F.interpolate(
        bg_image.unsqueeze(0), size=(input_size, input_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    fg_cv2 = tensor_to_cv2(fg_image)
    bg_cv2 = tensor_to_cv2(bg_resized)

    fg_h, fg_w = fg_cv2.shape[:2]
    M, _ = _build_perspective_transform(
        fg_h, fg_w, input_size, scale_min, scale_max, perspective_strength,
    )
    composited, binary_mask = composite_image(fg_cv2, bg_cv2, M, input_size)

    out_tensor = cv2_to_tensor(composited)
    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

    result: dict = {
        "image": (out_tensor * 255).to(torch.uint8),
        "mask": mask_tensor.to(torch.uint8),
        "is_watermark": 0.0,
    }
    if model_label is not None:
        result["model_label"] = model_label
    return result


def generate_easy_negative(
    image: Tensor,
    input_size: int,
    model_label: str | None,
) -> dict:
    """Generate an easy negative: clean image, no compositing, mask all zeros.

    Args:
        image: (3, H, W) float32 [0,1] image.
        input_size: Output image size.
        model_label: Optional label string.

    Returns:
        Sample dict with all-zero mask and is_watermark=0.
    """
    out = F.interpolate(
        image.unsqueeze(0), size=(input_size, input_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0).clamp(0, 1)

    result: dict = {
        "image": (out * 255).to(torch.uint8),
        "mask": torch.zeros(1, input_size, input_size, dtype=torch.uint8),
        "is_watermark": 0.0,
    }
    if model_label is not None:
        result["model_label"] = model_label
    return result


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

    # Validate ratios
    positive_ratio = args.positive_ratio
    hard_negative_ratio = args.hard_negative_ratio
    easy_negative_ratio = 1.0 - positive_ratio - hard_negative_ratio
    if easy_negative_ratio < 0:
        raise ValueError(
            f"positive_ratio ({positive_ratio}) + hard_negative_ratio "
            f"({hard_negative_ratio}) > 1.0"
        )
    print(
        f"Sample ratios — positive: {positive_ratio:.0%}, "
        f"hard_negative: {hard_negative_ratio:.0%}, "
        f"easy_negative: {easy_negative_ratio:.0%}"
    )

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
    counts = {"positive": 0, "hard_negative": 0, "easy_negative": 0}
    start_time = time.time()

    for i, img_path in enumerate(image_paths):
        try:
            pil_image = Image.open(img_path).convert("RGB")
            fg_tensor = load_transform(pil_image)
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            continue

        # Pick a random background (different from current image)
        bg_paths = [p for p in image_paths if p != img_path]
        if not bg_paths:
            bg_tensor = fg_tensor
        else:
            bg_path = random.choice(bg_paths)
            try:
                bg_pil = Image.open(bg_path).convert("RGB")
                bg_tensor = load_transform(bg_pil)
            except Exception:
                bg_tensor = fg_tensor

        # Decide sample type
        roll = random.random()
        if roll < positive_ratio:
            sample = generate_positive(
                fg_tensor, bg_tensor, encoder, num_bits, device,
                args.input_size, args.sobel_sigma, args.sobel_floor,
                strength_values, args.scale_min, args.scale_max,
                args.perspective_strength, args.model_label,
            )
            counts["positive"] += 1
        elif roll < positive_ratio + hard_negative_ratio:
            sample = generate_hard_negative(
                fg_tensor, bg_tensor,
                args.input_size, args.scale_min, args.scale_max,
                args.perspective_strength, args.model_label,
            )
            counts["hard_negative"] += 1
        else:
            sample = generate_easy_negative(fg_tensor, args.input_size, args.model_label)
            counts["easy_negative"] += 1

        shard.append(sample)
        total_samples += 1

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

    if shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(shard, shard_path)
        print(f"  Shard {shard_idx:5d} saved ({len(shard)} samples)")
        shard_idx += 1

    metadata = {
        "num_shards": shard_idx,
        "num_samples": total_samples,
        "shard_size": args.shard_size,
        "input_size": args.input_size,
        "encoder_checkpoint": args.encoder,
        "sobel_sigma": args.sobel_sigma,
        "sobel_floor": args.sobel_floor,
        "strength_values": strength_values,
        "positive_ratio": positive_ratio,
        "hard_negative_ratio": hard_negative_ratio,
        "easy_negative_ratio": easy_negative_ratio,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "perspective_strength": args.perspective_strength,
        "model_label": args.model_label,
        "counts": counts,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone! {total_samples} samples in {shard_idx} shards ({elapsed:.0f}s)")
    print(
        f"  positive={counts['positive']}, "
        f"hard_negative={counts['hard_negative']}, "
        f"easy_negative={counts['easy_negative']}"
    )
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
