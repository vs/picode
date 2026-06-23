#!/usr/bin/env python3
"""Evaluate a trained checkpoint."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from picode.models.factory import create_decoder, create_encoder
from picode.training.config import ModelConfig
from picode.training.evaluation import DEFAULT_ROBUSTNESS_SWEEP, Evaluator


def load_checkpoint(path: Path, device: torch.device) -> tuple:
    """Load encoder and decoder from checkpoint."""
    data = torch.load(path, weights_only=False, map_location=device)

    # Get config
    config = data.get("config", {})
    num_bits = config.get("training", {}).get("num_bits", 100)
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")

    print(f"Model type: {model_type}")

    # Create models via factory
    mc = ModelConfig(
        type=model_type,
        encoder_size=model_cfg.get("encoder_size", 400),
        decoder_size=model_cfg.get("decoder_size", 400),
    )
    # Detect v2 features from config
    loss_cfg = config.get("loss", {})
    use_mask = loss_cfg.get("mask_reg") is not None
    training_cfg = config.get("training", {})
    residual_strength = training_cfg.get("residual_strength", 0)
    # Compute annealed strength at saved step (config stores initial value)
    if residual_strength > 0:
        step = data.get("step", 0)
        anneal_target = training_cfg.get("residual_strength_anneal_target", residual_strength)
        anneal_start = training_cfg.get("residual_strength_anneal_start", 0)
        anneal_steps = training_cfg.get("residual_strength_anneal_steps", 1)
        anneal_schedule = training_cfg.get("anneal_schedule", "linear")
        if step >= anneal_start and anneal_steps > 0:
            t = min((step - anneal_start) / anneal_steps, 1.0)
            if anneal_schedule == "exponential" and residual_strength > 0 and anneal_target > 0:
                strength = residual_strength * (anneal_target / residual_strength) ** t
            else:
                strength = residual_strength + t * (anneal_target - residual_strength)
        else:
            strength = residual_strength
        print(f"Residual strength: {strength:.4f} (annealed from {residual_strength} at step {step})")
    else:
        strength = None

    blur_sigma = training_cfg.get("residual_blur_sigma", 0.0)
    encoder = create_encoder(
        mc, num_bits=num_bits, strength=strength, use_mask=use_mask,
        residual_blur_sigma=blur_sigma,
    ).to(device)
    decoder = create_decoder(mc, num_bits=num_bits).to(device)

    # Load weights
    encoder.load_state_dict(data["encoder_state"])
    decoder.load_state_dict(data["decoder_state"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, config


def evaluate_single_image(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    num_bits: int = 100,
    image_size: int = 400,
    model_type: str = "stegastamp",
    frame_pct: float = 0.04,
    decoder_size: int | None = None,
) -> dict:
    """Encode a message in an image and decode it back."""
    image = Image.open(image_path).convert("RGB")

    # Create random message
    message = torch.randint(0, 2, (1, num_bits), device=device).float()
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        if model_type == "picodeframe":
            frame_width = int(image_size * frame_pct)
            inner_size = image_size - 2 * frame_width

            # Crop to inner size, reflection-pad to full size
            image_cropped = ImageOps.fit(image, (inner_size, inner_size), method=Image.LANCZOS)
            inner_tensor = to_tensor(image_cropped).unsqueeze(0).to(device)
            padded = F.pad(inner_tensor, (frame_width,) * 4, mode="reflect")

            # Encode with frame_width
            encoded = encoder(padded, message, frame_width=frame_width).clamp(0.0, 1.0)

            # Decode with mask
            mask = torch.zeros(1, 1, image_size, image_size, device=device)
            mask[:, :, frame_width:image_size - frame_width,
                 frame_width:image_size - frame_width] = 1.0
            decoded_logits = decoder(encoded, mask=mask, frame_width=frame_width)
            decoded = (decoded_logits > 0).float()

            # PSNR on frame region only (center is identical by construction)
            frame_mask = (1 - mask).expand_as(encoded)
            frame_pixels_enc = encoded[frame_mask.bool()]
            frame_pixels_pad = padded[frame_mask.bool()]
            mse = ((frame_pixels_enc - frame_pixels_pad) ** 2).mean().item()
        else:
            image_cropped = ImageOps.fit(image, (image_size, image_size), method=Image.LANCZOS)
            image_tensor = to_tensor(image_cropped).unsqueeze(0).to(device)

            enc_out = encoder(image_tensor, message)
            encoded = (enc_out["encoded"] if isinstance(enc_out, dict) else enc_out).clamp(0.0, 1.0)
            decoder_input = encoded
            if decoder_size and encoded.shape[-1] != decoder_size:
                decoder_input = F.interpolate(
                    encoded, size=(decoder_size, decoder_size), mode="bilinear", align_corners=False,
                )
            decoded_logits = decoder(decoder_input)
            decoded = (decoded_logits > 0).float()
            mse = ((encoded - image_tensor) ** 2).mean().item()

        bit_acc = (decoded == message).float().mean().item()
        psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else float("inf")

    return {
        "bit_accuracy": bit_acc,
        "psnr": psnr,
        "message": message[0].cpu().numpy(),
        "decoded": decoded[0].cpu().numpy(),
    }


def run_robustness_sweep(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    num_bits: int = 100,
    image_size: int = 400,
    model_type: str = "stegastamp",
    frame_pct: float = 0.04,
    decoder_size: int | None = None,
) -> list:
    """Run robustness sweep on a single image."""
    image = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    message = torch.randint(0, 2, (1, num_bits), device=device).float()

    if model_type == "picodeframe":
        frame_width = int(image_size * frame_pct)
        inner_size = image_size - 2 * frame_width
        image_cropped = ImageOps.fit(image, (inner_size, inner_size), method=Image.LANCZOS)
        inner_tensor = to_tensor(image_cropped).unsqueeze(0).to(device)
        padded = F.pad(inner_tensor, (frame_width,) * 4, mode="reflect")

        mask = torch.zeros(1, 1, image_size, image_size, device=device)
        mask[:, :, frame_width:image_size - frame_width,
             frame_width:image_size - frame_width] = 1.0

        # Manually run robustness sweep with PicodeFrame encode/decode
        from picode.training.evaluation import RobustnessResult
        results: list = []
        with torch.no_grad():
            encoded = encoder(padded, message, frame_width=frame_width).clamp(0.0, 1.0)
            evaluator = Evaluator(encoder, decoder, device)
            for name, strengths in DEFAULT_ROBUSTNESS_SWEEP.items():
                for strength in strengths:
                    distortion = evaluator._create_distortion(name, strength)
                    distorted = distortion(encoded)
                    decoded_logits = decoder(distorted, mask=mask)
                    decoded_binary = (decoded_logits > 0).float()
                    bit_acc = (decoded_binary == message).float().mean().item()
                    msg_acc = (decoded_binary == message).all(dim=1).float().mean().item()
                    results.append(RobustnessResult(
                        distortion=name, strength=strength,
                        bit_accuracy=bit_acc, message_accuracy=msg_acc,
                    ))
        return results
    else:
        image_cropped = ImageOps.fit(image, (image_size, image_size), method=Image.LANCZOS)
        image_tensor = to_tensor(image_cropped).unsqueeze(0).to(device)
        evaluator = Evaluator(encoder, decoder, device, decoder_size=decoder_size)
        return evaluator.robustness_sweep(image_tensor, message, DEFAULT_ROBUSTNESS_SWEEP)


def evaluate_directory(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_dir: Path,
    device: torch.device,
    num_bits: int = 100,
    run_robustness: bool = False,
    max_images: int | None = None,
    image_size: int = 400,
    model_type: str = "stegastamp",
    frame_pct: float = 0.04,
    decoder_size: int | None = None,
) -> dict:
    """Evaluate on all images in a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print(f"No images found in {image_dir}")
        return {}

    if max_images and max_images < len(image_paths):
        image_paths = image_paths[:max_images]
        print(f"Evaluating on {len(image_paths)} images (limited from {image_dir})")
    else:
        print(f"Found {len(image_paths)} images in {image_dir}")

    all_bit_acc = []
    all_psnr = []
    all_robustness: dict[str, dict[float, list[float]]] = {}

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {image_path.name}")

        result = evaluate_single_image(
            encoder, decoder, image_path, device, num_bits, image_size,
            model_type, frame_pct, decoder_size,
        )
        all_bit_acc.append(result["bit_accuracy"])
        all_psnr.append(result["psnr"])
        print(f"  Bit accuracy: {result['bit_accuracy']:.4f}, PSNR: {result['psnr']:.2f} dB")

        if run_robustness:
            rob_results = run_robustness_sweep(
                encoder, decoder, image_path, device, num_bits, image_size,
                model_type, frame_pct, decoder_size,
            )
            for r in rob_results:
                if r.distortion not in all_robustness:
                    all_robustness[r.distortion] = {}
                if r.strength not in all_robustness[r.distortion]:
                    all_robustness[r.distortion][r.strength] = []
                all_robustness[r.distortion][r.strength].append(r.bit_accuracy)

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Images evaluated: {len(image_paths)}")
    print(f"Mean bit accuracy: {sum(all_bit_acc) / len(all_bit_acc):.4f}")
    print(f"Mean PSNR: {sum(all_psnr) / len(all_psnr):.2f} dB")

    if run_robustness and all_robustness:
        print("\nRobustness (mean bit accuracy):")
        for dist_name, strengths in all_robustness.items():
            print(f"\n  {dist_name}:")
            for strength, accs in sorted(strengths.items()):
                mean_acc = sum(accs) / len(accs)
                print(f"    {strength:>6.1f}: {mean_acc:.4f}")

    return {
        "bit_accuracy": sum(all_bit_acc) / len(all_bit_acc),
        "psnr": sum(all_psnr) / len(all_psnr),
        "robustness": all_robustness,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument("--image", type=Path, help="Test image (optional)")
    parser.add_argument("--dir", type=Path, help="Directory of test images")
    parser.add_argument("--max-images", type=int, help="Max images to evaluate")
    parser.add_argument("--robustness", action="store_true", help="Run robustness sweep")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, mps)")
    parser.add_argument(
        "--frame-pct", type=float, default=0.04, dest="frame_pct",
        help="Frame width as fraction of image (PicodeFrame only, default: 0.04)",
    )
    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    encoder, decoder, config = load_checkpoint(args.checkpoint, device)

    num_bits = config.get("training", {}).get("num_bits", 100)
    image_size = config.get("model", {}).get("encoder_size", 400)
    decoder_size_cfg = config.get("model", {}).get("decoder_size")
    decoder_size = decoder_size_cfg if decoder_size_cfg and decoder_size_cfg != image_size else None
    model_type = config.get("model", {}).get("type", "stegastamp")
    step = config.get("step", "unknown")
    print(f"Checkpoint step: {step}, num_bits: {num_bits}, image_size: {image_size}")
    if decoder_size:
        print(f"Decoder size: {decoder_size} (downsampling from {image_size})")
    print(f"Model type: {model_type}")

    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {enc_params:,}, Decoder params: {dec_params:,}")

    if args.dir:
        evaluate_directory(
            encoder, decoder, args.dir, device, num_bits,
            args.robustness, args.max_images, image_size,
            model_type, args.frame_pct, decoder_size,
        )
    elif args.image:
        print(f"\nEvaluating on: {args.image}")
        result = evaluate_single_image(
            encoder, decoder, args.image, device, num_bits, image_size,
            model_type, args.frame_pct, decoder_size,
        )
        print(f"  Bit accuracy: {result['bit_accuracy']:.4f}")
        print(f"  PSNR: {result['psnr']:.2f} dB")

        if args.robustness:
            print("\nRobustness sweep:")
            results = run_robustness_sweep(
                encoder, decoder, args.image, device, num_bits, image_size,
                model_type, args.frame_pct, decoder_size,
            )
            current_dist = None
            for r in results:
                if r.distortion != current_dist:
                    current_dist = r.distortion
                    print(f"\n  {current_dist}:")
                print(f"    {r.strength:>6.1f}: bit_acc={r.bit_accuracy:.4f}, msg_acc={r.message_accuracy:.4f}")
    else:
        print("\nNo test image provided. Use --image or --dir to evaluate.")
        print("Examples:")
        print("  python scripts/evaluate.py checkpoints/best.pt --image test.jpg --robustness")
        print("  python scripts/evaluate.py checkpoints/best.pt --dir data/eval --robustness")


if __name__ == "__main__":
    main()
