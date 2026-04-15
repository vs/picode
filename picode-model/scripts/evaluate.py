#!/usr/bin/env python3
"""Evaluate a trained checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from picode.models import stegastamp, picode as picode_model, picode_v2
from picode.training.evaluation import DEFAULT_ROBUSTNESS_SWEEP, Evaluator


def detect_model_type(encoder_state: dict) -> str:
    """Detect model type from state dict keys."""
    keys = list(encoder_state.keys())
    # Check for characteristic keys
    if any("message_expander" in k for k in keys):
        # picode_v2 uses MessageExpander
        return "picode_v2"
    elif any("msg_fc" in k for k in keys):
        # Old stegastamp with ConvBnRelu blocks
        return "stegastamp_legacy"
    elif any("secret_dense" in k for k in keys):
        if any("norm1" in k for k in keys):
            return "picode"
        else:
            return "stegastamp"
    return "stegastamp"


def load_checkpoint(path: Path, device: torch.device) -> tuple:
    """Load encoder and decoder from checkpoint."""
    data = torch.load(path, weights_only=False, map_location=device)

    # Get config
    config = data.get("config", {})
    num_bits = config.get("training", {}).get("num_bits", 100)
    model_type = config.get("training", {}).get("model")

    # Auto-detect from state dict if not in config
    if not model_type:
        model_type = detect_model_type(data["encoder_state"])

    print(f"Model type: {model_type}")

    # Create models based on type
    if model_type == "picode_v2":
        # Get residual_scale from config if available
        residual_scale = config.get("training", {}).get("residual_scale", 0.1)
        encoder = picode_v2.Encoder(num_bits=num_bits, residual_scale=residual_scale).to(device)
        decoder = picode_v2.Decoder(num_bits=num_bits).to(device)
    elif model_type == "picode":
        encoder = picode_model.Encoder(num_bits=num_bits).to(device)
        decoder = picode_model.Decoder(num_bits=num_bits).to(device)
    elif model_type == "stegastamp_legacy":
        # Import legacy model if available, otherwise skip
        print("WARNING: Legacy stegastamp model detected. This checkpoint is incompatible.")
        print("Please use a checkpoint trained with the current model architecture.")
        raise ValueError("Legacy model format not supported")
    else:
        encoder = stegastamp.Encoder(num_bits=num_bits).to(device)
        decoder = stegastamp.Decoder(num_bits=num_bits).to(device)

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
) -> dict:
    """Encode a message in an image and decode it back."""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Create random message
    message = torch.randint(0, 2, (1, num_bits), device=device).float()

    with torch.no_grad():
        # Encode
        encoded = encoder(image_tensor, message)

        # Decode
        decoded_logits = decoder(encoded)
        decoded = (decoded_logits > 0).float()

        # Metrics
        bit_acc = (decoded == message).float().mean().item()
        mse = ((encoded - image_tensor) ** 2).mean().item()
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
) -> list:
    """Run robustness sweep on a single image."""
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    message = torch.randint(0, 2, (1, num_bits), device=device).float()

    evaluator = Evaluator(encoder, decoder, device)
    results = evaluator.robustness_sweep(image_tensor, message, DEFAULT_ROBUSTNESS_SWEEP)

    return results


def evaluate_directory(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_dir: Path,
    device: torch.device,
    num_bits: int = 100,
    run_robustness: bool = False,
    max_images: int | None = None,
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

        result = evaluate_single_image(encoder, decoder, image_path, device, num_bits)
        all_bit_acc.append(result["bit_accuracy"])
        all_psnr.append(result["psnr"])
        print(f"  Bit accuracy: {result['bit_accuracy']:.4f}, PSNR: {result['psnr']:.2f} dB")

        if run_robustness:
            rob_results = run_robustness_sweep(encoder, decoder, image_path, device, num_bits)
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
    step = config.get("step", "unknown")
    print(f"Checkpoint step: {step}, num_bits: {num_bits}")

    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {enc_params:,}, Decoder params: {dec_params:,}")

    if args.dir:
        evaluate_directory(encoder, decoder, args.dir, device, num_bits, args.robustness, args.max_images)
    elif args.image:
        print(f"\nEvaluating on: {args.image}")
        result = evaluate_single_image(encoder, decoder, args.image, device, num_bits)
        print(f"  Bit accuracy: {result['bit_accuracy']:.4f}")
        print(f"  PSNR: {result['psnr']:.2f} dB")

        if args.robustness:
            print("\nRobustness sweep:")
            results = run_robustness_sweep(encoder, decoder, args.image, device, num_bits)
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
