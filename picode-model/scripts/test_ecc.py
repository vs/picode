#!/usr/bin/env python3
"""Compare BCH (hard-decision) vs LDPC (soft-decision) ECC on a trained model.

Tests the full pipeline: ECC encode → stego encode → distort → stego decode → ECC decode.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from picode.ecc.bch.bch import BCH
from picode.ecc.ldpc.ldpc import LDPC


def load_model(checkpoint_path: Path, device: torch.device):
    """Load encoder/decoder from checkpoint."""
    from picode.models.factory import create_decoder, create_encoder
    from picode.training.config import ModelConfig

    data = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = data.get("config", {})
    num_bits = config.get("training", {}).get("num_bits", 100)
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")
    training_cfg = config.get("training", {})

    # Compute annealed strength
    residual_strength = training_cfg.get("residual_strength", 0)
    strength = None
    if residual_strength > 0:
        step = data.get("step", 0)
        anneal_target = training_cfg.get("residual_strength_anneal_target", residual_strength)
        anneal_start = training_cfg.get("residual_strength_anneal_start", 0)
        anneal_steps = training_cfg.get("residual_strength_anneal_steps", 1)
        if step >= anneal_start and anneal_steps > 0:
            t = min((step - anneal_start) / anneal_steps, 1.0)
            strength = residual_strength + t * (anneal_target - residual_strength)
        else:
            strength = residual_strength

    mc = ModelConfig(
        type=model_type,
        encoder_size=model_cfg.get("encoder_size", 400),
        decoder_size=model_cfg.get("decoder_size", 400),
    )
    encoder = create_encoder(mc, num_bits=num_bits, strength=strength).to(device)
    decoder = create_decoder(mc, num_bits=num_bits).to(device)
    encoder.load_state_dict(data["encoder_state"])
    decoder.load_state_dict(data["decoder_state"])
    encoder.eval()
    decoder.eval()

    decoder_size = model_cfg.get("decoder_size")
    image_size = model_cfg.get("encoder_size", 400)

    return encoder, decoder, num_bits, image_size, decoder_size


def create_distortion(name: str, strength: float):
    """Create a distortion function."""
    from picode.distortions.native import (
        BrightnessHue,
        GaussianBlur,
        GaussianNoise,
        JPEGCompression,
    )

    if name == "jpeg":
        return JPEGCompression(quality=int(strength))
    elif name == "blur":
        return GaussianBlur(sigma=strength)
    elif name == "noise":
        return GaussianNoise(std=strength)
    elif name == "brightness":
        return BrightnessHue(intensity=1.0, rnd_bri=strength, rnd_hue=0.0)
    else:
        raise ValueError(f"Unknown distortion: {name}")


def run_ecc_test(
    encoder,
    decoder,
    image_paths: list[Path],
    device: torch.device,
    num_bits: int,
    image_size: int,
    decoder_size: int | None,
):
    """Run full ECC comparison: no ECC vs BCH vs LDPC."""
    to_tensor = transforms.ToTensor()

    # Setup ECC codes — adapt to num_bits
    if num_bits >= 94:
        # 100-bit model: BCH(63,36,t=5) + BCH(31,21,t=2) = 57 data in 94 coded
        bch_63 = BCH(63, 36)
        bch_31 = BCH(31, 21)
        bch_codes = [(bch_63, 63, 36), (bch_31, 31, 21)]
        bch_data_bits = 36 + 21  # 57
        bch_code_bits = 63 + 31  # 94
        bch_padding = num_bits - bch_code_bits
    elif num_bits >= 63:
        # 80-bit model: BCH(63,36,t=5) + padding = 36 data in 63 coded
        bch_63 = BCH(63, 36)
        bch_codes = [(bch_63, 63, 36)]
        bch_data_bits = 36
        bch_code_bits = 63
        bch_padding = num_bits - bch_code_bits
    else:
        # 31-bit or smaller: BCH(31,21,t=2)
        bch_31 = BCH(31, 21)
        bch_codes = [(bch_31, 31, 21)]
        bch_data_bits = 21
        bch_code_bits = 31
        bch_padding = num_bits - bch_code_bits

    bch_desc = " + ".join(f"({n},{k},t={c.t})" for c, n, k in bch_codes)
    print(f"BCH: {bch_desc} = {bch_data_bits} data bits in {bch_code_bits} coded bits (+{bch_padding} padding)")

    # LDPC: single code, n=num_bits (d_c must divide n)
    # Find best d_c that divides num_bits
    for d_c in [5, 4, 8, 10]:
        if num_bits % d_c == 0:
            break
    ldpc = LDPC(n=num_bits, d_v=2, d_c=d_c, snr=8.0, seed=42)
    ldpc_data_bits = ldpc.message_length
    print(f"LDPC: ({num_bits}, {ldpc_data_bits}) = {ldpc_data_bits} data bits in {num_bits} coded bits, rate={ldpc.rate:.3f}")

    # Distortion scenarios — including stress tests
    distortions = [
        ("clean", None, None),
        ("jpeg_q5", "jpeg", 5),
        ("jpeg_q10", "jpeg", 10),
        ("jpeg_q30", "jpeg", 30),
        ("blur_2", "blur", 2.0),
        ("blur_4", "blur", 4.0),
        ("blur_6", "blur", 6.0),
        ("noise_005", "noise", 0.05),
        ("noise_010", "noise", 0.10),
        ("noise_015", "noise", 0.15),
        ("noise_020", "noise", 0.20),
        ("brightness_03", "brightness", 0.3),
        ("brightness_05", "brightness", 0.5),
        ("brightness_07", "brightness", 0.7),
    ]

    # Results accumulators
    results = {dist_name: {"no_ecc": [], "bch_hard": [], "ldpc_soft": []} for dist_name, _, _ in distortions}

    num_trials = 10  # random messages per image

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image_cropped = ImageOps.fit(image, (image_size, image_size), method=Image.LANCZOS)
        image_tensor = to_tensor(image_cropped).unsqueeze(0).to(device)
        print(f"\n  {img_path.name}:")

        for trial in range(num_trials):
            # Generate random data for each ECC scheme
            # --- No ECC: 100 random bits ---
            raw_message = torch.randint(0, 2, (1, num_bits), device=device).float()

            # --- BCH: data bits → coded bits + padding ---
            bch_data_parts = []
            bch_coded_parts = []
            for bch_c, bch_n, bch_k in bch_codes:
                data_part = torch.randint(0, 2, (1, bch_k)).float()
                coded_part = bch_c.encode(data_part)
                bch_data_parts.append(data_part)
                bch_coded_parts.append(coded_part)
            if bch_padding > 0:
                bch_coded_parts.append(torch.zeros(1, bch_padding))
            bch_message = torch.cat(bch_coded_parts, dim=1).to(device)

            # --- LDPC: k data bits → 100 coded bits ---
            ldpc_data = torch.randint(0, 2, (1, ldpc_data_bits)).float()
            ldpc_message = ldpc.encode(ldpc_data).to(device)

            with torch.no_grad():
                # Encode all three messages into images
                enc_raw = encoder(image_tensor, raw_message)
                enc_bch = encoder(image_tensor, bch_message)
                enc_ldpc = encoder(image_tensor, ldpc_message)

                encoded_raw = (enc_raw["encoded"] if isinstance(enc_raw, dict) else enc_raw).clamp(0, 1)
                encoded_bch = (enc_bch["encoded"] if isinstance(enc_bch, dict) else enc_bch).clamp(0, 1)
                encoded_ldpc = (enc_ldpc["encoded"] if isinstance(enc_ldpc, dict) else enc_ldpc).clamp(0, 1)

            for dist_name, dist_type, dist_strength in distortions:
                with torch.no_grad():
                    # Apply distortion
                    if dist_type:
                        distortion = create_distortion(dist_type, dist_strength)
                        d_raw = distortion(encoded_raw)
                        d_bch = distortion(encoded_bch)
                        d_ldpc = distortion(encoded_ldpc)
                    else:
                        d_raw = encoded_raw
                        d_bch = encoded_bch
                        d_ldpc = encoded_ldpc

                    # Downsample for decoder if needed
                    def maybe_downsample(x):
                        if decoder_size and x.shape[-1] != decoder_size:
                            return F.interpolate(x, size=(decoder_size, decoder_size), mode="bilinear", align_corners=False)
                        return x

                    # Decode — get raw probabilities (sigmoid output)
                    probs_raw = torch.sigmoid(decoder(maybe_downsample(d_raw)))
                    probs_bch = torch.sigmoid(decoder(maybe_downsample(d_bch)))
                    probs_ldpc = torch.sigmoid(decoder(maybe_downsample(d_ldpc)))

                # --- No ECC: hard threshold ---
                bits_raw = (probs_raw > 0.5).float()
                acc_raw = (bits_raw == raw_message).float().mean().item()
                results[dist_name]["no_ecc"].append(acc_raw)

                # --- BCH: hard threshold then ECC decode ---
                bits_bch = (probs_bch > 0.5).float()
                offset = 0
                acc_bch_weighted = 0.0
                for i, (bch_c, bch_n, bch_k) in enumerate(bch_codes):
                    bch_recv = bits_bch[:, offset:offset + bch_n].cpu()
                    bch_dec, bch_ok = bch_c.decode(bch_recv)
                    acc_part = (bch_dec == bch_data_parts[i]).float().mean().item()
                    acc_bch_weighted += acc_part * bch_k
                    offset += bch_n
                acc_bch = acc_bch_weighted / bch_data_bits
                results[dist_name]["bch_hard"].append(acc_bch)

                # --- LDPC: soft-decision decode ---
                ldpc_probs = probs_ldpc[:, :num_bits].cpu()
                ldpc_dec, ldpc_ok = ldpc.decode(ldpc_probs)
                acc_ldpc = (ldpc_dec == ldpc_data).float().mean().item()
                results[dist_name]["ldpc_soft"].append(acc_ldpc)

    # Print results
    print("\n" + "=" * 70)
    print("ECC COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Distortion':<18} {'No ECC (100b)':<16} {'BCH hard (57b)':<16} {'LDPC soft ({ldpc_data_bits}b)':<16}")
    print("-" * 70)
    for dist_name, _, _ in distortions:
        no_ecc = np.mean(results[dist_name]["no_ecc"])
        bch = np.mean(results[dist_name]["bch_hard"])
        ldpc_acc = np.mean(results[dist_name]["ldpc_soft"])
        print(f"{dist_name:<18} {no_ecc:>13.1%}    {bch:>13.1%}    {ldpc_acc:>13.1%}")

    print("\nNote: No ECC = raw bit accuracy (100 bits)")
    print("      BCH = data bit accuracy after ECC correction (57 payload bits)")
    print(f"      LDPC = data bit accuracy after soft BP decoding ({ldpc_data_bits} payload bits)")


def main():
    parser = argparse.ArgumentParser(description="Test ECC on steganography model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10, help="Random messages per image")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    encoder, decoder, num_bits, image_size, decoder_size = load_model(args.checkpoint, device)
    print(f"Model: {num_bits} bits, {image_size}x{image_size} encoder, {decoder_size}x{decoder_size} decoder")

    image_paths = sorted(args.dir.glob("*.jpg"))[:args.max_images]
    print(f"Images: {len(image_paths)}")

    run_ecc_test(encoder, decoder, image_paths, device, num_bits, image_size, decoder_size)


if __name__ == "__main__":
    main()
