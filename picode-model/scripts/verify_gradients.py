#!/usr/bin/env python3
"""Verify gradient flow through the entire steganography pipeline.

This script checks that gradients properly flow from the message loss
back through: Decoder → Distortions → Encoder.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from picode.models.stegastamp import Encoder, Decoder
from picode.distortions.native import (
    Compose,
    GaussianNoise,
    BrightnessHue,
    Contrast,
    Saturation,
    JPEGCompression,
    PerspectiveWarp,
)


def check_gradient_flow(
    name: str,
    tensor: Tensor,
    expected_grad: bool = True,
) -> dict[str, float | bool]:
    """Check if tensor has gradients and return stats."""
    has_grad = tensor.grad is not None

    result = {
        "name": name,
        "has_grad": has_grad,
        "expected": expected_grad,
        "match": has_grad == expected_grad,
    }

    if has_grad:
        grad = tensor.grad
        result["grad_mean"] = grad.abs().mean().item()
        result["grad_max"] = grad.abs().max().item()
        result["grad_min"] = grad.abs().min().item()
        result["grad_std"] = grad.std().item()
        result["grad_nonzero"] = (grad.abs() > 1e-10).float().mean().item()

    return result


def test_distortion_gradients():
    """Test gradient flow through individual distortions."""
    print("\n" + "=" * 60)
    print("Testing Individual Distortion Gradients")
    print("=" * 60)

    distortions = [
        ("GaussianNoise", GaussianNoise(intensity=1.0, std=0.02)),
        ("BrightnessHue", BrightnessHue(intensity=1.0, rnd_bri=0.3, rnd_hue=0.1)),
        ("Contrast", Contrast(intensity=1.0, contrast_low=0.5, contrast_high=1.5)),
        ("Saturation", Saturation(intensity=1.0, rnd_sat=1.0)),
        ("JPEGCompression", JPEGCompression(intensity=1.0, quality=50)),
        ("PerspectiveWarp", PerspectiveWarp(intensity=0.5, scale=0.1)),
    ]

    for name, distortion in distortions:
        # Create input with requires_grad
        x = torch.rand(2, 3, 64, 64, requires_grad=True)

        # Forward pass
        y = distortion(x)

        # Create a simple loss (sum of outputs)
        loss = y.sum()

        # Backward pass
        loss.backward()

        # Check gradients
        result = check_gradient_flow(name, x)

        status = "✅" if result["match"] else "❌"
        grad_info = ""
        if result["has_grad"]:
            grad_info = f"mean={result['grad_mean']:.2e}, nonzero={result['grad_nonzero']*100:.1f}%"

        print(f"{status} {name}: grad={result['has_grad']}, {grad_info}")


def test_full_pipeline_gradients():
    """Test gradient flow through the full encoder → distortion → decoder pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline Gradients")
    print("=" * 60)

    device = torch.device("cpu")
    num_bits = 100
    batch_size = 2

    # Create models
    encoder = Encoder(num_bits=num_bits).to(device)
    decoder = Decoder(num_bits=num_bits).to(device)

    # Create distortion pipeline (same as gradient_test.yaml)
    distortions = Compose([
        BrightnessHue(intensity=1.0, rnd_bri=0.1, rnd_hue=0.03),
        Saturation(intensity=1.0, rnd_sat=0.3),
        Contrast(intensity=1.0, contrast_low=0.9, contrast_high=1.1),
        GaussianNoise(intensity=1.0, std=0.01),
        JPEGCompression(intensity=1.0, quality=90),
        PerspectiveWarp(intensity=1.0, scale=0.02),
    ])

    # Create test data
    images = torch.rand(batch_size, 3, 400, 400, device=device)
    messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

    # Forward pass with gradient tracking
    encoded = encoder(images, messages)
    print(f"\n1. Encoded shape: {encoded.shape}, requires_grad: {encoded.requires_grad}")

    distorted = distortions(encoded)
    print(f"2. Distorted shape: {distorted.shape}, requires_grad: {distorted.requires_grad}")

    decoded_logits = decoder(distorted)
    print(f"3. Decoded logits shape: {decoded_logits.shape}, requires_grad: {decoded_logits.requires_grad}")

    # Compute message loss
    loss_msg = F.binary_cross_entropy_with_logits(decoded_logits, messages)
    print(f"\n4. Message loss: {loss_msg.item():.4f}")

    # Backward pass
    loss_msg.backward()

    # Check encoder gradients
    print("\n" + "-" * 40)
    print("Encoder Parameter Gradients:")
    print("-" * 40)
    encoder_grad_stats = []
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            nonzero = (param.grad.abs() > 1e-10).float().mean().item()
            encoder_grad_stats.append((name, grad_mean, grad_max, nonzero))

    # Show first few and last few layers
    for name, grad_mean, grad_max, nonzero in encoder_grad_stats[:3]:
        print(f"  {name}: mean={grad_mean:.2e}, max={grad_max:.2e}, nonzero={nonzero*100:.1f}%")
    print("  ...")
    for name, grad_mean, grad_max, nonzero in encoder_grad_stats[-3:]:
        print(f"  {name}: mean={grad_mean:.2e}, max={grad_max:.2e}, nonzero={nonzero*100:.1f}%")

    # Check decoder gradients
    print("\n" + "-" * 40)
    print("Decoder Parameter Gradients:")
    print("-" * 40)
    decoder_grad_stats = []
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            nonzero = (param.grad.abs() > 1e-10).float().mean().item()
            decoder_grad_stats.append((name, grad_mean, grad_max, nonzero))

    for name, grad_mean, grad_max, nonzero in decoder_grad_stats[:3]:
        print(f"  {name}: mean={grad_mean:.2e}, max={grad_max:.2e}, nonzero={nonzero*100:.1f}%")
    print("  ...")
    for name, grad_mean, grad_max, nonzero in decoder_grad_stats[-3:]:
        print(f"  {name}: mean={grad_mean:.2e}, max={grad_max:.2e}, nonzero={nonzero*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    encoder_has_grads = any(p.grad is not None and p.grad.abs().max() > 1e-10
                           for p in encoder.parameters())
    decoder_has_grads = any(p.grad is not None and p.grad.abs().max() > 1e-10
                           for p in decoder.parameters())

    print(f"Encoder receives gradients from message loss: {'✅ YES' if encoder_has_grads else '❌ NO'}")
    print(f"Decoder receives gradients from message loss: {'✅ YES' if decoder_has_grads else '❌ NO'}")

    # Compute gradient magnitude ratio (encoder vs decoder)
    if encoder_has_grads and decoder_has_grads:
        enc_total_grad = sum(p.grad.abs().mean().item() for p in encoder.parameters() if p.grad is not None)
        dec_total_grad = sum(p.grad.abs().mean().item() for p in decoder.parameters() if p.grad is not None)
        ratio = enc_total_grad / dec_total_grad if dec_total_grad > 0 else float('inf')
        print(f"Encoder/Decoder gradient ratio: {ratio:.4f}")

        if ratio < 0.01:
            print("⚠️  WARNING: Encoder gradients are much smaller than decoder!")
            print("   This suggests gradient vanishing through distortions.")


def test_gradient_magnitude_through_chain():
    """Test how gradient magnitude changes through each component."""
    print("\n" + "=" * 60)
    print("Testing Gradient Magnitude Through Chain")
    print("=" * 60)

    device = torch.device("cpu")

    # Simple chain: x → distortion → y → loss
    distortions = [
        ("Identity", lambda x: x),
        ("GaussianNoise", GaussianNoise(intensity=1.0, std=0.02)),
        ("BrightnessHue", BrightnessHue(intensity=1.0, rnd_bri=0.3, rnd_hue=0.1)),
        ("Contrast", Contrast(intensity=1.0, contrast_low=0.5, contrast_high=1.5)),
        ("Saturation", Saturation(intensity=1.0, rnd_sat=1.0)),
        ("JPEGCompression Q50", JPEGCompression(intensity=1.0, quality=50)),
        ("JPEGCompression Q90", JPEGCompression(intensity=1.0, quality=90)),
        ("PerspectiveWarp", PerspectiveWarp(intensity=0.5, scale=0.1)),
    ]

    # Compose all
    full_compose = Compose([
        GaussianNoise(intensity=1.0, std=0.02),
        BrightnessHue(intensity=1.0, rnd_bri=0.3, rnd_hue=0.1),
        Contrast(intensity=1.0, contrast_low=0.5, contrast_high=1.5),
        Saturation(intensity=1.0, rnd_sat=1.0),
        JPEGCompression(intensity=1.0, quality=50),
        PerspectiveWarp(intensity=0.5, scale=0.1),
    ])
    distortions.append(("Full Compose", full_compose))

    print(f"\n{'Distortion':<25} {'Input Grad Mean':<18} {'Grad/Input Ratio':<18}")
    print("-" * 61)

    for name, distortion in distortions:
        # Create input
        x = torch.rand(2, 3, 64, 64, requires_grad=True)

        # Forward
        y = distortion(x)

        # Backward with unit gradient
        y.backward(torch.ones_like(y))

        grad_mean = x.grad.abs().mean().item()
        input_mean = x.abs().mean().item()
        ratio = grad_mean / input_mean if input_mean > 0 else 0

        print(f"{name:<25} {grad_mean:<18.6f} {ratio:<18.6f}")


def test_decoder_without_distortion():
    """Test if decoder learns without any distortions."""
    print("\n" + "=" * 60)
    print("Testing Decoder Learning WITHOUT Distortions")
    print("=" * 60)

    device = torch.device("cpu")
    num_bits = 100
    batch_size = 2

    encoder = Encoder(num_bits=num_bits).to(device)
    decoder = Decoder(num_bits=num_bits).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    # Fixed test batch
    images = torch.rand(batch_size, 3, 400, 400, device=device)
    messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

    print("\nTraining without distortions (should learn quickly):")
    for step in range(50):
        optimizer.zero_grad()

        encoded = encoder(images, messages)
        # NO DISTORTION
        decoded_logits = decoder(encoded)

        loss = F.binary_cross_entropy_with_logits(decoded_logits, messages)
        loss.backward()
        optimizer.step()

        # Compute bit accuracy
        with torch.no_grad():
            probs = torch.sigmoid(decoded_logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == messages).float().mean().item()

        if step % 10 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, accuracy={accuracy*100:.1f}%")


def test_decoder_with_distortion():
    """Test if decoder learns with distortions."""
    print("\n" + "=" * 60)
    print("Testing Decoder Learning WITH Distortions")
    print("=" * 60)

    device = torch.device("cpu")
    num_bits = 100
    batch_size = 2

    encoder = Encoder(num_bits=num_bits).to(device)
    decoder = Decoder(num_bits=num_bits).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    # Mild distortions
    distortions = Compose([
        GaussianNoise(intensity=1.0, std=0.01),
        BrightnessHue(intensity=1.0, rnd_bri=0.1, rnd_hue=0.03),
    ])

    # Fixed test batch
    images = torch.rand(batch_size, 3, 400, 400, device=device)
    messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

    print("\nTraining with mild distortions:")
    for step in range(50):
        optimizer.zero_grad()

        encoded = encoder(images, messages)
        distorted = distortions(encoded)
        decoded_logits = decoder(distorted)

        loss = F.binary_cross_entropy_with_logits(decoded_logits, messages)
        loss.backward()
        optimizer.step()

        # Compute bit accuracy
        with torch.no_grad():
            probs = torch.sigmoid(decoded_logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == messages).float().mean().item()

        if step % 10 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, accuracy={accuracy*100:.1f}%")


if __name__ == "__main__":
    torch.manual_seed(42)

    test_distortion_gradients()
    test_gradient_magnitude_through_chain()
    test_full_pipeline_gradients()
    test_decoder_without_distortion()
    test_decoder_with_distortion()
