#!/usr/bin/env python3
"""Debug encoder gradient magnitude issue."""

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


def analyze_gradient_flow_per_layer():
    """Analyze gradient magnitude at each layer of the pipeline."""
    print("=" * 70)
    print("Analyzing Gradient Flow Through Pipeline")
    print("=" * 70)

    device = torch.device("cpu")
    num_bits = 100
    batch_size = 2

    encoder = Encoder(num_bits=num_bits).to(device)
    decoder = Decoder(num_bits=num_bits).to(device)

    # Track intermediate tensors with hooks
    gradients = {}
    activations = {}

    def save_grad(name):
        def hook(grad):
            gradients[name] = grad.clone()
        return hook

    # Input
    images = torch.rand(batch_size, 3, 400, 400, device=device, requires_grad=True)
    messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

    # Forward through encoder
    encoded = encoder(images, messages)
    encoded.retain_grad()
    activations["encoded"] = encoded.clone()

    # Forward through distortions (one by one)
    distortion_outputs = [("encoded", encoded)]

    x = encoded
    distortion_list = [
        ("noise", GaussianNoise(intensity=1.0, std=0.01)),
        ("brightness_hue", BrightnessHue(intensity=1.0, rnd_bri=0.1, rnd_hue=0.03)),
        ("saturation", Saturation(intensity=1.0, rnd_sat=0.3)),
        ("contrast", Contrast(intensity=1.0, contrast_low=0.9, contrast_high=1.1)),
        ("jpeg", JPEGCompression(intensity=1.0, quality=90)),
        ("perspective", PerspectiveWarp(intensity=1.0, scale=0.02)),
    ]

    for name, distortion in distortion_list:
        x = distortion(x)
        x.retain_grad()
        distortion_outputs.append((name, x))
        activations[name] = x.clone()

    distorted = x

    # Forward through decoder
    decoded_logits = decoder(distorted)
    decoded_logits.retain_grad()

    # Compute loss
    loss = F.binary_cross_entropy_with_logits(decoded_logits, messages)

    # Backward
    loss.backward()

    # Analyze gradients
    print(f"\nLoss: {loss.item():.4f}")
    print("\n" + "-" * 70)
    print(f"{'Layer':<20} {'Grad Mean':<15} {'Grad Max':<15} {'Activation Mean':<15}")
    print("-" * 70)

    print(f"{'decoded_logits':<20} {decoded_logits.grad.abs().mean().item():<15.6f} "
          f"{decoded_logits.grad.abs().max().item():<15.6f} "
          f"{decoded_logits.abs().mean().item():<15.6f}")

    for name, tensor in reversed(distortion_outputs):
        if tensor.grad is not None:
            grad_mean = tensor.grad.abs().mean().item()
            grad_max = tensor.grad.abs().max().item()
            act_mean = tensor.abs().mean().item()
            print(f"{name:<20} {grad_mean:<15.6f} {grad_max:<15.6f} {act_mean:<15.6f}")

    if images.grad is not None:
        print(f"{'input_images':<20} {images.grad.abs().mean().item():<15.6f} "
              f"{images.grad.abs().max().item():<15.6f} "
              f"{images.abs().mean().item():<15.6f}")

    # Encoder parameter gradients
    print("\n" + "-" * 70)
    print("Encoder Parameter Gradients (by layer):")
    print("-" * 70)

    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            param_mean = param.abs().mean().item()
            ratio = grad_mean / param_mean if param_mean > 0 else 0
            print(f"  {name:<30} grad={grad_mean:.6f}, param={param_mean:.6f}, ratio={ratio:.6f}")


def test_gradient_with_varying_distortion_strength():
    """Test how distortion strength affects encoder gradients."""
    print("\n" + "=" * 70)
    print("Testing Encoder Gradients vs Distortion Strength")
    print("=" * 70)

    device = torch.device("cpu")
    num_bits = 100
    batch_size = 2

    strengths = [0.0, 0.1, 0.3, 0.5, 1.0]

    print(f"\n{'Strength':<12} {'Enc Grad Mean':<18} {'Dec Grad Mean':<18} {'Enc/Dec Ratio':<15}")
    print("-" * 63)

    for strength in strengths:
        encoder = Encoder(num_bits=num_bits).to(device)
        decoder = Decoder(num_bits=num_bits).to(device)

        images = torch.rand(batch_size, 3, 400, 400, device=device)
        messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

        # Forward
        encoded = encoder(images, messages)

        # Distortions with varying strength
        if strength > 0:
            distortions = Compose([
                GaussianNoise(intensity=strength, std=0.02),
                BrightnessHue(intensity=strength, rnd_bri=0.3, rnd_hue=0.1),
                JPEGCompression(intensity=strength, quality=50),
            ])
            distorted = distortions(encoded)
        else:
            distorted = encoded

        decoded_logits = decoder(distorted)
        loss = F.binary_cross_entropy_with_logits(decoded_logits, messages)
        loss.backward()

        # Compute total gradient magnitudes
        enc_grad = sum(p.grad.abs().mean().item() for p in encoder.parameters() if p.grad is not None)
        dec_grad = sum(p.grad.abs().mean().item() for p in decoder.parameters() if p.grad is not None)
        ratio = enc_grad / dec_grad if dec_grad > 0 else 0

        print(f"{strength:<12.1f} {enc_grad:<18.6f} {dec_grad:<18.6f} {ratio:<15.6f}")


def test_residual_magnitude():
    """Check if encoder is producing a meaningful residual."""
    print("\n" + "=" * 70)
    print("Checking Encoder Residual Magnitude")
    print("=" * 70)

    device = torch.device("cpu")
    num_bits = 100

    encoder = Encoder(num_bits=num_bits).to(device)

    images = torch.rand(2, 3, 400, 400, device=device)
    messages = torch.randint(0, 2, (2, num_bits), device=device).float()

    encoded = encoder(images, messages)

    # Compute residual
    residual = encoded - images

    print(f"\nImage stats:    mean={images.mean().item():.4f}, std={images.std().item():.4f}")
    print(f"Encoded stats:  mean={encoded.mean().item():.4f}, std={encoded.std().item():.4f}")
    print(f"Residual stats: mean={residual.mean().item():.4f}, std={residual.std().item():.4f}")
    print(f"Residual range: min={residual.min().item():.4f}, max={residual.max().item():.4f}")
    print(f"Residual L2:    {(residual**2).mean().sqrt().item():.4f}")

    # Check if residual is visible
    print(f"\nResidual SNR (signal to noise): {images.std().item() / residual.std().item():.2f}")


def test_decoder_sensitivity():
    """Test how sensitive decoder is to small perturbations."""
    print("\n" + "=" * 70)
    print("Testing Decoder Sensitivity to Perturbations")
    print("=" * 70)

    device = torch.device("cpu")
    num_bits = 100

    decoder = Decoder(num_bits=num_bits).to(device)

    # Base image
    images = torch.rand(2, 3, 400, 400, device=device)

    # Get base prediction
    with torch.no_grad():
        base_logits = decoder(images)
        base_probs = torch.sigmoid(base_logits)

    # Test with different perturbation magnitudes
    perturbations = [0.001, 0.01, 0.05, 0.1, 0.2]

    print(f"\n{'Perturbation':<15} {'Logit Change':<18} {'Prob Change':<18}")
    print("-" * 51)

    for pert in perturbations:
        noise = torch.randn_like(images) * pert
        perturbed = (images + noise).clamp(0, 1)

        with torch.no_grad():
            pert_logits = decoder(perturbed)
            pert_probs = torch.sigmoid(pert_logits)

        logit_change = (pert_logits - base_logits).abs().mean().item()
        prob_change = (pert_probs - base_probs).abs().mean().item()

        print(f"{pert:<15.3f} {logit_change:<18.4f} {prob_change:<18.4f}")


def test_learning_rate_scaling():
    """Test if different learning rates for encoder/decoder help."""
    print("\n" + "=" * 70)
    print("Testing Separate Learning Rates")
    print("=" * 70)

    device = torch.device("cpu")
    num_bits = 100
    batch_size = 2
    num_steps = 50

    # Distortions
    distortions = Compose([
        GaussianNoise(intensity=1.0, std=0.01),
        BrightnessHue(intensity=1.0, rnd_bri=0.1, rnd_hue=0.03),
    ])

    configs = [
        ("Same LR (1e-3)", 1e-3, 1e-3),
        ("Enc 10x LR", 1e-2, 1e-3),
        ("Enc 30x LR", 3e-2, 1e-3),
    ]

    for config_name, enc_lr, dec_lr in configs:
        print(f"\n--- {config_name} ---")

        encoder = Encoder(num_bits=num_bits).to(device)
        decoder = Decoder(num_bits=num_bits).to(device)

        optimizer = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": enc_lr},
            {"params": decoder.parameters(), "lr": dec_lr},
        ])

        # Fixed data for fair comparison
        torch.manual_seed(42)
        images = torch.rand(batch_size, 3, 400, 400, device=device)
        messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

        for step in range(num_steps):
            optimizer.zero_grad()

            encoded = encoder(images, messages)
            distorted = distortions(encoded)
            decoded_logits = decoder(distorted)

            loss = F.binary_cross_entropy_with_logits(decoded_logits, messages)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                with torch.no_grad():
                    acc = ((torch.sigmoid(decoded_logits) > 0.5).float() == messages).float().mean()
                print(f"  Step {step:2d}: loss={loss.item():.4f}, acc={acc.item()*100:.1f}%")


if __name__ == "__main__":
    torch.manual_seed(42)

    analyze_gradient_flow_per_layer()
    test_gradient_with_varying_distortion_strength()
    test_residual_magnitude()
    test_decoder_sensitivity()
    test_learning_rate_scaling()
