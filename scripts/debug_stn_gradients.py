#!/usr/bin/env python3
"""Debug STN gradient flow issue."""

import torch
import torch.nn.functional as F

from picode.models.stegastamp import Decoder


def test_stn_gradient_flow():
    """Test if STN gradients flow after weight update."""
    print("=" * 60)
    print("Testing STN Gradient Flow")
    print("=" * 60)

    decoder = Decoder(num_bits=100)

    # Check initial stn_fc_weight
    print(f"\nInitial stn_fc_weight:")
    print(f"  mean: {decoder.stn_fc_weight.abs().mean().item():.6f}")
    print(f"  max:  {decoder.stn_fc_weight.abs().max().item():.6f}")
    print(f"  all zeros: {(decoder.stn_fc_weight == 0).all().item()}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

    # Simulate a few training steps
    for step in range(5):
        optimizer.zero_grad()

        # Random input
        x = torch.rand(2, 3, 400, 400)
        target = torch.randint(0, 2, (2, 100)).float()

        logits = decoder(x)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        # Check stn_params.0 gradient
        stn_conv1_grad = decoder.stn_params[0].weight.grad
        stn_fc_weight_grad = decoder.stn_fc_weight.grad

        print(f"\nStep {step}:")
        print(f"  stn_fc_weight grad mean: {stn_fc_weight_grad.abs().mean().item():.6f}")
        print(f"  stn_conv1 grad mean: {stn_conv1_grad.abs().mean().item():.6f}")
        print(f"  stn_conv1 grad nonzero: {(stn_conv1_grad.abs() > 1e-10).float().mean().item()*100:.1f}%")

        optimizer.step()

        # Check stn_fc_weight after update
        print(f"  stn_fc_weight after update: mean={decoder.stn_fc_weight.abs().mean().item():.6f}")


def test_fix_with_small_init():
    """Test if small random initialization fixes the issue."""
    print("\n" + "=" * 60)
    print("Testing with Small Random Initialization")
    print("=" * 60)

    decoder = Decoder(num_bits=100)

    # Fix: Initialize stn_fc_weight with small random values
    with torch.no_grad():
        decoder.stn_fc_weight.data = torch.randn_like(decoder.stn_fc_weight) * 0.01

    print(f"\nFixed stn_fc_weight:")
    print(f"  mean: {decoder.stn_fc_weight.abs().mean().item():.6f}")
    print(f"  max:  {decoder.stn_fc_weight.abs().max().item():.6f}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

    for step in range(5):
        optimizer.zero_grad()

        x = torch.rand(2, 3, 400, 400)
        target = torch.randint(0, 2, (2, 100)).float()

        logits = decoder(x)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        stn_conv1_grad = decoder.stn_params[0].weight.grad

        print(f"\nStep {step}:")
        print(f"  stn_conv1 grad mean: {stn_conv1_grad.abs().mean().item():.6f}")
        print(f"  stn_conv1 grad nonzero: {(stn_conv1_grad.abs() > 1e-10).float().mean().item()*100:.1f}%")

        optimizer.step()


def compare_learning_with_and_without_stn():
    """Compare learning speed with functional vs broken STN."""
    print("\n" + "=" * 60)
    print("Comparing Learning: Zero Init vs Small Init")
    print("=" * 60)

    num_steps = 30

    # Test 1: Original (zero init)
    print("\n--- Zero Init (Original) ---")
    decoder_zero = Decoder(num_bits=100)
    optimizer_zero = torch.optim.Adam(decoder_zero.parameters(), lr=0.001)

    # Fixed data
    torch.manual_seed(42)
    x_fixed = torch.rand(2, 3, 400, 400)
    target_fixed = torch.randint(0, 2, (2, 100)).float()

    for step in range(num_steps):
        optimizer_zero.zero_grad()
        logits = decoder_zero(x_fixed)
        loss = F.binary_cross_entropy_with_logits(logits, target_fixed)
        loss.backward()
        optimizer_zero.step()

        if step % 10 == 0:
            acc = ((torch.sigmoid(logits) > 0.5).float() == target_fixed).float().mean()
            print(f"  Step {step:2d}: loss={loss.item():.4f}, acc={acc.item()*100:.1f}%")

    # Test 2: Small random init
    print("\n--- Small Random Init (Fixed) ---")
    decoder_small = Decoder(num_bits=100)
    with torch.no_grad():
        decoder_small.stn_fc_weight.data = torch.randn_like(decoder_small.stn_fc_weight) * 0.01
    optimizer_small = torch.optim.Adam(decoder_small.parameters(), lr=0.001)

    for step in range(num_steps):
        optimizer_small.zero_grad()
        logits = decoder_small(x_fixed)
        loss = F.binary_cross_entropy_with_logits(logits, target_fixed)
        loss.backward()
        optimizer_small.step()

        if step % 10 == 0:
            acc = ((torch.sigmoid(logits) > 0.5).float() == target_fixed).float().mean()
            print(f"  Step {step:2d}: loss={loss.item():.4f}, acc={acc.item()*100:.1f}%")


if __name__ == "__main__":
    test_stn_gradient_flow()
    test_fix_with_small_init()
    compare_learning_with_and_without_stn()
