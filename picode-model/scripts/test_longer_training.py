#!/usr/bin/env python3
"""Test longer training to see if model eventually converges."""

import torch
import torch.nn.functional as F

from picode.models.stegastamp import Encoder, Decoder
from picode.distortions.native import (
    Compose,
    GaussianNoise,
    BrightnessHue,
)


def test_convergence(num_steps: int = 500, use_distortions: bool = True, fixed_messages: bool = False):
    """Test training convergence."""
    print(f"\n{'=' * 60}")
    print(f"Training for {num_steps} steps")
    print(f"Distortions: {use_distortions}, Fixed messages: {fixed_messages}")
    print(f"{'=' * 60}")

    device = torch.device("cpu")
    num_bits = 100
    batch_size = 2

    encoder = Encoder(num_bits=num_bits).to(device)
    decoder = Decoder(num_bits=num_bits).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    if use_distortions:
        distortions = Compose([
            GaussianNoise(intensity=1.0, std=0.01),
            BrightnessHue(intensity=1.0, rnd_bri=0.1, rnd_hue=0.03),
        ])
    else:
        distortions = None

    # Fixed images
    torch.manual_seed(42)
    images = torch.rand(batch_size, 3, 400, 400, device=device)

    # Fixed or random messages
    if fixed_messages:
        messages_fixed = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

    print(f"\n{'Step':<8} {'Loss':<12} {'Msg Loss':<12} {'Accuracy':<12}")
    print("-" * 44)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Get messages
        if fixed_messages:
            messages = messages_fixed
        else:
            messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

        # Forward
        encoded = encoder(images, messages)

        if distortions:
            distorted = distortions(encoded)
        else:
            distorted = encoded

        decoded_logits = decoder(distorted)

        # Loss
        loss_msg = F.binary_cross_entropy_with_logits(decoded_logits, messages)
        loss_l2 = F.mse_loss(encoded, images)
        loss = loss_msg + loss_l2

        loss.backward()
        optimizer.step()

        # Accuracy
        with torch.no_grad():
            probs = torch.sigmoid(decoded_logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == messages).float().mean().item()

        if step % 50 == 0:
            print(f"{step:<8} {loss.item():<12.4f} {loss_msg.item():<12.4f} {accuracy*100:<12.1f}%")

    # Final evaluation
    print(f"\nFinal: loss={loss.item():.4f}, msg_loss={loss_msg.item():.4f}, accuracy={accuracy*100:.1f}%")

    return accuracy


if __name__ == "__main__":
    # Test 1: No distortions, random messages
    test_convergence(500, use_distortions=False, fixed_messages=False)

    # Test 2: With distortions, random messages
    test_convergence(500, use_distortions=True, fixed_messages=False)

    # Test 3: With distortions, fixed messages
    test_convergence(500, use_distortions=True, fixed_messages=True)
