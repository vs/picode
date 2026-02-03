"""Integration tests with distortions library."""

import torch

from picode.models.stegastamp import Decoder, Encoder, train_step


class TestDistortionIntegration:
    """Test encoder/decoder with distortions library."""

    def test_with_compose(self) -> None:
        """Works with Compose distortion pipeline."""
        from picode.distortions.native import Compose, GaussianNoise

        encoder = Encoder(num_bits=100)
        decoder = Decoder(num_bits=100)
        images = torch.rand(2, 3, 400, 400)

        distortion = Compose([
            GaussianNoise(intensity=0.1),
        ])

        losses = train_step(
            encoder=encoder,
            decoder=decoder,
            images=images,
            distortion=distortion,
            use_lpips=False,
        )

        assert losses["loss"] > 0

    def test_gradient_through_distortion(self) -> None:
        """Gradients flow through distortions."""
        from picode.distortions.native import GaussianNoise

        encoder = Encoder(num_bits=100)
        decoder = Decoder(num_bits=100)
        images = torch.rand(1, 3, 400, 400)
        messages = torch.randint(0, 2, (1, 100)).float()

        # Forward pass with distortion
        encoded = encoder(images, messages)
        distorted = GaussianNoise(intensity=0.2)(encoded)
        decoded = decoder(distorted)

        # Backward pass
        loss = decoded.sum()
        loss.backward()

        # Check gradients exist
        assert encoder.msg_fc.weight.grad is not None
        assert decoder.fc2.weight.grad is not None
