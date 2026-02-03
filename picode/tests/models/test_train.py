"""Tests for training utilities."""

import torch

from picode.models.stegastamp.decoder import Decoder
from picode.models.stegastamp.encoder import Encoder
from picode.models.stegastamp.train import StegaStampTrainer, train_step


class TestTrainStep:
    """Test single training step."""

    def test_returns_losses(self) -> None:
        """train_step returns loss dictionary."""
        encoder = Encoder(num_bits=100)
        decoder = Decoder(num_bits=100)
        images = torch.rand(2, 3, 400, 400)

        losses = train_step(
            encoder=encoder,
            decoder=decoder,
            images=images,
            distortion=None,
            use_lpips=False,
        )

        assert "loss" in losses
        assert "loss_msg" in losses
        assert "loss_l2" in losses

    def test_with_distortion(self) -> None:
        """train_step works with distortion function."""
        encoder = Encoder(num_bits=100)
        decoder = Decoder(num_bits=100)
        images = torch.rand(2, 3, 400, 400)

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        losses = train_step(
            encoder=encoder,
            decoder=decoder,
            images=images,
            distortion=identity,
            use_lpips=False,
        )

        assert losses["loss"] > 0


class TestStegaStampTrainer:
    """Test trainer class."""

    def test_initialization(self) -> None:
        """Trainer initializes encoder and decoder."""
        trainer = StegaStampTrainer(num_bits=100)
        assert trainer.encoder is not None
        assert trainer.decoder is not None

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces output."""
        trainer = StegaStampTrainer(num_bits=100)
        image = torch.rand(1, 3, 400, 400)
        message = torch.randint(0, 2, (1, 100)).float()

        encoded = trainer.encode(image, message)
        decoded = trainer.decode(encoded)

        assert encoded.shape == image.shape
        assert decoded.shape == message.shape
