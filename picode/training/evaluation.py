"""Evaluation utilities for training."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional import structural_similarity_index_measure

from picode.distortions.base import Distortion
from picode.distortions.native import (
    BrightnessHue,
    Contrast,
    GaussianBlur,
    GaussianNoise,
    JPEGCompression,
    Saturation,
)


@dataclass
class EvalMetrics:
    """Evaluation results."""

    bit_accuracy: float
    message_accuracy: float
    psnr: float
    ssim: float
    lpips: float | None


@dataclass
class RobustnessResult:
    """Results for a single distortion at one strength."""

    distortion: str
    strength: float
    bit_accuracy: float
    message_accuracy: float


DEFAULT_ROBUSTNESS_SWEEP: dict[str, list[float]] = {
    "jpeg": [90, 70, 50, 30, 10],
    "noise": [0.01, 0.02, 0.05, 0.1],
    "blur": [0.5, 1.0, 2.0, 3.0],
    "brightness": [0.1, 0.2, 0.3, 0.5],
    "contrast": [0.1, 0.2, 0.3, 0.5],
}


class Evaluator:
    """Evaluate encoder/decoder quality and robustness."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        lpips_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            encoder: Encoder model.
            decoder: Decoder model.
            device: Device to run evaluation on.
            lpips_fn: Optional LPIPS function for perceptual quality.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.lpips_fn = lpips_fn

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Any,
        num_bits: int,
        max_batches: int | None = None,
    ) -> EvalMetrics:
        """Run full evaluation on a dataset.

        Args:
            dataloader: DataLoader yielding image batches.
            num_bits: Number of bits in the message.
            max_batches: Maximum number of batches to evaluate.

        Returns:
            EvalMetrics with averaged results.
        """
        self.encoder.eval()
        self.decoder.eval()

        all_bit_acc: list[float] = []
        all_msg_acc: list[float] = []
        all_psnr: list[float] = []
        all_ssim: list[float] = []
        all_lpips: list[float] = []

        for i, images in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            images = images.to(self.device)
            messages = torch.randint(
                0, 2, (images.shape[0], num_bits), device=self.device
            ).float()

            encoded = self.encoder(images, messages)
            decoded = self.decoder(encoded)
            decoded_binary = (decoded > 0.5).float()

            # Message metrics
            bit_acc = (decoded_binary == messages).float().mean()
            msg_acc = (decoded_binary == messages).all(dim=1).float().mean()
            all_bit_acc.append(bit_acc.item())
            all_msg_acc.append(msg_acc.item())

            # Image metrics
            all_psnr.append(self._psnr(images, encoded).item())
            all_ssim.append(self._ssim(images, encoded).item())
            if self.lpips_fn:
                all_lpips.append(self._lpips(images, encoded).item())

        return EvalMetrics(
            bit_accuracy=sum(all_bit_acc) / len(all_bit_acc),
            message_accuracy=sum(all_msg_acc) / len(all_msg_acc),
            psnr=sum(all_psnr) / len(all_psnr),
            ssim=sum(all_ssim) / len(all_ssim),
            lpips=sum(all_lpips) / len(all_lpips) if all_lpips else None,
        )

    @torch.no_grad()
    def robustness_sweep(
        self,
        images: Tensor,
        messages: Tensor,
        distortions: dict[str, list[float]],
    ) -> list[RobustnessResult]:
        """Test decoder against distortions at various strengths.

        Args:
            images: Batch of images (B, C, H, W).
            messages: Batch of messages (B, num_bits).
            distortions: Dict mapping distortion names to list of strengths.

        Returns:
            List of RobustnessResult for each distortion/strength combo.
        """
        self.encoder.eval()
        self.decoder.eval()

        images = images.to(self.device)
        messages = messages.to(self.device)

        results: list[RobustnessResult] = []
        encoded = self.encoder(images, messages)

        for name, strengths in distortions.items():
            for strength in strengths:
                distortion = self._create_distortion(name, strength)
                distorted = distortion(encoded)

                decoded = self.decoder(distorted)
                decoded_binary = (decoded > 0.5).float()

                bit_acc = (decoded_binary == messages).float().mean().item()
                msg_acc = (decoded_binary == messages).all(dim=1).float().mean().item()

                results.append(
                    RobustnessResult(
                        distortion=name,
                        strength=strength,
                        bit_accuracy=bit_acc,
                        message_accuracy=msg_acc,
                    )
                )

        return results

    def _psnr(self, original: Tensor, encoded: Tensor) -> Tensor:
        """Compute PSNR between original and encoded images.

        Args:
            original: Original images (B, C, H, W).
            encoded: Encoded images (B, C, H, W).

        Returns:
            PSNR value in dB.
        """
        mse = F.mse_loss(encoded, original)
        if mse == 0:
            return torch.tensor(float("inf"))
        return 10 * torch.log10(1.0 / mse)

    def _ssim(self, original: Tensor, encoded: Tensor) -> Tensor:
        """Compute SSIM between original and encoded images.

        Args:
            original: Original images (B, C, H, W).
            encoded: Encoded images (B, C, H, W).

        Returns:
            SSIM value in [0, 1].
        """
        result = structural_similarity_index_measure(encoded, original)
        # structural_similarity_index_measure returns Tensor by default
        # (only returns tuple if return_full_image=True)
        assert isinstance(result, Tensor)
        return result

    def _lpips(self, original: Tensor, encoded: Tensor) -> Tensor:
        """Compute LPIPS between original and encoded images.

        Args:
            original: Original images (B, C, H, W).
            encoded: Encoded images (B, C, H, W).

        Returns:
            LPIPS value (lower is better).

        Raises:
            ValueError: If LPIPS function not provided.
        """
        if self.lpips_fn is None:
            raise ValueError("LPIPS function not provided")
        # LPIPS expects images in [-1, 1] range
        orig_scaled = original * 2 - 1
        enc_scaled = encoded * 2 - 1
        return self.lpips_fn(orig_scaled, enc_scaled).mean()

    def _create_distortion(self, name: str, strength: float) -> Distortion:
        """Create distortion by name and strength.

        Args:
            name: Distortion name (blur, noise, jpeg, brightness, contrast, saturation).
            strength: Distortion strength parameter.

        Returns:
            Configured Distortion instance.

        Raises:
            ValueError: If unknown distortion name.
        """
        if name == "blur":
            # GaussianBlur(intensity, kernel_size, sigma)
            return GaussianBlur(intensity=1.0, sigma=strength)
        elif name == "noise":
            # GaussianNoise(intensity, std)
            return GaussianNoise(intensity=1.0, std=strength)
        elif name == "jpeg":
            # JPEGCompression(intensity, quality)
            return JPEGCompression(intensity=1.0, quality=int(strength))
        elif name == "brightness":
            # BrightnessHue(intensity, rnd_bri, rnd_hue)
            return BrightnessHue(intensity=1.0, rnd_bri=strength, rnd_hue=0.0)
        elif name == "contrast":
            # Contrast(intensity, contrast_low, contrast_high)
            return Contrast(intensity=1.0, contrast_low=1 - strength, contrast_high=1 + strength)
        elif name == "saturation":
            # Saturation(intensity, rnd_sat)
            return Saturation(intensity=1.0, rnd_sat=strength)
        else:
            raise ValueError(f"Unknown distortion: {name}")
