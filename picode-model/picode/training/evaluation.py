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
        decoder_size: int | None = None,
        frame_pct: float | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            encoder: Encoder model.
            decoder: Decoder model.
            device: Device to run evaluation on.
            lpips_fn: Optional LPIPS function for perceptual quality.
            decoder_size: If set, resize encoded images to this size before decoding.
                Used for PicodeLite where decoder_size < encoder_size.
            frame_pct: Frame width as fraction of image size. When set, uses
                PicodeFrame encode/decode flow (reflection padding, mask).
        """
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.lpips_fn = lpips_fn
        self.decoder_size = decoder_size
        self.frame_pct = frame_pct

    def _resize_for_decoder(self, images: Tensor) -> Tensor:
        """Resize images to decoder_size if configured and sizes differ."""
        if self.decoder_size is not None and images.shape[-1] != self.decoder_size:
            return F.interpolate(
                images,
                size=(self.decoder_size, self.decoder_size),
                mode="bilinear",
                align_corners=False,
            )
        return images

    def _encode_picodeframe(
        self, images: Tensor, messages: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode using PicodeFrame flow: extract inner, pad, encode with mask.

        Args:
            images: (B, C, H, W) images in [0, 1].
            messages: (B, num_bits) binary messages.

        Returns:
            Tuple of (encoded, mask, padded_inner) where mask is (B, 1, H, W)
            with 1 in center, 0 in border.
        """
        assert self.frame_pct is not None
        image_size = images.shape[-1]
        fw = int(self.frame_pct * image_size)

        inner = images[:, :, fw:image_size - fw, fw:image_size - fw]
        padded_inner = F.pad(inner, (fw, fw, fw, fw), mode="reflect")

        mask = torch.zeros(
            images.shape[0], 1, image_size, image_size,
            device=self.device, dtype=images.dtype,
        )
        mask[:, :, fw:image_size - fw, fw:image_size - fw] = 1.0

        encoded = self.encoder(padded_inner, messages, frame_width=fw).clamp(0.0, 1.0)
        return encoded, mask, padded_inner

    def _decode_picodeframe(
        self, images: Tensor, mask: Tensor,
    ) -> Tensor:
        """Decode PicodeFrame output with border mask.

        Args:
            images: (B, C, H, W) encoded (possibly distorted) images.
            mask: (B, 1, H, W) center mask (1=center, 0=border).

        Returns:
            Decoded logits (B, num_bits).
        """
        decoder_input = self._resize_for_decoder(images)
        return self.decoder(decoder_input, mask=mask)

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
        # Dispatch to PicoTier-specific evaluation if applicable
        if hasattr(self.encoder, "tier_embedding"):
            return self._evaluate_picotier(dataloader, max_batches)

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

            if self.frame_pct is not None:
                encoded, mask, padded_inner = self._encode_picodeframe(
                    images, messages,
                )
                decoded_logits = self._decode_picodeframe(encoded, mask)
            else:
                enc_out = self.encoder(images, messages)
                encoded = (enc_out["encoded"] if isinstance(enc_out, dict) else enc_out).clamp(0.0, 1.0)
                decoder_input = self._resize_for_decoder(encoded)
                decoded_logits = self.decoder(decoder_input)

            decoded_binary = (decoded_logits > 0).float()

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
    def _evaluate_picotier(
        self,
        dataloader: Any,
        max_batches: int | None = None,
    ) -> EvalMetrics:
        """Run per-tier evaluation for PicoTier encoder/decoder.

        Evaluates all tiers for each batch, using tier-specific message lengths
        (zero-padded to MAX_BITS) and masked bit accuracy over the active bits only.

        Args:
            dataloader: DataLoader yielding image batches (tensors or (image, ...) tuples).
            max_batches: Maximum number of batches to evaluate.

        Returns:
            EvalMetrics averaged across all tiers (message_accuracy and ssim set to 0.0
            as they are not meaningful for variable-length messages).
        """
        from picode.models.picotier.tiers import MAX_BITS, NUM_TIERS, TIERS

        self.encoder.eval()
        self.decoder.eval()

        tier_metrics: dict[int, dict[str, list[float]]] = {
            i: {"bit_acc": [], "psnr": []} for i in range(NUM_TIERS)
        }

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            images = (
                batch[0].to(self.device)
                if isinstance(batch, (list, tuple))
                else batch.to(self.device)
            )

            for tier_idx in range(NUM_TIERS):
                n_bits = int(TIERS[tier_idx]["bits"])
                batch_size = images.shape[0]

                # Generate messages: random bits for active positions, zeros for padding
                messages = torch.zeros(batch_size, MAX_BITS, device=self.device)
                messages[:, :n_bits] = torch.randint(
                    0, 2, (batch_size, n_bits), device=self.device
                ).float()
                tiers = torch.full(
                    (batch_size,), tier_idx, device=self.device, dtype=torch.long
                )

                # Encode
                enc_out = self.encoder(images, messages, tiers)
                encoded = (
                    enc_out["encoded"] if isinstance(enc_out, dict) else enc_out
                ).clamp(0.0, 1.0)

                # PSNR against original
                mse = F.mse_loss(encoded, images)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

                # Decode (resize if needed)
                decoder_input = self._resize_for_decoder(encoded)
                decoded_logits, tier_logits = self.decoder(decoder_input, tier=tiers)

                # Masked bit accuracy: only the first n_bits positions are meaningful
                decoded_bits = (torch.sigmoid(decoded_logits[:, :n_bits]) > 0.5).float()
                bit_acc = (decoded_bits == messages[:, :n_bits]).float().mean()

                tier_metrics[tier_idx]["bit_acc"].append(bit_acc.item())
                tier_metrics[tier_idx]["psnr"].append(psnr.item())

        # Average across tiers
        overall_bit_acc = 0.0
        overall_psnr = 0.0
        for i in range(NUM_TIERS):
            n = len(tier_metrics[i]["bit_acc"])
            if n > 0:
                overall_bit_acc += sum(tier_metrics[i]["bit_acc"]) / n
                overall_psnr += sum(tier_metrics[i]["psnr"]) / n

        overall_bit_acc /= NUM_TIERS
        overall_psnr /= NUM_TIERS

        return EvalMetrics(
            bit_accuracy=overall_bit_acc,
            message_accuracy=0.0,  # Not meaningful for variable-length messages
            psnr=overall_psnr,
            ssim=0.0,
            lpips=None,
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

        if self.frame_pct is not None:
            encoded, mask, _ = self._encode_picodeframe(images, messages)
        else:
            enc_out = self.encoder(images, messages)
            encoded = (enc_out["encoded"] if isinstance(enc_out, dict) else enc_out).clamp(0.0, 1.0)
            mask = None

        for name, strengths in distortions.items():
            for strength in strengths:
                distortion = self._create_distortion(name, strength)
                distorted = distortion(encoded)

                if mask is not None:
                    decoded_logits = self._decode_picodeframe(distorted, mask)
                else:
                    decoder_input = self._resize_for_decoder(distorted)
                    decoded_logits = self.decoder(decoder_input)
                decoded_binary = (decoded_logits > 0).float()

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
