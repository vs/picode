"""Main Trainer class for StegaStamp training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from picode.models.base import Decoder as BaseDecoder
from picode.models.base import Encoder as BaseEncoder
from picode.models.stegastamp import Decoder as StegaDecoder
from picode.models.stegastamp import Encoder as StegaEncoder
from picode.training.checkpointing import Checkpointer
from picode.training.config import (
    Config,
    DelayedLossRamp,
    _dict_to_config,
    load_config,
)
from picode.training.data import create_dataloader
from picode.training.distortion_strategy import create_distortion_strategy
from picode.training.evaluation import (
    DEFAULT_ROBUSTNESS_SWEEP,
    EvalMetrics,
    Evaluator,
    RobustnessResult,
)
from picode.training.logging import CompositeLogger, create_logger


def rgb_to_yuv(rgb: Tensor) -> Tensor:
    """Convert RGB tensor to YUV color space.

    Uses BT.601 conversion matrix (same as tf.image.rgb_to_yuv).

    Args:
        rgb: (B, 3, H, W) tensor in [0, 1] with channels R, G, B.

    Returns:
        (B, 3, H, W) tensor with channels Y, U, V.
    """
    # BT.601 conversion matrix
    # Y =  0.299*R + 0.587*G + 0.114*B
    # U = -0.147*R - 0.289*G + 0.436*B (Cb)
    # V =  0.615*R - 0.515*G - 0.100*B (Cr)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b

    return torch.cat([y, u, v], dim=1)


def compute_yuv_l2_loss(
    original: Tensor, encoded: Tensor, yuv_weights: list[float] | tuple[float, float, float]
) -> Tensor:
    """Compute YUV-weighted L2 loss.

    Converts images to YUV and computes weighted MSE per channel.
    Original StegaStamp uses Y=1, U=100, V=100 to heavily penalize
    color shifts while allowing more luma changes.

    Args:
        original: (B, 3, H, W) original image in [0, 1].
        encoded: (B, 3, H, W) encoded image in [0, 1].
        yuv_weights: [Y_weight, U_weight, V_weight].

    Returns:
        Scalar loss tensor.
    """
    original_yuv = rgb_to_yuv(original)
    encoded_yuv = rgb_to_yuv(encoded)

    diff = encoded_yuv - original_yuv
    mse_per_channel = (diff ** 2).mean(dim=(0, 2, 3))  # (3,)

    weights = torch.tensor(list(yuv_weights), device=original.device, dtype=original.dtype)
    weighted_loss = (mse_per_channel * weights).sum() / weights.sum()

    return weighted_loss


class Trainer:
    """Main trainer class that orchestrates training.

    Handles:
    - Model initialization
    - Data loading
    - Distortion application
    - Loss computation with ramping
    - Checkpointing
    - Logging
    - Evaluation

    Args:
        config: Training configuration.
    """

    def __init__(self, config: Config) -> None:
        """Initialize trainer from configuration.

        Args:
            config: Complete training configuration.
        """
        self.config = config

        # Set up device (prefer CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Create StegaStamp models
        num_bits = config.training.num_bits
        self.encoder: BaseEncoder = StegaEncoder(num_bits=num_bits).to(self.device)
        self.decoder: BaseDecoder = StegaDecoder(num_bits=num_bits).to(self.device)

        # Create optimizer with optional separate learning rates
        encoder_lr = config.training.lr * config.training.encoder_lr_scale
        decoder_lr = config.training.lr
        param_groups = [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.decoder.parameters(), "lr": decoder_lr},
        ]
        self.optimizer = torch.optim.Adam(param_groups)
        self.scheduler = None  # No scheduler by default

        # Create dataloader
        self.dataloader = create_dataloader(config.data, config.training.image_size)
        self._data_iter: Iterator[Tensor] | None = None

        # Create distortion strategy
        self.distortion = create_distortion_strategy(config.distortion)

        # Create logger
        self.logger: CompositeLogger = create_logger(config.logging, config.experiment_name)

        # Create checkpointer
        self.checkpointer = Checkpointer(config.checkpoint, config.experiment_name)

        # Create evaluator
        self.evaluator = Evaluator(self.encoder, self.decoder, self.device)

        # Create discriminator if GAN training enabled
        self.discriminator: nn.Module | None = None
        self.d_optimizer: torch.optim.RMSprop | None = None

        if config.loss.gan_config.enabled:
            from picode.models.stegastamp.discriminator import Discriminator

            self.discriminator = Discriminator().to(self.device)
            self.d_optimizer = torch.optim.RMSprop(
                self.discriminator.parameters(), lr=config.loss.gan_config.discriminator_lr
            )

        # Training state
        self.global_step = 0
        self.best_metric = float("inf")

        # Optional LPIPS loss (lazy loaded)
        self._lpips_fn: nn.Module | None = None

    @classmethod
    def from_config(cls, path: str, overrides: dict[str, Any] | None = None) -> Trainer:
        """Create trainer from YAML config file.

        Args:
            path: Path to YAML config file.
            overrides: Optional dict of overrides to apply.

        Returns:
            Initialized Trainer.
        """
        config = load_config(path, overrides)
        return cls(config)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> Trainer:
        """Create trainer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Trainer with restored state.
        """
        data = torch.load(checkpoint_path, weights_only=False)

        # Reconstruct config
        config = _dict_to_config(data["config"])

        # Create trainer
        trainer = cls(config)

        # Restore model states
        trainer.encoder.load_state_dict(data["encoder_state"])
        trainer.decoder.load_state_dict(data["decoder_state"])
        trainer.optimizer.load_state_dict(data["optimizer_state"])

        # Restore training state
        trainer.global_step = data["step"]
        trainer.best_metric = data["best_metric"]

        return trainer

    def fit(self) -> None:
        """Run the main training loop.

        Trains until num_steps is reached, handling:
        - Batch iteration with automatic restart
        - Periodic logging
        - Periodic checkpointing
        - Final checkpoint at end
        """
        self.encoder.train()
        self.decoder.train()

        num_steps = self.config.training.num_steps
        log_every = self.config.logging.log_every_steps

        while self.global_step < num_steps:
            # Get next batch
            images = self._get_batch()

            # Train step
            metrics = self._train_step(images)

            # Log metrics
            if self.global_step % log_every == 0:
                self.logger.log_scalars(metrics, self.global_step)

            # Checkpoint
            if self.checkpointer.should_save(self.global_step):
                self.checkpointer.save(
                    step=self.global_step,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    optimizer=self.optimizer,
                    scheduler=None,
                    config=self.config,
                    metrics=metrics,
                )

            self.global_step += 1

        # Final checkpoint
        final_metrics = {"loss": 0.0}  # Placeholder
        self.checkpointer.save(
            step=self.global_step,
            encoder=self.encoder,
            decoder=self.decoder,
            optimizer=self.optimizer,
            scheduler=None,
            config=self.config,
            metrics=final_metrics,
        )

        self.logger.close()

    def _get_batch(self) -> Tensor:
        """Get next batch from dataloader, restarting if exhausted.

        Returns:
            Batch of images on device.
        """
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)

        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            batch = next(self._data_iter)

        return batch.to(self.device)

    def _train_step(self, images: Tensor) -> dict[str, float]:
        """Execute a single training step.

        Args:
            images: Batch of images (B, C, H, W) in [0, 1].

        Returns:
            Dict of metrics for this step.
        """
        batch_size = images.shape[0]
        num_bits = self.config.training.num_bits

        # Generate random messages
        messages = torch.randint(0, 2, (batch_size, num_bits), device=self.device).float()

        # Forward pass
        encoded = self.encoder(images, messages)
        distorted = self.distortion(encoded, self.global_step)
        decoded_logits = self.decoder(distorted)

        # Compute residual for diagnostics (encoded - original)
        residual = encoded - images

        metrics: dict[str, float] = {}

        # Compute losses (decoder outputs logits, loss uses BCE with logits)
        losses = self._compute_ramped_losses(images, encoded, messages, decoded_logits)

        # Warmup phase: only use scaled message loss
        if self.global_step < self.config.training.warmup_steps:
            msg_scale = self._ramp(
                self.config.loss.message.scale,
                self.config.loss.message.ramp_steps,
                self.global_step,
            )
            total_loss = msg_scale * losses["loss_msg"]
        else:
            total_loss = losses["loss"]

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore[no-untyped-call]

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=10.0)

        self.optimizer.step()

        # GAN training step (if enabled)
        loss_D = torch.tensor(0.0, device=self.device)

        if (
            self.discriminator is not None
            and self.d_optimizer is not None
            and self.config.loss.gan_config.enabled
        ):
            # Discriminator step
            self.d_optimizer.zero_grad()

            # Real images
            d_real = self.discriminator(images)
            # Fake (encoded) images - detach to not backprop through encoder
            d_fake = self.discriminator(encoded.detach())

            # WGAN loss: maximize D(real) - D(fake)
            # Discriminator wants: D(real) high, D(fake) low
            # So minimize: D(fake) - D(real)
            loss_D = d_fake.mean() - d_real.mean()
            loss_D.backward()

            self.d_optimizer.step()

            # Clip discriminator weights (WGAN)
            clip_val = self.config.loss.gan_config.clip_weights
            for p in self.discriminator.parameters():
                p.data.clamp_(-clip_val, clip_val)

        # Add GAN metrics
        if self.discriminator is not None:
            metrics["loss_D"] = loss_D.item() if isinstance(loss_D, Tensor) else loss_D

        # Convert to float metrics
        metrics.update({k: v.item() for k, v in losses.items()})

        # Diagnostic metrics for debugging training issues
        # Residual statistics (encoder output - original image)
        metrics["residual_mean"] = residual.mean().item()
        metrics["residual_std"] = residual.std().item()
        metrics["residual_abs_max"] = residual.abs().max().item()

        # Decoder output distribution (check for trivial solution)
        with torch.no_grad():
            decoded_probs = torch.sigmoid(decoded_logits)
            metrics["decoder_prob_mean"] = decoded_probs.mean().item()
            metrics["decoder_prob_std"] = decoded_probs.std().item()

            # Bit accuracy (how many bits are correct)
            predicted_bits = (decoded_probs > 0.5).float()
            bit_accuracy = (predicted_bits == messages).float().mean().item()
            metrics["bit_accuracy"] = bit_accuracy

        return metrics

    def _compute_ramped_losses(
        self,
        original: Tensor,
        encoded: Tensor,
        messages: Tensor,
        decoded_logits: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all losses with ramping applied.

        Args:
            original: Original images (B, C, H, W).
            encoded: Encoded images (B, C, H, W).
            messages: Original messages (B, num_bits).
            decoded_logits: Decoded message logits (B, num_bits) - NOT probabilities.

        Returns:
            Dict with loss tensors:
            - loss: Total weighted loss
            - loss_msg: Message BCE loss
            - loss_l2: Image L2 loss
            - loss_lpips: LPIPS loss (if available)
            - loss_edge: Edge-weighted loss (after delay)
        """
        loss_cfg = self.config.loss
        step = self.global_step
        no_im_loss_steps = self.config.training.no_im_loss_steps

        # Message loss
        msg_scale = self._ramp(loss_cfg.message.scale, loss_cfg.message.ramp_steps, step)
        if loss_cfg.message_loss_type == "mse":
            # MSE loss: avoids trivial solution (predicting 0.5 for all bits)
            # Apply sigmoid to get probabilities, then MSE against binary targets
            decoded_probs = torch.sigmoid(decoded_logits)
            loss_msg = F.mse_loss(decoded_probs, messages)
        else:
            # BCE loss (default): standard binary cross-entropy with logits
            loss_msg = F.binary_cross_entropy_with_logits(decoded_logits, messages)
        weighted_msg = msg_scale * loss_msg

        # During no_im_loss_steps phase, only train on message loss (like StegaStamp)
        # This forces encoder-decoder to first learn message encoding before image quality
        skip_image_loss = step < no_im_loss_steps
        # Image loss ramps start AFTER no_im_loss_steps phase
        effective_step = max(0, step - no_im_loss_steps)

        # L2 loss with YUV weighting (matches original StegaStamp)
        yuv_weights = loss_cfg.yuv_weights
        if yuv_weights != (1.0, 1.0, 1.0):
            loss_l2 = compute_yuv_l2_loss(original, encoded, yuv_weights)
        else:
            loss_l2 = F.mse_loss(encoded, original)
        if skip_image_loss:
            weighted_l2 = torch.tensor(0.0, device=loss_l2.device)
        else:
            l2_scale = self._ramp(loss_cfg.l2.scale, loss_cfg.l2.ramp_steps, effective_step)
            weighted_l2 = l2_scale * loss_l2

        # Start with message and l2 loss
        total = weighted_msg + weighted_l2

        losses: dict[str, Tensor] = {
            "loss_msg": loss_msg,
            "loss_l2": loss_l2,
        }

        # LPIPS loss (optional) - also skipped during no_im_loss_steps
        if skip_image_loss:
            lpips_scale = 0.0
        else:
            lpips_scale = self._ramp(
                loss_cfg.lpips.scale, loss_cfg.lpips.ramp_steps, effective_step
            )
        if lpips_scale > 0 and self._get_lpips_fn() is not None:
            loss_lpips = self._compute_lpips(original, encoded)
            weighted_lpips = lpips_scale * loss_lpips
            total = total + weighted_lpips
            losses["loss_lpips"] = loss_lpips

        # GAN generator loss (if enabled)
        if (
            not skip_image_loss
            and hasattr(self, "discriminator")
            and self.discriminator is not None
        ):
            g_scale = self._ramp(
                self.config.loss.gan_config.g_loss_scale,
                self.config.loss.gan_config.g_loss_ramp_steps,
                effective_step,
            )
            if g_scale > 0:
                d_fake = self.discriminator(encoded)
                loss_G = -d_fake.mean()  # Maximize D(fake) = minimize -D(fake)
                weighted_G = g_scale * loss_G
                total = total + weighted_G
                losses["loss_G"] = loss_G

        # Edge loss (after delay) - skip during no_im_loss_steps
        if not skip_image_loss and effective_step >= loss_cfg.l2_edge_delay_steps:
            edge_step = effective_step - loss_cfg.l2_edge_delay_steps
            edge_scale = self._ramp(
                loss_cfg.l2_edge_gain, loss_cfg.l2_edge_ramp_steps, edge_step
            )
            if edge_scale > 0:
                loss_edge = self._compute_edge_loss(original, encoded)
                weighted_edge = edge_scale * loss_edge
                total = total + weighted_edge
                losses["loss_edge"] = loss_edge

        losses["loss"] = total
        return losses

    @staticmethod
    def _ramp(scale: float, ramp_steps: int, step: int) -> float:
        """Compute ramped scale at current step.

        Linear ramp from 0 to scale over ramp_steps.

        Args:
            scale: Target scale value.
            ramp_steps: Number of steps to ramp over.
            step: Current step.

        Returns:
            Ramped scale value.
        """
        if ramp_steps <= 0:
            return scale
        return min(scale * step / ramp_steps, scale)

    @staticmethod
    def _delayed_ramp(config: DelayedLossRamp, step: int) -> float:
        """Compute ramped scale with delay.

        Timeline: [0, delay) -> 0, [delay, delay + ramp) -> ramps to scale

        Args:
            config: DelayedLossRamp configuration.
            step: Current step.

        Returns:
            Ramped scale value (0 before delay).
        """
        if step < config.delay_steps:
            return 0.0
        adjusted_step = step - config.delay_steps
        if config.ramp_steps <= 0:
            return config.scale
        return min(config.scale * adjusted_step / config.ramp_steps, config.scale)

    def _compute_edge_loss(self, original: Tensor, encoded: Tensor) -> Tensor:
        """Compute edge-weighted L2 loss using Sobel edge detection.

        Higher weight on edges to preserve image structure.

        Args:
            original: Original images (B, C, H, W).
            encoded: Encoded images (B, C, H, W).

        Returns:
            Edge-weighted loss tensor.
        """
        # Convert to grayscale
        gray_orig = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, 2]
        gray_orig = gray_orig.unsqueeze(1)  # (B, 1, H, W)

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=original.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=original.device
        ).view(1, 1, 3, 3)

        # Compute gradients
        grad_x = F.conv2d(gray_orig, sobel_x, padding=1)
        grad_y = F.conv2d(gray_orig, sobel_y, padding=1)

        # Edge magnitude
        edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Normalize edge magnitude to [0, 1]
        edge_mag = edge_mag / (edge_mag.max() + 1e-8)

        # Weight the residual by edges
        residual = (encoded - original) ** 2  # (B, C, H, W)
        weighted_residual = residual * edge_mag  # Broadcast edge weights

        return weighted_residual.mean()

    def _get_lpips_fn(self) -> nn.Module | None:
        """Lazily load LPIPS function.

        Returns:
            LPIPS module or None if not available.
        """
        if self._lpips_fn is None:
            try:
                import lpips

                self._lpips_fn = lpips.LPIPS(net="alex").to(self.device)
                self._lpips_fn.eval()
            except ImportError:
                pass
        return self._lpips_fn

    def _compute_lpips(self, original: Tensor, encoded: Tensor) -> Tensor:
        """Compute LPIPS loss.

        Args:
            original: Original images (B, C, H, W) in [0, 1].
            encoded: Encoded images (B, C, H, W) in [0, 1].

        Returns:
            LPIPS loss tensor.
        """
        lpips_fn = self._get_lpips_fn()
        if lpips_fn is None:
            return torch.tensor(0.0, device=self.device)

        # LPIPS expects images in [-1, 1]
        orig_scaled = original * 2 - 1
        enc_scaled = encoded * 2 - 1

        # LPIPS network is in eval mode and not in optimizer, so its weights won't update.
        # Gradients flow through to the encoder, which is what we want.
        loss = lpips_fn(orig_scaled, enc_scaled)

        return cast(Tensor, loss.mean())

    def evaluate(self, dataloader: DataLoader[Tensor] | None = None) -> EvalMetrics:
        """Run evaluation on a dataset.

        Args:
            dataloader: Optional dataloader for evaluation.
                Uses training dataloader if not provided.

        Returns:
            EvalMetrics with evaluation results.
        """
        dl = dataloader if dataloader is not None else self.dataloader
        return self.evaluator.evaluate(dl, self.config.training.num_bits)

    def robustness_sweep(
        self, images: Tensor, messages: Tensor
    ) -> list[RobustnessResult]:
        """Test decoder robustness against various distortions.

        Args:
            images: Batch of images (B, C, H, W).
            messages: Batch of messages (B, num_bits).

        Returns:
            List of RobustnessResult for each distortion/strength combo.
        """
        return self.evaluator.robustness_sweep(images, messages, DEFAULT_ROBUSTNESS_SWEEP)
