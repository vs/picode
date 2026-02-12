"""Main Trainer class for StegaStamp training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from picode.models.stegastamp import Decoder, Encoder
from picode.training.checkpointing import Checkpointer
from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DistortionConfig,
    DistortionRamp,
    LoggingConfig,
    LossConfig,
    LossRamp,
    TrainingConfig,
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


def _dict_to_config_full(data: dict[str, Any]) -> Config:
    """Convert nested dict to Config for checkpoint restoration.

    Args:
        data: Dictionary representation of Config.

    Returns:
        Reconstructed Config object.
    """
    # Handle DataConfig (required)
    data["data"] = DataConfig(**data["data"])

    # Handle TrainingConfig
    if "training" in data:
        data["training"] = TrainingConfig(**data["training"])

    # Handle LossConfig with nested LossRamps
    if "loss" in data:
        loss_data = data["loss"]
        for key in ["message", "l2", "lpips"]:
            if key in loss_data and isinstance(loss_data[key], dict):
                loss_data[key] = LossRamp(**loss_data[key])
        if "gan" in loss_data and loss_data["gan"] is not None:
            loss_data["gan"] = LossRamp(**loss_data["gan"])
        if "yuv_weights" in loss_data:
            loss_data["yuv_weights"] = tuple(loss_data["yuv_weights"])
        data["loss"] = LossConfig(**loss_data)

    # Handle DistortionConfig with nested DistortionRamps
    if "distortion" in data:
        dist_data = data["distortion"]
        for key in ["perspective", "brightness", "saturation", "hue", "noise", "jpeg_quality"]:
            if key in dist_data and isinstance(dist_data[key], dict):
                dist_data[key] = DistortionRamp(**dist_data[key])
        if "contrast" in dist_data:
            dist_data["contrast"] = tuple(dist_data["contrast"])
        data["distortion"] = DistortionConfig(**dist_data)

    # Handle CheckpointConfig
    if "checkpoint" in data:
        data["checkpoint"] = CheckpointConfig(**data["checkpoint"])

    # Handle LoggingConfig
    if "logging" in data:
        data["logging"] = LoggingConfig(**data["logging"])

    return Config(**data)


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

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create models
        num_bits = config.training.num_bits
        self.encoder = Encoder(num_bits=num_bits).to(self.device)
        self.decoder = Decoder(num_bits=num_bits).to(self.device)

        # Create optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.training.lr)

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
        config = _dict_to_config_full(data["config"])

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
        decoded = self.decoder(distorted)

        # Compute losses
        losses = self._compute_ramped_losses(images, encoded, messages, decoded)

        # Warmup phase: only use message loss
        if self.global_step < self.config.training.warmup_steps:
            total_loss = losses["loss_msg"]
        else:
            total_loss = losses["loss"]

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Convert to float metrics
        metrics = {k: v.item() for k, v in losses.items()}
        return metrics

    def _compute_ramped_losses(
        self,
        original: Tensor,
        encoded: Tensor,
        messages: Tensor,
        decoded: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all losses with ramping applied.

        Args:
            original: Original images (B, C, H, W).
            encoded: Encoded images (B, C, H, W).
            messages: Original messages (B, num_bits).
            decoded: Decoded messages (B, num_bits).

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

        # Message loss (BCE)
        msg_scale = self._ramp(loss_cfg.message.scale, loss_cfg.message.ramp_steps, step)
        loss_msg = F.binary_cross_entropy(decoded, messages)
        weighted_msg = msg_scale * loss_msg

        # L2 loss (MSE)
        l2_scale = self._ramp(loss_cfg.l2.scale, loss_cfg.l2.ramp_steps, step)
        loss_l2 = F.mse_loss(encoded, original)
        weighted_l2 = l2_scale * loss_l2

        # Start with message and l2 loss
        total = weighted_msg + weighted_l2

        losses: dict[str, Tensor] = {
            "loss_msg": loss_msg,
            "loss_l2": loss_l2,
        }

        # LPIPS loss (optional)
        lpips_scale = self._ramp(loss_cfg.lpips.scale, loss_cfg.lpips.ramp_steps, step)
        if lpips_scale > 0 and self._get_lpips_fn() is not None:
            loss_lpips = self._compute_lpips(original, encoded)
            weighted_lpips = lpips_scale * loss_lpips
            total = total + weighted_lpips
            losses["loss_lpips"] = loss_lpips

        # Edge loss (after delay)
        if step >= loss_cfg.l2_edge_delay_steps:
            edge_step = step - loss_cfg.l2_edge_delay_steps
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

        with torch.no_grad():
            # Detach to avoid tracking LPIPS gradients
            loss = lpips_fn(orig_scaled, enc_scaled)

        return loss.mean()

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
