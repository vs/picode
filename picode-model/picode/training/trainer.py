"""Main Trainer class for steganography model training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from picode.distortions.native.perspective import (
    get_identity_transform,
    get_rand_transform_matrix,
    perspective_transform,
)
from picode.models.base import Decoder as BaseDecoder
from picode.models.base import Encoder as BaseEncoder
from picode.models.factory import create_decoder, create_encoder
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


def _filter_compatible(
    state_dict: dict[str, Tensor], model: nn.Module,
) -> dict[str, Tensor]:
    """Filter state_dict to only include keys with matching shapes in model.

    Allows resuming from checkpoints with minor architecture changes
    (e.g. residual layer channel count). Mismatched keys are skipped
    and logged; the model keeps its initialized weights for those params.
    """
    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            print(f"  Skipping {k}: checkpoint {v.shape} != model {model_state[k].shape}")
        else:
            print(f"  Skipping {k}: not in model")
    skipped = set(model_state) - set(filtered)
    if skipped:
        print(f"  Freshly initialized: {sorted(skipped)}")
    return filtered


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

    # Original StegaStamp uses dot product without normalization
    weights = torch.tensor(list(yuv_weights), device=original.device, dtype=original.dtype)
    weighted_loss = (mse_per_channel * weights).sum()  # No normalization to match original

    return weighted_loss


def create_border_falloff_mask(
    height: int,
    width: int,
    falloff_speed: int = 4,
    device: torch.device | None = None,
) -> Tensor:
    """Create cosine-weighted border falloff mask matching original StegaStamp.

    The mask has value 0 at center, ramping up to 1 at borders using
    a cosine function. This penalizes perturbations at image edges more heavily.

    Args:
        height: Image height (e.g., 400).
        width: Image width (e.g., 400).
        falloff_speed: Controls falloff rate (default 4 = 25% border region).
        device: Target device.

    Returns:
        (1, 1, H, W) tensor for broadcasting with images.
    """
    falloff_im = np.ones((height, width), dtype=np.float32)

    # Vertical falloff (top and bottom borders)
    for i in range(height // falloff_speed):
        factor = (np.cos(4 * np.pi * i / height + np.pi) + 1) / 2
        falloff_im[-i - 1, :] *= factor
        falloff_im[i, :] *= factor

    # Horizontal falloff (left and right borders)
    for j in range(width // falloff_speed):
        factor = (np.cos(4 * np.pi * j / width + np.pi) + 1) / 2
        falloff_im[:, -j - 1] *= factor
        falloff_im[:, j] *= factor

    # Invert: now high values at borders
    falloff_im = 1 - falloff_im

    # Convert to tensor with broadcast dimensions
    mask = torch.from_numpy(falloff_im).unsqueeze(0).unsqueeze(0)
    if device is not None:
        mask = mask.to(device)

    return mask


def compute_yuv_l2_loss_with_falloff(
    original: Tensor,
    encoded: Tensor,
    yuv_weights: tuple[float, float, float],
    falloff_mask: Tensor,
    edge_gain: float = 10.0,
) -> Tensor:
    """Compute YUV-weighted L2 loss with border falloff mask.

    Matches original StegaStamp: im_diff += im_diff * falloff_im * edge_gain

    Args:
        original: (B, 3, H, W) original image in [0, 1].
        encoded: (B, 3, H, W) encoded image in [0, 1].
        yuv_weights: [Y_weight, U_weight, V_weight].
        falloff_mask: (1, 1, H, W) border falloff mask (0 center, 1 border).
        edge_gain: Amplification factor for border regions.

    Returns:
        Scalar loss tensor.
    """
    original_yuv = rgb_to_yuv(original)
    encoded_yuv = rgb_to_yuv(encoded)

    # Compute difference
    diff = encoded_yuv - original_yuv

    # Apply border falloff amplification (original formula: im_diff += im_diff * falloff_im)
    scaled_falloff = falloff_mask * edge_gain
    diff = diff + diff * scaled_falloff

    # Weighted MSE per channel
    mse_per_channel = (diff ** 2).mean(dim=(0, 2, 3))  # (3,)
    weights = torch.tensor(list(yuv_weights), device=original.device, dtype=original.dtype)
    weighted_loss = (mse_per_channel * weights).sum()

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

        # Seed RNGs for reproducibility
        if config.training.seed is not None:
            torch.manual_seed(config.training.seed)
            np.random.seed(config.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.training.seed)

        # Set up device (prefer CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Create models via factory
        num_bits = config.training.num_bits

        # Determine v2 encoder params
        _strength: float | None = None
        if config.training.residual_strength > 0:
            _strength = config.training.residual_strength
        _use_mask = config.loss.mask_reg is not None
        _max_res_amp = config.frame.residual_max_amplitude if config.frame else 0.0

        self.encoder: BaseEncoder = create_encoder(
            config.model, num_bits, strength=_strength, use_mask=_use_mask,
            max_residual_amplitude=_max_res_amp,
            residual_blur_sigma=config.training.residual_blur_sigma,
        ).to(self.device)
        self.decoder: BaseDecoder = create_decoder(config.model, num_bits).to(self.device)

        # Precompute decoder-input blur kernel (training-time only)
        self._decoder_blur_kernel: Tensor | None = None
        self._decoder_blur_pad: int = 0
        if config.training.decoder_blur_sigma > 0:
            import math
            sigma = config.training.decoder_blur_sigma
            k = 2 * math.ceil(3 * sigma) + 1
            ax = torch.arange(k, dtype=torch.float32) - k // 2
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
            kernel = (kernel / kernel.sum()).view(1, 1, k, k)
            # Expand to 3 channels (groups=3)
            self._decoder_blur_kernel = kernel.expand(3, -1, -1, -1).contiguous().to(self.device)
            self._decoder_blur_pad = k // 2

        # Determine image sizes based on model type
        # PicodeLite: encoder_size (800) for training images, decoder_size (320) for decoder input
        # PicoTrust: uses model.encoder_size/decoder_size (default 256x256)
        # StegaStamp/PicodeFrame: same size for both (training.image_size, typically 400)
        if config.model.type in ("picodelite", "picotrust", "picodetier"):
            train_image_size = config.model.encoder_size
            self._decoder_size = config.model.decoder_size
        elif config.model.type == "picodeframe":
            train_image_size = 400
            self._decoder_size = 400
        else:
            train_image_size = config.training.image_size
            self._decoder_size = config.training.image_size  # Same as encoder for StegaStamp

        # Create optimizer with optional separate learning rates
        encoder_lr = config.training.lr * config.training.encoder_lr_scale
        decoder_lr = config.training.lr

        if config.model.type in ("stegastamp", "picodeframe", "picotrust", "picodetier"):
            # StegaStamp, PicodeFrame, and PicoTrust have STN with separate LR
            stn_lr = config.training.lr * config.training.stn_lr_scale
            stn_param_names = {"stn_fc_weight", "stn_fc_bias"}
            stn_params = []
            decoder_params = []
            for name, param in self.decoder.named_parameters():
                if name in stn_param_names:
                    stn_params.append(param)
                else:
                    decoder_params.append(param)
            param_groups = [
                {"params": self.encoder.parameters(), "lr": encoder_lr},
                {"params": decoder_params, "lr": decoder_lr},
                {"params": stn_params, "lr": stn_lr},
            ]
        else:
            # PicodeLite: no STN, simpler param groups
            param_groups = [
                {"params": self.encoder.parameters(), "lr": encoder_lr},
                {"params": self.decoder.parameters(), "lr": decoder_lr},
            ]

        self.optimizer = torch.optim.Adam(param_groups)

        # Create LR scheduler
        if config.training.lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            eta_min = config.training.lr * config.training.lr_min_ratio
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = CosineAnnealingLR(
                self.optimizer, T_max=config.training.num_steps, eta_min=eta_min
            )
        else:
            self.scheduler = None

        # Create dataloader
        self.dataloader = create_dataloader(config.data, train_image_size)
        self._data_iter: Iterator[Tensor] | None = None

        # Create distortion strategy
        self.distortion = create_distortion_strategy(config.distortion)

        # Create logger
        self.logger: CompositeLogger = create_logger(config.logging, config.experiment_name)

        # Create checkpointer
        self.checkpointer = Checkpointer(config.checkpoint, config.experiment_name)

        # Create evaluator
        frame_pct: float | None = None
        if config.model.type == "picodeframe" and config.frame is not None:
            frame_pct = config.frame.max_frame_pct
        self.evaluator = Evaluator(
            self.encoder, self.decoder, self.device,
            decoder_size=self._decoder_size,
            frame_pct=frame_pct,
        )

        # Create discriminator if GAN training enabled
        self.discriminator: nn.Module | None = None
        self.d_optimizer: torch.optim.Optimizer | None = None

        if config.loss.gan_config.enabled:
            if config.loss.gan_config.discriminator_type == "patchgan":
                from picode.models.stegastamp.patchgan import PatchGANDiscriminator

                self.discriminator = PatchGANDiscriminator().to(self.device)
                self.d_optimizer = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=config.loss.gan_config.discriminator_lr,
                    betas=(0.5, 0.999),
                )
            else:
                from picode.models.stegastamp.discriminator import Discriminator

                self.discriminator = Discriminator().to(self.device)
                self.d_optimizer = torch.optim.RMSprop(
                    self.discriminator.parameters(),
                    lr=config.loss.gan_config.discriminator_lr,
                )

        # Training state
        self.global_step = 0
        self.best_metric = float("inf")
        self._phase2_triggered = False

        # EMA shadow weights for collapse recovery
        self._ema_encoder: dict[str, torch.Tensor] | None = None
        self._ema_decoder: dict[str, torch.Tensor] | None = None
        self._last_recovery_step: int = -1000  # Allow immediate first check

        # Optional LPIPS loss (lazy loaded)
        self._lpips_fn: nn.Module | None = None

        # Store train_image_size for later use
        self._train_image_size = train_image_size

        # Create border falloff mask (cached for efficiency)
        if config.loss.use_border_falloff:
            self._border_falloff_mask: Tensor | None = create_border_falloff_mask(
                height=train_image_size,
                width=train_image_size,
                falloff_speed=config.loss.border_falloff_speed,
                device=self.device,
            )
        else:
            self._border_falloff_mask = None

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
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Trainer:
        """Create trainer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            config_path: Optional YAML config to use instead of checkpoint's saved config.
                Useful when resuming with updated hyperparameters (e.g., new loss weights).
            overrides: Optional overrides to apply on top of config.

        Returns:
            Trainer with restored state.
        """
        data = torch.load(checkpoint_path, weights_only=False)

        # Use provided config or reconstruct from checkpoint
        if config_path:
            config = load_config(config_path, overrides)
        else:
            config = _dict_to_config(data["config"])

        # Create trainer
        trainer = cls(config)

        # Restore model states, filtering out shape-mismatched keys
        enc_filtered = _filter_compatible(data["encoder_state"], trainer.encoder)
        dec_filtered = _filter_compatible(data["decoder_state"], trainer.decoder)
        arch_changed = (
            len(enc_filtered) != len(data["encoder_state"])
            or len(dec_filtered) != len(data["decoder_state"])
        )
        trainer.encoder.load_state_dict(enc_filtered, strict=False)
        trainer.decoder.load_state_dict(dec_filtered, strict=False)

        # Skip optimizer state if architecture changed (Adam buffers have old shapes)
        if arch_changed:
            print("  Architecture changed — resetting optimizer state")
        else:
            trainer.optimizer.load_state_dict(data["optimizer_state"])

        # Restore training state
        trainer.global_step = data["step"]
        trainer.best_metric = data["best_metric"]

        # Restore scheduler state — skip when using a new config (fine-tuning).
        # CosineAnnealingLR.get_lr() computes relative updates from current optimizer LR,
        # so a corrupted LR (e.g. from collapse detector halvings) persists in the state.
        # A fresh schedule gives a warm restart with full LR.
        if config_path is not None:
            print("  New config provided — using fresh LR schedule (warm restart)")
            # Reset optimizer LRs to base values — load_state_dict restored corrupted LRs
            # but we want to keep Adam momentum buffers.
            if trainer.scheduler is not None:
                for i, group in enumerate(trainer.optimizer.param_groups):
                    group["lr"] = trainer.scheduler.base_lrs[i]
                print(f"  Reset optimizer LRs to {trainer.scheduler.base_lrs}")
        elif trainer.scheduler is not None and data.get("scheduler_state") is not None:
            trainer.scheduler.load_state_dict(data["scheduler_state"])

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

            if self.scheduler is not None:
                metrics["lr"] = self.scheduler.get_last_lr()[0]

            # EMA update and collapse detection
            ema_decay = self.config.training.ema_decay
            if ema_decay > 0:
                enc_state = self.encoder.state_dict()
                dec_state = self.decoder.state_dict()

                if self._ema_encoder is None:
                    # Initialize EMA from current weights
                    self._ema_encoder = {k: v.clone() for k, v in enc_state.items()}
                    self._ema_decoder = {k: v.clone() for k, v in dec_state.items()}
                else:
                    # Update EMA
                    for k in self._ema_encoder:
                        self._ema_encoder[k].mul_(ema_decay).add_(
                            enc_state[k], alpha=1 - ema_decay
                        )
                    assert self._ema_decoder is not None
                    for k in self._ema_decoder:
                        self._ema_decoder[k].mul_(ema_decay).add_(
                            dec_state[k], alpha=1 - ema_decay
                        )

                # Collapse detection (only after warmup and cooldown)
                threshold = self.config.training.collapse_threshold
                cooldown = self.config.training.collapse_recovery_cooldown
                # Don't detect collapse until model has had time to learn.
                # no_im_loss_steps marks when distortions begin — collapse is only
                # meaningful after the model has trained successfully for a while.
                min_collapse_step = max(
                    self.config.training.warmup_steps,
                    self.config.training.no_im_loss_steps,
                )
                warmup_done = self.global_step > min_collapse_step
                cooldown_done = (self.global_step - self._last_recovery_step) > cooldown
                prob_std = metrics.get("decoder_prob_std", 1.0)

                if warmup_done and cooldown_done and prob_std < threshold:
                    print(
                        f"\n*** COLLAPSE DETECTED at step {self.global_step} "
                        f"(prob_std={prob_std:.4f} < {threshold}) ***"
                    )
                    print("Restoring from EMA weights and halving LR...")

                    # Restore from EMA
                    self.encoder.load_state_dict(self._ema_encoder)
                    assert self._ema_decoder is not None
                    self.decoder.load_state_dict(self._ema_decoder)

                    # Halve LR
                    for group in self.optimizer.param_groups:
                        group["lr"] *= 0.5
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(f"New LR: {current_lr:.2e}")

                    self._last_recovery_step = self.global_step

                    # Re-initialize EMA from restored weights
                    enc_state = self.encoder.state_dict()
                    dec_state = self.decoder.state_dict()
                    self._ema_encoder = {k: v.clone() for k, v in enc_state.items()}
                    self._ema_decoder = {k: v.clone() for k, v in dec_state.items()}

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
                    scheduler=self.scheduler,
                    config=self.config,
                    metrics=metrics,
                )

            self.global_step += 1

            if self.scheduler is not None:
                self.scheduler.step()

        # Final checkpoint
        final_metrics = {"loss": 0.0}  # Placeholder
        self.checkpointer.save(
            step=self.global_step,
            encoder=self.encoder,
            decoder=self.decoder,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
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

        Dispatches to model-specific training step based on model type.

        Args:
            images: Batch of images (B, C, H, W) in [0, 1].

        Returns:
            Dict of metrics for this step.
        """
        if self.config.model.type == "picodeframe":
            return self._train_step_picodeframe(images)
        elif self.config.model.type == "picodetier":
            return self._train_step_picodetier(images)
        return self._train_step_default(images)

    def _train_step_default(self, images: Tensor) -> dict[str, float]:
        """Execute a single training step for StegaStamp/PicodeLite.

        Implements the StegaStamp warp-encode-unwarp flow:
        1. Generate random perspective transform matrices
        2. Warp input image to canonical encoding space
        3. Encode in warped space
        4. Unwarp residual back to original space
        5. Apply border handling
        6. Apply remaining distortions
        7. Decode

        Args:
            images: Batch of images (B, C, H, W) in [0, 1].

        Returns:
            Dict of metrics for this step.
        """
        batch_size = images.shape[0]
        num_bits = self.config.training.num_bits
        # Get image size from actual input (works for both model types)
        image_size = images.shape[-1]  # H dimension

        # Generate random messages
        messages = torch.randint(0, 2, (batch_size, num_bits), device=self.device).float()

        # Compute ramped perspective strength
        rnd_trans = self.config.training.rnd_trans
        rnd_trans_ramp = self.config.training.rnd_trans_ramp
        if rnd_trans_ramp > 0:
            perspective_strength = min(
                rnd_trans * self.global_step / rnd_trans_ramp, rnd_trans
            )
        else:
            perspective_strength = rnd_trans

        # Generate perspective transform matrices (or identity if strength is 0)
        if perspective_strength > 0:
            M_forward, M_inverse = get_rand_transform_matrix(
                batch_size, image_size, perspective_strength, self.device
            )
        else:
            M_forward = get_identity_transform(batch_size, self.device)
            M_inverse = get_identity_transform(batch_size, self.device)

        # 1. Warp input image to canonical space
        # M_forward maps src->dst, so to warp we need to sample from src given dst
        # This means we apply M_inverse to get source coords for each dest coord
        images_warped = perspective_transform(images, M_inverse, padding_mode="border")

        # Update encoder strength if annealing is active
        if (
            hasattr(self.encoder, 'strength')
            and self.encoder.strength is not None
            and self.config.training.residual_strength > 0
        ):
            self.encoder.strength = self._compute_strength(
                initial=self.config.training.residual_strength,
                target=self.config.training.residual_strength_anneal_target,
                start_step=self.config.training.residual_strength_anneal_start,
                anneal_steps=self.config.training.residual_strength_anneal_steps,
                current_step=self.global_step,
                schedule=self.config.training.anneal_schedule,
            )

        # Update blur sigma if ramping is active
        if (
            hasattr(self.encoder, 'residual_blur_sigma')
            and self.config.training.residual_blur_sigma > 0
        ):
            ramp_start = self.config.training.residual_blur_sigma_ramp_start
            ramp_steps = self.config.training.residual_blur_sigma_ramp_steps
            start_val = self.config.training.residual_blur_sigma_start
            target_val = self.config.training.residual_blur_sigma
            if ramp_steps > 0 and self.global_step >= ramp_start:
                progress = min((self.global_step - ramp_start) / ramp_steps, 1.0)
                self.encoder.residual_blur_sigma = start_val + progress * (target_val - start_val)
            elif ramp_steps > 0:
                self.encoder.residual_blur_sigma = start_val
            # else: no ramp, use fixed value from init

        # 2. Encode in warped space
        encoder_output = self.encoder(images_warped, messages)

        # Handle v2 dict return or v1 tensor return
        if isinstance(encoder_output, dict):
            encoded_warped = encoder_output["encoded"]
            encoder_mask = encoder_output.get("mask")  # (B, 1, H, W) or None
        else:
            encoded_warped = encoder_output
            encoder_mask = None

        # 3. Compute residual in warped space
        residual_warped = encoded_warped - images_warped

        # 4. Unwarp residual back to original space
        # To unwarp, we sample from warped coords given original coords
        # This means we apply M_forward to get warped coords for each original coord
        residual_unwarped = perspective_transform(
            residual_warped, M_forward, padding_mode="zeros"
        )

        # 5. Apply border handling and combine with original image
        encoded = self._apply_border_mode(
            images, residual_unwarped, M_forward, self.config.training.borders
        )

        # 6. Apply distortions to encoded image
        distorted = self.distortion(encoded, self.global_step)

        # 7. Resize to decoder size if different from encoder size (e.g., PicodeLite 800->320)
        if distorted.shape[-1] != self._decoder_size:
            decoder_input = torch.nn.functional.interpolate(
                distorted, size=(self._decoder_size, self._decoder_size),
                mode="bilinear", align_corners=False
            )
        else:
            decoder_input = distorted

        # 7b. Apply decoder-input blur (training-time smoothing)
        if self._decoder_blur_kernel is not None:
            decoder_input = F.conv2d(
                decoder_input, self._decoder_blur_kernel,
                padding=self._decoder_blur_pad, groups=3,
            )

        # 8. Decode
        decoded_logits = self.decoder(decoder_input)

        # Compute residual for diagnostics (encoded - original)
        residual = encoded - images

        metrics: dict[str, float] = {}

        # Compute losses (decoder outputs logits, loss uses BCE with logits)
        losses = self._compute_ramped_losses(images, encoded, messages, decoded_logits)

        # Mask regularization (PicoTrust v2)
        if encoder_mask is not None and self.config.loss.mask_reg is not None:
            mask_reg_scale = self._ramp(
                self.config.loss.mask_reg.scale,
                self.config.loss.mask_reg.ramp_steps,
                self.global_step,
            )
            if mask_reg_scale > 0:
                loss_mask_reg = self._compute_mask_reg_loss(encoder_mask, images_warped)
                losses["loss_mask_reg"] = loss_mask_reg
                losses["loss"] = losses["loss"] + mask_reg_scale * loss_mask_reg

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

        # Clip gradients
        if self.config.training.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=self.config.training.grad_clip_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), max_norm=self.config.training.grad_clip_norm
            )
        elif self.config.training.generator_grad_clip > 0:
            clip_val = self.config.training.generator_grad_clip
            for p in self.encoder.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)
            for p in self.decoder.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)

        self.optimizer.step()

        # Phase 2: reduce decoder LR (one-time trigger)
        if (
            not self._phase2_triggered
            and self.config.training.phase2_step > 0
            and self.global_step >= self.config.training.phase2_step
        ):
            self._phase2_triggered = True
            scale = self.config.training.phase2_decoder_lr_scale
            # Decoder param group is index 1
            self.optimizer.param_groups[1]["lr"] *= scale
            # STN param group is index 2 (if exists)
            if len(self.optimizer.param_groups) > 2:
                self.optimizer.param_groups[2]["lr"] *= scale

        # GAN training step (if enabled)
        loss_D = torch.tensor(0.0, device=self.device)

        if (
            self.discriminator is not None
            and self.d_optimizer is not None
            and self.config.loss.gan_config.enabled
        ):
            self.d_optimizer.zero_grad()
            d_real = self.discriminator(images)
            d_fake = self.discriminator(encoded.detach())

            if self.config.loss.gan_config.discriminator_type == "patchgan":
                # LSGAN loss: D(real) -> 1, D(fake) -> 0
                loss_D = 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()
                loss_D.backward()
                self.d_optimizer.step()
            else:
                # WGAN loss: minimize D(fake) - D(real)
                loss_D = d_fake.mean() - d_real.mean()
                loss_D.backward()
                # Clip discriminator gradients by value
                if self.config.loss.gan_config.gradient_clip > 0:
                    clip_val = self.config.loss.gan_config.gradient_clip
                    for p in self.discriminator.parameters():
                        if p.grad is not None:
                            p.grad.data.clamp_(-clip_val, clip_val)
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

            # Per-image accuracy: std and min across batch
            per_image_acc = (predicted_bits == messages).float().mean(dim=1)  # (B,)
            metrics["bit_acc_std"] = per_image_acc.std().item()
            metrics["bit_acc_min"] = per_image_acc.min().item()

        # v2 diagnostic metrics
        if encoder_mask is not None:
            metrics["mask_mean"] = encoder_mask.mean().item()
            metrics["mask_std"] = encoder_mask.std().item()
        if hasattr(self.encoder, 'strength') and self.encoder.strength is not None:
            metrics["strength"] = self.encoder.strength

        return metrics

    def _train_step_picodetier(self, images: Tensor) -> dict[str, float]:
        """Execute a single training step for PicodeTier.

        Similar to _train_step_default but with:
        - Per-tier message generation with variable bit counts and masks
        - Tier-aware strength annealing (each tier anneals to its own target)
        - Masked message loss (only active bits contribute)
        - Tier classifier cross-entropy loss

        Args:
            images: Batch of images (B, C, H, W) in [0, 1].

        Returns:
            Dict of metrics for this step.
        """
        from picode.models.picodetier.tiers import MAX_BITS, NUM_TIERS, TIERS

        batch_size = images.shape[0]
        image_size = images.shape[-1]

        # --- Per-tier message generation ---
        tiers = torch.randint(0, NUM_TIERS, (batch_size,), device=self.device)

        messages = torch.zeros(batch_size, MAX_BITS, device=self.device)
        masks = torch.zeros(batch_size, MAX_BITS, device=self.device)
        for i in range(batch_size):
            t_idx = int(tiers[i].item())
            n_bits = int(TIERS[t_idx]["bits"])
            messages[i, :n_bits] = torch.randint(
                0, 2, (n_bits,), device=self.device,
            ).float()
            masks[i, :n_bits] = 1.0

        # --- Perspective warp (same as _train_step_default) ---
        rnd_trans = self.config.training.rnd_trans
        rnd_trans_ramp = self.config.training.rnd_trans_ramp
        if rnd_trans_ramp > 0:
            perspective_strength = min(
                rnd_trans * self.global_step / rnd_trans_ramp, rnd_trans
            )
        else:
            perspective_strength = rnd_trans

        if perspective_strength > 0:
            M_forward, M_inverse = get_rand_transform_matrix(
                batch_size, image_size, perspective_strength, self.device
            )
        else:
            M_forward = get_identity_transform(batch_size, self.device)
            M_inverse = get_identity_transform(batch_size, self.device)

        images_warped = perspective_transform(images, M_inverse, padding_mode="border")

        # --- Tier-aware strength annealing ---
        if self.config.training.residual_strength > 0:
            initial = self.config.training.residual_strength
            start = self.config.training.residual_strength_anneal_start
            steps = self.config.training.residual_strength_anneal_steps
            progress = 0.0
            if self.global_step >= start and steps > 0:
                progress = min((self.global_step - start) / steps, 1.0)
            for t_idx in range(NUM_TIERS):
                target = float(TIERS[t_idx]["strength"])
                self.encoder.tier_strengths[t_idx] = initial + progress * (target - initial)

        # --- Encode ---
        encoder_output = self.encoder(images_warped, messages, tiers)
        encoded_warped = encoder_output["encoded"]

        # --- Unwarp and border handling ---
        residual_warped = encoded_warped - images_warped
        residual_unwarped = perspective_transform(
            residual_warped, M_forward, padding_mode="zeros"
        )
        encoded = self._apply_border_mode(
            images, residual_unwarped, M_forward, self.config.training.borders
        )

        # --- Distortions ---
        distorted = self.distortion(encoded, self.global_step)

        # --- Resize to decoder size if needed ---
        if distorted.shape[-1] != self._decoder_size:
            decoder_input = F.interpolate(
                distorted, size=(self._decoder_size, self._decoder_size),
                mode="bilinear", align_corners=False,
            )
        else:
            decoder_input = distorted

        # --- Decode ---
        decoded_logits, tier_logits = self.decoder(decoder_input, tier=tiers)

        # --- Compute losses ---
        residual = encoded - images
        step = self.global_step
        loss_cfg = self.config.loss
        no_im = self.config.training.no_im_loss_steps
        skip_im = step < no_im
        eff_step = max(0, step - no_im)

        # 1. Masked message loss
        msg_scale = self._ramp(loss_cfg.message.scale, loss_cfg.message.ramp_steps, step)
        if loss_cfg.message_loss_type == "mse":
            decoded_probs = torch.sigmoid(decoded_logits)
            loss_per_bit = (decoded_probs - messages) ** 2
        else:
            loss_per_bit = F.binary_cross_entropy_with_logits(
                decoded_logits, messages, reduction="none",
            )
        loss_msg = (loss_per_bit * masks).sum() / masks.sum()

        # 2. Tier classifier loss
        tier_scale = self._ramp(
            loss_cfg.tier_classifier.scale, loss_cfg.tier_classifier.ramp_steps, step,
        )
        loss_tier = F.cross_entropy(tier_logits, tiers)

        # 3. STN regularization
        if hasattr(self.decoder, "stn_scale_reg"):
            loss_stn = self.decoder.stn_scale_reg()
            weighted_stn = 0.1 * loss_stn
        else:
            loss_stn = torch.tensor(0.0, device=self.device)
            weighted_stn = 0.0

        # 4. Image losses (skip during no_im_loss phase)
        loss_l2 = F.mse_loss(encoded, images)
        if skip_im:
            weighted_l2 = torch.tensor(0.0, device=self.device)
        else:
            l2_scale = self._ramp(loss_cfg.l2.scale, loss_cfg.l2.ramp_steps, eff_step)
            weighted_l2 = l2_scale * loss_l2

        total = msg_scale * loss_msg + tier_scale * loss_tier + weighted_l2 + weighted_stn

        losses: dict[str, Tensor] = {
            "loss_msg": loss_msg,
            "loss_tier": loss_tier,
            "loss_l2": loss_l2,
            "loss_stn_reg": loss_stn,
        }

        # LPIPS
        if not skip_im:
            lpips_scale = self._ramp(
                loss_cfg.lpips.scale, loss_cfg.lpips.ramp_steps, eff_step,
            )
        else:
            lpips_scale = 0.0
        if lpips_scale > 0 and self._get_lpips_fn() is not None:
            loss_lpips = self._compute_lpips(images, encoded)
            total = total + lpips_scale * loss_lpips
            losses["loss_lpips"] = loss_lpips

        # FFL
        if not skip_im and loss_cfg.ffl is not None:
            ffl_scale = self._delayed_ramp(loss_cfg.ffl, eff_step)
            if ffl_scale > 0:
                loss_ffl = self._compute_ffl(images, encoded)
                total = total + ffl_scale * loss_ffl
                losses["loss_ffl"] = loss_ffl

        # GAN generator loss
        if (
            not skip_im
            and hasattr(self, "discriminator")
            and self.discriminator is not None
        ):
            g_scale = self._ramp(
                loss_cfg.gan_config.g_loss_scale,
                loss_cfg.gan_config.g_loss_ramp_steps,
                eff_step,
            )
            if g_scale > 0:
                d_fake = self.discriminator(encoded)
                if loss_cfg.gan_config.discriminator_type == "patchgan":
                    loss_G = 0.5 * ((d_fake - 1) ** 2).mean()
                else:
                    loss_G = -d_fake.mean()
                total = total + g_scale * loss_G
                losses["loss_G"] = loss_G

        losses["loss"] = total

        # --- Warmup: message + tier only ---
        if step < self.config.training.warmup_steps:
            total_loss = msg_scale * loss_msg + tier_scale * loss_tier
        else:
            total_loss = total

        # --- Backward and optimize ---
        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore[no-untyped-call]

        if self.config.training.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=self.config.training.grad_clip_norm,
            )
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), max_norm=self.config.training.grad_clip_norm,
            )
        elif self.config.training.generator_grad_clip > 0:
            clip_val = self.config.training.generator_grad_clip
            for p in self.encoder.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)
            for p in self.decoder.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)

        self.optimizer.step()

        # Phase 2: reduce decoder LR
        if (
            not self._phase2_triggered
            and self.config.training.phase2_step > 0
            and self.global_step >= self.config.training.phase2_step
        ):
            self._phase2_triggered = True
            scale = self.config.training.phase2_decoder_lr_scale
            self.optimizer.param_groups[1]["lr"] *= scale
            if len(self.optimizer.param_groups) > 2:
                self.optimizer.param_groups[2]["lr"] *= scale

        # GAN discriminator step
        loss_D = torch.tensor(0.0, device=self.device)
        if (
            self.discriminator is not None
            and self.d_optimizer is not None
            and self.config.loss.gan_config.enabled
        ):
            self.d_optimizer.zero_grad()
            d_real = self.discriminator(images)
            d_fake = self.discriminator(encoded.detach())

            if self.config.loss.gan_config.discriminator_type == "patchgan":
                loss_D = 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()
                loss_D.backward()
                self.d_optimizer.step()
            else:
                loss_D = d_fake.mean() - d_real.mean()
                loss_D.backward()
                if self.config.loss.gan_config.gradient_clip > 0:
                    clip_val = self.config.loss.gan_config.gradient_clip
                    for p in self.discriminator.parameters():
                        if p.grad is not None:
                            p.grad.data.clamp_(-clip_val, clip_val)
                self.d_optimizer.step()
                clip_val = self.config.loss.gan_config.clip_weights
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_val, clip_val)

        # --- Metrics ---
        metrics: dict[str, float] = {}

        if self.discriminator is not None:
            metrics["loss_D"] = loss_D.item() if isinstance(loss_D, Tensor) else loss_D

        metrics.update({k: v.item() for k, v in losses.items()})

        # Residual statistics
        metrics["residual_mean"] = residual.mean().item()
        metrics["residual_std"] = residual.std().item()
        metrics["residual_abs_max"] = residual.abs().max().item()

        # Decoder output distribution
        with torch.no_grad():
            decoded_probs = torch.sigmoid(decoded_logits)
            metrics["decoder_prob_mean"] = decoded_probs.mean().item()
            metrics["decoder_prob_std"] = decoded_probs.std().item()

            # Masked bit accuracy (only active bits)
            predicted_bits = (decoded_probs > 0.5).float()
            correct_masked = ((predicted_bits == messages).float() * masks).sum()
            bit_accuracy = (correct_masked / masks.sum()).item()
            metrics["bit_accuracy"] = bit_accuracy

            # Tier classification accuracy
            tier_preds = tier_logits.argmax(dim=1)
            tier_accuracy = (tier_preds == tiers).float().mean().item()
            metrics["tier_accuracy"] = tier_accuracy

        # Per-tier strength values
        if hasattr(self.encoder, "tier_strengths"):
            for t_idx in range(NUM_TIERS):
                metrics[f"strength_tier{t_idx}"] = self.encoder.tier_strengths[t_idx].item()

        return metrics

    def _train_step_picodeframe(self, images: Tensor) -> dict[str, float]:
        """Execute a single training step for PicodeFrame.

        PicodeFrame encoding flow:
        1. Load 400x400 images (ground truth including borders)
        2. Sample random frame_width
        3. Extract inner image, reflection-pad back to 400x400
        4. Encode with frame_width parameter
        5. Apply distortions
        6. Decode
        7. Compute frame-specific losses

        Args:
            images: Batch of images (B, C, H, W) in [0, 1].

        Returns:
            Dict of metrics for this step.
        """
        from picode.models.picodeframe import loss as frame_loss
        from picode.models.picodeframe.decoder import Decoder as FrameDecoder

        batch_size = images.shape[0]
        num_bits = self.config.training.num_bits
        image_size = images.shape[-1]  # 400

        # Generate random messages
        messages = torch.randint(0, 2, (batch_size, num_bits), device=self.device).float()

        # Sample frame width. During fixed_frame_steps, use max width so the
        # encoder/decoder learn on the easiest (widest) frames first.
        frame_cfg = self.config.frame
        if frame_cfg is not None:
            min_fw = int(frame_cfg.min_frame_pct * image_size)
            max_fw = int(frame_cfg.max_frame_pct * image_size)
            fixed_steps = frame_cfg.fixed_frame_steps
        else:
            min_fw = int(0.02 * image_size)
            max_fw = int(0.05 * image_size)
            fixed_steps = 0

        if fixed_steps > 0 and self.global_step < fixed_steps:
            fw = max_fw
        else:
            fw = torch.randint(min_fw, max_fw + 1, (1,)).item()
        fw = int(fw)

        # Extract inner image and reflection-pad back to full size
        inner = images[:, :, fw:image_size - fw, fw:image_size - fw]
        padded_inner = F.pad(inner, (fw, fw, fw, fw), mode="reflect")

        # Create mask (1 in center, 0 in border)
        mask = torch.zeros(batch_size, 1, image_size, image_size, device=self.device)
        mask[:, :, fw:image_size - fw, fw:image_size - fw] = 1.0

        # Encode
        encoded = self.encoder(padded_inner, messages, frame_width=fw)

        # During warmup, skip distortions and perspective warp entirely.
        # The message is hidden in a thin border (~4% of pixels) — the decoder
        # needs clean signal to first learn the message channel before we add noise.
        in_warmup = self.global_step < self.config.training.warmup_steps

        if in_warmup:
            decoder_input = encoded
        else:
            # Apply perspective warp (ramped)
            rnd_trans = self.config.training.rnd_trans
            rnd_trans_ramp = self.config.training.rnd_trans_ramp
            # Ramp from warmup end, not from step 0
            warp_step = self.global_step - self.config.training.warmup_steps
            if rnd_trans_ramp > 0:
                perspective_strength = min(
                    rnd_trans * warp_step / rnd_trans_ramp, rnd_trans
                )
            else:
                perspective_strength = rnd_trans

            if perspective_strength > 0:
                M_forward, M_inverse = get_rand_transform_matrix(
                    batch_size, image_size, perspective_strength, self.device
                )
                encoded_warped = perspective_transform(
                    encoded, M_inverse, padding_mode="border"
                )
            else:
                encoded_warped = encoded

            # Apply distortions (curriculum also ramps from 0)
            decoder_input = self.distortion(encoded_warped, warp_step)

        # Decode — pass mask as border indicator channel for the CNN,
        # and frame_width for the border-pooling branch.
        decoded_logits = self.decoder(decoder_input, mask=mask, frame_width=fw)

        # Compute losses
        step = self.global_step
        no_im_loss_steps = self.config.training.no_im_loss_steps
        skip_image_loss = step < no_im_loss_steps
        effective_step = max(0, step - no_im_loss_steps)

        # Message loss
        msg_scale = self._ramp(
            self.config.loss.message.scale,
            self.config.loss.message.ramp_steps,
            step,
        )
        if self.config.loss.message_loss_type == "mse":
            loss_msg = frame_loss.message_loss_mse(decoded_logits, messages)
        else:
            loss_msg = frame_loss.message_loss(decoded_logits, messages)

        losses: dict[str, Tensor] = {"loss_msg": loss_msg}

        # Frame L2 loss
        if frame_cfg is not None:
            fl2_scale_cfg = frame_cfg.frame_l2_scale
            fl2_ramp = frame_cfg.frame_l2_ramp_steps
            flpips_scale_cfg = frame_cfg.frame_lpips_scale
            flpips_ramp = frame_cfg.frame_lpips_ramp_steps
            fcolor_scale_cfg = frame_cfg.frame_color_scale
            fcolor_ramp = frame_cfg.frame_color_ramp_steps
            stn_reg_scale = frame_cfg.stn_reg_scale
        else:
            fl2_scale_cfg = 2.0
            fl2_ramp = 1
            flpips_scale_cfg = 1.5
            flpips_ramp = 10000
            fcolor_scale_cfg = 2.0
            fcolor_ramp = 1
            stn_reg_scale = 0.1

        loss_fl2 = frame_loss.frame_l2_loss(encoded, images, mask)
        losses["loss_frame_l2"] = loss_fl2

        # Frame color loss (penalize color shifts in residual)
        loss_fcolor = frame_loss.frame_color_loss(encoded, images, mask)
        losses["loss_frame_color"] = loss_fcolor

        # STN regularization
        assert isinstance(self.decoder, FrameDecoder)
        loss_stn = frame_loss.stn_scale_loss(self.decoder)
        losses["loss_stn_reg"] = loss_stn

        # Compute total loss.
        # Frame L2 is always included (even during warmup) to prevent the
        # encoder's residual from growing unbounded when no image loss is active.
        # Color loss is delayed until after warmup so the message path establishes
        # first — its direct gradient to the encoder can overwhelm the attenuated
        # message gradient during early training.
        decoder_warmup = self.config.training.decoder_warmup_steps
        if step < decoder_warmup:
            # Decoder bootstrap: message-only, no L2.
            # Lets encoder produce strong border modifications so decoder
            # has clear signal to learn from.
            total_loss = msg_scale * loss_msg
        elif step < self.config.training.warmup_steps:
            total_loss = msg_scale * loss_msg + fl2_scale_cfg * loss_fl2
        else:
            fcolor_scale = self._ramp(fcolor_scale_cfg, fcolor_ramp, effective_step)
            total_loss = (
                msg_scale * loss_msg + stn_reg_scale * loss_stn
                + fcolor_scale * loss_fcolor
            )

            if not skip_image_loss:
                fl2_scale = self._ramp(fl2_scale_cfg, fl2_ramp, effective_step)
                total_loss = total_loss + fl2_scale * loss_fl2

                # Frame LPIPS loss
                flpips_scale = self._ramp(flpips_scale_cfg, flpips_ramp, effective_step)
                if flpips_scale > 0 and self._get_lpips_fn() is not None:
                    loss_flpips = frame_loss.frame_lpips_loss(
                        encoded, images, mask, self._get_lpips_fn()  # type: ignore[arg-type]
                    )
                    total_loss = total_loss + flpips_scale * loss_flpips
                    losses["loss_frame_lpips"] = loss_flpips

        losses["loss"] = total_loss

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore[no-untyped-call]

        # Clip gradients
        if self.config.training.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=self.config.training.grad_clip_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), max_norm=self.config.training.grad_clip_norm
            )
        elif self.config.training.generator_grad_clip > 0:
            clip_val = self.config.training.generator_grad_clip
            for p in self.encoder.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)
            for p in self.decoder.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)

        self.optimizer.step()

        # Metrics
        metrics: dict[str, float] = {}
        metrics.update({k: v.item() for k, v in losses.items()})

        # Residual statistics (only in frame region)
        # Use padded_inner (encoder input), not images (original crop) — they differ
        # in the border region because of reflection padding.
        residual = encoded - padded_inner
        frame_residual = residual * (1 - mask)
        metrics["residual_mean"] = frame_residual.mean().item()
        metrics["residual_std"] = frame_residual.std().item()
        metrics["residual_abs_max"] = frame_residual.abs().max().item()
        metrics["frame_width"] = float(fw)

        with torch.no_grad():
            decoded_probs = torch.sigmoid(decoded_logits)
            metrics["decoder_prob_mean"] = decoded_probs.mean().item()
            metrics["decoder_prob_std"] = decoded_probs.std().item()

            predicted_bits = (decoded_probs > 0.5).float()
            bit_accuracy = (predicted_bits == messages).float().mean().item()
            metrics["bit_accuracy"] = bit_accuracy

            # Per-image accuracy: std and min across batch
            per_image_acc = (predicted_bits == messages).float().mean(dim=1)  # (B,)
            metrics["bit_acc_std"] = per_image_acc.std().item()
            metrics["bit_acc_min"] = per_image_acc.min().item()

        return metrics

    def _apply_border_mode(
        self,
        original: Tensor,
        residual: Tensor,
        M_forward: Tensor,
        border_mode: str,
    ) -> Tensor:
        """Apply border handling after unwarping residual.

        The border mode determines how to handle pixels outside the valid
        transformed region (where the residual is zero due to padding).

        Modes:
        - no_edge: Simply add residual to original (may have visible edge artifacts)
        - black: Black border outside valid region
        - white: White border outside valid region
        - random: Random gray value border (same value for entire border)
        - randomrgb: Random RGB color border (same color for entire border)
        - image: Use original image as background (default, recommended)

        Args:
            original: Original images (B, C, H, W) in [0, 1].
            residual: Unwarped residual (B, C, H, W).
            M_forward: Forward homography matrices (B, 3, 3) - used to compute valid mask.
            border_mode: Border handling mode.

        Returns:
            Final encoded image with border handling applied.
        """
        B, C, H, W = original.shape

        if border_mode == "no_edge":
            # Simply add residual - visible edge artifacts possible
            return (original + residual).clamp(0.0, 1.0)

        # Create a mask of valid transformed region
        # Transform a white image and see where it maps to
        ones = torch.ones(B, 1, H, W, device=original.device, dtype=original.dtype)
        valid_mask = perspective_transform(ones, M_forward, padding_mode="zeros")
        # Threshold to binary mask (pixels > 0.5 are valid)
        valid_mask = (valid_mask > 0.5).float()

        # Apply residual only in valid region
        encoded_valid = original + residual * valid_mask

        if border_mode == "image":
            # Use original image as background (residual is 0 outside valid region)
            # This is effectively what we already have
            return encoded_valid.clamp(0.0, 1.0)

        elif border_mode == "black":
            # Black border outside valid region
            background = torch.zeros_like(original)

        elif border_mode == "white":
            # White border outside valid region
            background = torch.ones_like(original)

        elif border_mode == "random":
            # Random gray value (same for entire batch)
            gray_value = torch.rand(1, device=original.device, dtype=original.dtype)
            background = torch.full_like(original, gray_value.item())

        elif border_mode == "randomrgb":
            # Random RGB color (same for entire batch)
            rgb_values = torch.rand(1, 3, 1, 1, device=original.device, dtype=original.dtype)
            background = rgb_values.expand(B, C, H, W)

        else:
            # Default to image mode
            background = original

        # Composite: valid region from encoded, background elsewhere
        # Use valid_mask to blend (expand to match channels)
        valid_mask_3ch = valid_mask.expand(-1, C, -1, -1)
        result = encoded_valid * valid_mask_3ch + background * (1 - valid_mask_3ch)

        return result.clamp(0.0, 1.0)

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

        # STN regularization (for models with STN: stegastamp, picodeframe, picotrust)
        if hasattr(self.decoder, "stn_scale_reg"):
            loss_stn = self.decoder.stn_scale_reg()
            weighted_stn = 0.1 * loss_stn
        else:
            loss_stn = torch.tensor(0.0, device=encoded.device)
            weighted_stn = 0.0

        # Start with message, l2, and STN reg loss
        total = weighted_msg + weighted_l2 + weighted_stn

        losses: dict[str, Tensor] = {
            "loss_msg": loss_msg,
            "loss_l2": loss_l2,
            "loss_stn_reg": loss_stn,
        }

        # Chrominance loss — penalise cross-channel variance of the residual.
        # Zero when R=G=B delta (pure luminance); positive on colour shifts.
        # Delayed to avoid interfering with encoder-decoder bootstrap.
        if not skip_image_loss and loss_cfg.chroma is not None:
            chroma_scale = self._delayed_ramp(loss_cfg.chroma, effective_step)
            if chroma_scale > 0:
                residual = encoded - original                      # (B, 3, H, W)
                loss_chroma = residual.var(dim=1).mean()           # scalar
                total = total + chroma_scale * loss_chroma
                losses["loss_chroma"] = loss_chroma

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

        # Focal Frequency Loss (FFL) — penalizes per-frequency reconstruction error.
        # Targets periodic wave artifacts by weighting hard-to-reconstruct frequencies.
        if not skip_image_loss and loss_cfg.ffl is not None:
            ffl_scale = self._delayed_ramp(loss_cfg.ffl, effective_step)
            if ffl_scale > 0:
                loss_ffl = self._compute_ffl(original, encoded)
                total = total + ffl_scale * loss_ffl
                losses["loss_ffl"] = loss_ffl

        # Laplacian loss — penalizes high-frequency content in the residual.
        if not skip_image_loss and loss_cfg.laplacian is not None:
            lap_scale = self._delayed_ramp(loss_cfg.laplacian, effective_step)
            if lap_scale > 0:
                residual = encoded - original
                loss_lap = self._compute_laplacian_loss(residual)
                total = total + lap_scale * loss_lap
                losses["loss_laplacian"] = loss_lap

        # SSIM loss -- structural similarity
        if not skip_image_loss and loss_cfg.ssim is not None:
            ssim_scale = self._delayed_ramp(loss_cfg.ssim, effective_step)
            if ssim_scale > 0:
                loss_ssim = self._compute_ssim_loss(original, encoded)
                total = total + ssim_scale * loss_ssim
                losses["loss_ssim"] = loss_ssim

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
                if self.config.loss.gan_config.discriminator_type == "patchgan":
                    # LSGAN generator loss: D(fake) -> 1
                    loss_G = 0.5 * ((d_fake - 1) ** 2).mean()
                else:
                    loss_G = -d_fake.mean()  # WGAN: maximize D(fake)
                weighted_G = g_scale * loss_G
                total = total + weighted_G
                losses["loss_G"] = loss_G

        # Border falloff loss (after delay) - skip during no_im_loss_steps
        # This matches original StegaStamp: amplifies L2 loss at image borders
        if not skip_image_loss and effective_step >= loss_cfg.l2_edge_delay_steps:
            edge_step = effective_step - loss_cfg.l2_edge_delay_steps
            edge_scale = self._ramp(
                loss_cfg.l2_edge_gain, loss_cfg.l2_edge_ramp_steps, edge_step
            )
            if edge_scale > 0:
                if loss_cfg.use_border_falloff and self._border_falloff_mask is not None:
                    # Border falloff: amplify loss at borders using cosine mask
                    loss_edge = compute_yuv_l2_loss_with_falloff(
                        original, encoded, yuv_weights, self._border_falloff_mask, edge_scale
                    )
                else:
                    # Fallback to Sobel edge loss if border falloff disabled
                    loss_edge = self._compute_edge_loss(original, encoded)
                    loss_edge = edge_scale * loss_edge
                total = total + loss_edge
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

    @staticmethod
    def _compute_ssim_loss(
        original: Tensor, encoded: Tensor,
        window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2,
    ) -> Tensor:
        """Compute 1 - SSIM as a loss (0 = identical).

        Args:
            original: Original images (B, C, H, W).
            encoded: Encoded images (B, C, H, W).
            window_size: Size of the averaging window.
            C1: Stabilization constant for luminance.
            C2: Stabilization constant for contrast.

        Returns:
            Scalar loss tensor (1 - mean SSIM).
        """
        channels = original.shape[1]
        kernel = torch.ones(channels, 1, window_size, window_size,
                            device=original.device) / (window_size ** 2)
        pad = window_size // 2

        mu_x = F.conv2d(original, kernel, groups=channels, padding=pad)
        mu_y = F.conv2d(encoded, kernel, groups=channels, padding=pad)

        sigma_x_sq = F.conv2d(original ** 2, kernel, groups=channels, padding=pad) - mu_x ** 2
        sigma_y_sq = F.conv2d(encoded ** 2, kernel, groups=channels, padding=pad) - mu_y ** 2
        sigma_xy = (
            F.conv2d(original * encoded, kernel, groups=channels, padding=pad) - mu_x * mu_y
        )

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return 1.0 - ssim_map.mean()

    @staticmethod
    def _compute_mask_reg_loss(mask: Tensor, image: Tensor) -> Tensor:
        """Encourage mask to correlate with local image texture.

        Higher-variance regions (edges, textures) should receive stronger encoding.

        Args:
            mask: Learned spatial mask (B, 1, H, W).
            image: Input image (B, C, H, W).

        Returns:
            Scalar MSE loss between mask and normalized local variance.
        """
        gray = image.mean(dim=1, keepdim=True)
        kernel = torch.ones(1, 1, 7, 7, device=image.device) / 49.0
        local_mean = F.conv2d(gray, kernel, padding=3)
        local_var = F.conv2d(gray ** 2, kernel, padding=3) - local_mean ** 2
        local_var = local_var.clamp(min=0)
        lv_max = local_var.amax(dim=(-2, -1), keepdim=True) + 1e-8
        local_var_norm = local_var / lv_max
        return F.mse_loss(mask, local_var_norm.detach())

    @staticmethod
    def _compute_strength(
        initial: float, target: float,
        start_step: int, anneal_steps: int, current_step: int,
        schedule: str = "linear",
    ) -> float:
        """Compute annealed residual strength.

        Args:
            initial: Starting strength value.
            target: Target strength value after annealing.
            start_step: Step at which annealing begins.
            anneal_steps: Number of steps to anneal over.
            current_step: Current training step.
            schedule: "linear" or "exponential".

        Returns:
            Interpolated strength value.
        """
        if current_step < start_step:
            return initial
        progress = min((current_step - start_step) / max(anneal_steps, 1), 1.0)
        if schedule == "exponential" and initial > 0 and target > 0:
            return initial * (target / initial) ** progress
        return initial + progress * (target - initial)

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

    @staticmethod
    def _compute_ffl(original: Tensor, encoded: Tensor, alpha: float = 1.0) -> Tensor:
        """Compute Focal Frequency Loss (Jiang et al., 2021).

        Penalizes per-frequency reconstruction error in the 2D FFT domain,
        with adaptive weighting that focuses on hard-to-reconstruct frequencies.
        This directly targets periodic wave artifacts.

        Args:
            original: Original images (B, C, H, W) in [0, 1].
            encoded: Encoded images (B, C, H, W) in [0, 1].
            alpha: Focal exponent — higher values focus more on hard frequencies.

        Returns:
            Scalar FFL loss.
        """
        # 2D FFT of both images
        freq_orig = torch.fft.rfft2(original, norm="ortho")
        freq_enc = torch.fft.rfft2(encoded, norm="ortho")

        # Per-frequency L2 distance (on complex magnitudes)
        diff = torch.abs(freq_orig - freq_enc)  # (B, C, H, W//2+1)

        # Focal weight: frequencies with larger error get higher weight
        # Detach so weights don't contribute gradients (like focal loss)
        weight = diff.detach() ** alpha
        # Normalize weights to [0, 1] per sample
        weight = weight / (weight.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        return (weight * diff).mean()

    @staticmethod
    def _compute_laplacian_loss(residual: Tensor) -> Tensor:
        """Penalize high-frequency content in the residual via Laplacian magnitude.

        Computes the discrete Laplacian (2nd spatial derivative) of the residual
        and returns the mean absolute value. High-frequency patterns have large
        Laplacian; smooth patterns have near-zero.

        Args:
            residual: Residual tensor (B, C, H, W), typically encoded - original.

        Returns:
            Scalar Laplacian loss.
        """
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=residual.dtype,
            device=residual.device,
        ).view(1, 1, 3, 3).expand(residual.shape[1], -1, -1, -1)
        laplacian = F.conv2d(residual, kernel, padding=1, groups=residual.shape[1])
        return laplacian.abs().mean()

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
