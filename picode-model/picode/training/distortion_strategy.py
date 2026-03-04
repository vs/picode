"""Distortion strategies for training.

Provides different strategies for applying distortions during training:
- NoDistortion: Identity (no distortion applied)
- FixedDistortion: Same distortion(s) at fixed strength
- RandomDistortion: Randomly sample from pool of distortions
- CurriculumDistortion: Ramp distortion strengths over training steps
"""

import random
from typing import Protocol, cast

from torch import Tensor

from picode.distortions.base import Distortion
from picode.distortions.native import (
    BrightnessHue,
    Compose,
    Contrast,
    GaussianNoise,
    JPEGCompression,
    PerspectiveWarp,
    Saturation,
)
from picode.training.config import DistortionConfig, DistortionRamp


class DistortionStrategy(Protocol):
    """Protocol for distortion strategies."""

    def __call__(self, image: Tensor, step: int) -> Tensor:
        """Apply distortion to image at given training step.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].
            step: Current training step.

        Returns:
            Distorted tensor.
        """
        ...


class NoDistortion:
    """Identity - no distortion applied."""

    def __call__(self, image: Tensor, step: int) -> Tensor:
        """Return image unchanged.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].
            step: Current training step (unused).

        Returns:
            Same image tensor.
        """
        return image


class FixedDistortion:
    """Apply same distortion(s) every batch at fixed strength.

    Args:
        distortions: List of Distortion instances to apply.
    """

    def __init__(self, distortions: list[Distortion]) -> None:
        self.compose = Compose(distortions)

    def __call__(self, image: Tensor, step: int) -> Tensor:
        """Apply fixed distortions.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].
            step: Current training step (unused).

        Returns:
            Distorted tensor.
        """
        return cast(Tensor, self.compose(image))


class RandomDistortion:
    """Randomly sample from pool of distortions each batch.

    Args:
        distortions: Pool of Distortion instances to sample from.
        num_apply: Number of distortions to apply. Can be an int or
            a tuple (min, max) for random range.
    """

    def __init__(
        self,
        distortions: list[Distortion],
        num_apply: int | tuple[int, int] = (1, 3),
    ) -> None:
        self.distortions = distortions
        self.num_apply = num_apply

    def __call__(self, image: Tensor, step: int) -> Tensor:
        """Apply random subset of distortions.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].
            step: Current training step (unused).

        Returns:
            Distorted tensor.
        """
        if isinstance(self.num_apply, tuple):
            n = random.randint(self.num_apply[0], self.num_apply[1])
        else:
            n = self.num_apply

        selected = random.sample(self.distortions, min(n, len(self.distortions)))
        return cast(Tensor, Compose(selected)(image))


class CurriculumDistortion:
    """Ramp distortion strengths over training steps.

    Gradually increases distortion strength from 0 to the configured
    strength over the specified ramp_steps. This allows the model to
    first learn basic encoding before facing stronger distortions.

    Args:
        config: DistortionConfig with ramp parameters.
    """

    def __init__(self, config: DistortionConfig) -> None:
        self.config = config

    def _ramp(self, ramp: DistortionRamp, step: int) -> float:
        """Compute ramped strength at current step.

        Args:
            ramp: DistortionRamp with strength and ramp_steps.
            step: Current training step.

        Returns:
            Ramped strength value.
        """
        if ramp.ramp_steps <= 0:
            return ramp.strength
        return min(ramp.strength * step / ramp.ramp_steps, ramp.strength)

    def __call__(self, image: Tensor, step: int) -> Tensor:
        """Apply distortions with ramped strengths.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].
            step: Current training step.

        Returns:
            Distorted tensor.
        """
        distortions: list[Distortion] = []

        # Noise - ramp the std parameter
        noise_std = self._ramp(self.config.noise, step)
        if noise_std > 0:
            distortions.append(GaussianNoise(intensity=1.0, std=noise_std))

        # Brightness and Hue - ramp rnd_bri and rnd_hue
        bri_strength = self._ramp(self.config.brightness, step)
        hue_strength = self._ramp(self.config.hue, step)
        if bri_strength > 0 or hue_strength > 0:
            distortions.append(BrightnessHue(
                intensity=1.0,
                rnd_bri=bri_strength,
                rnd_hue=hue_strength,
            ))

        # Saturation - ramp rnd_sat
        sat_strength = self._ramp(self.config.saturation, step)
        if sat_strength > 0:
            distortions.append(Saturation(intensity=1.0, rnd_sat=sat_strength))

        # Contrast - ramp the contrast range
        contrast_t = min(step / self.config.contrast_ramp_steps, 1.0)
        low = 1.0 + contrast_t * (self.config.contrast[0] - 1.0)
        high = 1.0 + contrast_t * (self.config.contrast[1] - 1.0)
        if low != 1.0 or high != 1.0:
            distortions.append(Contrast(intensity=1.0, contrast_low=low, contrast_high=high))

        # JPEG - ramp quality (lower quality = more distortion)
        if self.config.enable_jpeg:
            jpeg_strength = self._ramp(self.config.jpeg_quality, step)
            if jpeg_strength > 0:
                # jpeg_quality.strength is the amount to reduce from 100
                # So quality = 100 - jpeg_strength
                quality = max(5, int(100 - jpeg_strength))
                distortions.append(JPEGCompression(intensity=1.0, quality=quality))

        # Perspective - ramp scale parameter
        persp_strength = self._ramp(self.config.perspective, step)
        if persp_strength > 0:
            distortions.append(PerspectiveWarp(intensity=1.0, scale=persp_strength))

        if not distortions:
            return image

        return cast(Tensor, Compose(distortions)(image))


def create_distortion_strategy(config: DistortionConfig) -> DistortionStrategy:
    """Factory function to create strategy from config.

    Args:
        config: DistortionConfig specifying strategy type and parameters.

    Returns:
        Appropriate DistortionStrategy instance.

    Raises:
        ValueError: If strategy type is unknown.
    """
    if config.strategy == "none":
        return NoDistortion()
    elif config.strategy == "curriculum":
        return CurriculumDistortion(config)
    elif config.strategy == "fixed":
        # Build fixed distortions from config
        distortions: list[Distortion] = []
        if config.noise.strength > 0:
            distortions.append(GaussianNoise(intensity=1.0, std=config.noise.strength))
        if config.brightness.strength > 0 or config.hue.strength > 0:
            distortions.append(BrightnessHue(
                intensity=1.0,
                rnd_bri=config.brightness.strength,
                rnd_hue=config.hue.strength,
            ))
        if config.saturation.strength > 0:
            distortions.append(Saturation(intensity=1.0, rnd_sat=config.saturation.strength))
        if config.enable_jpeg and config.jpeg_quality.strength > 0:
            quality = max(5, int(100 - config.jpeg_quality.strength))
            distortions.append(JPEGCompression(intensity=1.0, quality=quality))
        return FixedDistortion(distortions)
    elif config.strategy == "random":
        # Build pool of distortions
        pool: list[Distortion] = []
        if config.noise.strength > 0:
            pool.append(GaussianNoise(intensity=1.0, std=config.noise.strength))
        if config.brightness.strength > 0 or config.hue.strength > 0:
            pool.append(BrightnessHue(
                intensity=1.0,
                rnd_bri=config.brightness.strength,
                rnd_hue=config.hue.strength,
            ))
        if config.saturation.strength > 0:
            pool.append(Saturation(intensity=1.0, rnd_sat=config.saturation.strength))
        if config.enable_jpeg and config.jpeg_quality.strength > 0:
            quality = max(5, int(100 - config.jpeg_quality.strength))
            pool.append(JPEGCompression(intensity=1.0, quality=quality))
        return RandomDistortion(pool)
    else:
        raise ValueError(f"Unknown distortion strategy: {config.strategy}")
