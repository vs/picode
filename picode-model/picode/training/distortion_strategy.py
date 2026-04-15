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
    GaussianBlur,
    GaussianNoise,
    JPEGCompression,
    PerspectiveWarp,
    Rotation,
    Saturation,
    Scale,
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


class FixedLightDistortion:
    """Light distortions for print-scan scenarios with probabilistic application.

    Designed for imperceptibility-first training (picode_v3). Each distortion
    is applied independently with its own probability, mimicking real-world
    print-scan pipeline at mild intensities.

    Application order: Color → Blur → Noise → Geometric → JPEG

    Default parameters (from picode_v3 design):
    - JPEG: quality 75-95, p=0.8
    - Rotation: ±3°, p=0.5
    - Rescale: 0.85-1.0x, p=0.4
    - Gaussian Blur: σ 0.5-1.0, p=0.5
    - Color Jitter: brightness ±0.05, contrast ±0.05, saturation ±0.1, p=0.6
    - Gaussian Noise: σ 0.01-0.02, p=0.4
    """

    def __init__(
        self,
        jpeg_quality: tuple[int, int] = (75, 95),
        jpeg_prob: float = 0.8,
        rotation_angle: float = 3.0,
        rotation_prob: float = 0.5,
        scale_range: tuple[float, float] = (0.85, 1.0),
        scale_prob: float = 0.4,
        blur_sigma: tuple[float, float] = (0.5, 1.0),
        blur_prob: float = 0.5,
        brightness: float = 0.05,
        contrast: float = 0.05,
        saturation: float = 0.1,
        color_prob: float = 0.6,
        noise_sigma: tuple[float, float] = (0.01, 0.02),
        noise_prob: float = 0.4,
    ) -> None:
        self.jpeg_quality = jpeg_quality
        self.jpeg_prob = jpeg_prob
        self.rotation_angle = rotation_angle
        self.rotation_prob = rotation_prob
        self.scale_range = scale_range
        self.scale_prob = scale_prob
        self.blur_sigma = blur_sigma
        self.blur_prob = blur_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.color_prob = color_prob
        self.noise_sigma = noise_sigma
        self.noise_prob = noise_prob

    def __call__(self, image: Tensor, step: int) -> Tensor:
        """Apply light distortions with probabilities.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].
            step: Current training step (unused, kept for interface compatibility).

        Returns:
            Distorted tensor.
        """
        x = image

        # 1. Color jitter (brightness, contrast, saturation)
        if random.random() < self.color_prob:
            distortions: list[Distortion] = []
            if self.brightness > 0:
                distortions.append(
                    BrightnessHue(intensity=1.0, rnd_bri=self.brightness, rnd_hue=0.0)
                )
            if self.contrast > 0:
                # contrast ±0.05 means range [0.95, 1.05]
                distortions.append(Contrast(
                    intensity=1.0,
                    contrast_low=1.0 - self.contrast,
                    contrast_high=1.0 + self.contrast,
                ))
            if self.saturation > 0:
                distortions.append(Saturation(intensity=1.0, rnd_sat=self.saturation))
            if distortions:
                x = cast(Tensor, Compose(distortions)(x))

        # 2. Gaussian blur
        if random.random() < self.blur_prob:
            # Sample sigma from range
            sigma = self.blur_sigma[0] + random.random() * (self.blur_sigma[1] - self.blur_sigma[0])
            x = cast(Tensor, GaussianBlur(intensity=1.0, sigma=sigma)(x))

        # 3. Gaussian noise
        if random.random() < self.noise_prob:
            # Sample sigma from range
            noise_range = self.noise_sigma[1] - self.noise_sigma[0]
            std = self.noise_sigma[0] + random.random() * noise_range
            x = cast(Tensor, GaussianNoise(intensity=1.0, std=std)(x))

        # 4. Geometric: Rotation
        if random.random() < self.rotation_prob:
            x = cast(Tensor, Rotation(intensity=1.0, max_angle=self.rotation_angle)(x))

        # 5. Geometric: Scale (rescale)
        if random.random() < self.scale_prob:
            x = cast(Tensor, Scale(
                intensity=1.0,
                min_scale=self.scale_range[0],
                max_scale=self.scale_range[1],
            )(x))

        # 6. JPEG compression
        if random.random() < self.jpeg_prob:
            # Sample quality from range
            quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
            x = cast(Tensor, JPEGCompression(intensity=1.0, quality=quality)(x))

        return x


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
    elif config.strategy == "fixed_light":
        # Light distortions for picode_v3 / imperceptibility-first training
        return FixedLightDistortion()
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
