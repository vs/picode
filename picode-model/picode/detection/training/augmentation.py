# picode/detection/training/augmentation.py
"""Augmentation pipeline for FastDetector training.

Includes domain randomization for sim-to-real transfer:
- Screen capture simulation (moire, color banding)
- Camera artifacts (lens distortion, chromatic aberration, vignetting)
- Environmental factors (reflections, glare, varying lighting)
- Background variations (textures, patterns, gradients)
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor


class PhotometricAugmentation:
    """Photometric augmentations that don't affect corner positions.

    Includes:
        - Brightness adjustment
        - Contrast adjustment
        - Saturation adjustment
        - Gaussian noise
        - Gaussian blur
        - JPEG compression simulation

    Args:
        brightness_range: (min, max) brightness adjustment factor
        contrast_range: (min, max) contrast adjustment factor
        saturation_range: (min, max) saturation adjustment factor
        noise_std: Standard deviation of Gaussian noise
        blur_kernel_range: (min, max) kernel size for blur
        jpeg_quality_range: (min, max) JPEG quality
        p: Probability of applying any augmentation
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        saturation_range: tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.02,
        blur_kernel_range: tuple[int, int] = (3, 7),
        jpeg_quality_range: tuple[int, int] = (50, 95),
        p: float = 0.5,
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.noise_std = noise_std
        self.blur_kernel_range = blur_kernel_range
        self.jpeg_quality_range = jpeg_quality_range
        self.p = p

    def __call__(self, image: Tensor) -> Tensor:
        """Apply photometric augmentations.

        Args:
            image: (C, H, W) tensor in [0, 1]

        Returns:
            Augmented image (C, H, W) in [0, 1]
        """
        if random.random() > self.p:
            return image

        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            image = image * factor

        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Saturation
        if random.random() < 0.5:
            factor = random.uniform(*self.saturation_range)
            gray = image.mean(dim=0, keepdim=True)
            image = image * factor + gray * (1 - factor)

        # Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise

        # Gaussian blur
        if random.random() < 0.3:
            kernel_size = random.choice(
                range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)
            )
            image = self._gaussian_blur(image, kernel_size)

        # Clamp to valid range
        return image.clamp(0.0, 1.0)

    def _gaussian_blur(self, image: Tensor, kernel_size: int) -> Tensor:
        """Apply Gaussian blur."""
        # Create Gaussian kernel
        sigma = kernel_size / 6.0
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.outer(kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size)

        # Apply convolution
        padding = kernel_size // 2
        image = image.unsqueeze(0)  # Add batch dim
        blurred = F.conv2d(
            image,
            kernel_2d.to(image.device),
            padding=padding,
            groups=3,
        )
        return blurred.squeeze(0)


class GeometricAugmentation:
    """Geometric augmentations that require corner adjustment.

    Includes:
        - Perspective transform
        - Rotation

    Args:
        perspective_strength: (min, max) perspective distortion strength
        rotation_degrees: (min, max) rotation angle in degrees
        p: Probability of applying augmentation
    """

    def __init__(
        self,
        perspective_strength: tuple[float, float] = (0.0, 0.1),
        rotation_degrees: tuple[float, float] = (-15, 15),
        p: float = 0.5,
    ) -> None:
        self.perspective_strength = perspective_strength
        self.rotation_degrees = rotation_degrees
        self.p = p

    def __call__(
        self, image: Tensor, corners: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply geometric augmentations.

        Args:
            image: (C, H, W) tensor in [0, 1]
            corners: (8,) tensor of normalized [0, 1] corner coordinates

        Returns:
            Tuple of (augmented image, adjusted corners)
        """
        if random.random() > self.p:
            return image, corners

        # Apply rotation
        if random.random() < 0.5:
            angle = random.uniform(*self.rotation_degrees)
            image, corners = self._rotate(image, corners, angle)

        # Apply perspective
        if random.random() < 0.5:
            strength = random.uniform(*self.perspective_strength)
            corners = self._perturb_corners(corners, strength)

        # Ensure corners remain valid
        corners = corners.clamp(0.0, 1.0)

        return image, corners

    def _rotate(
        self, image: Tensor, corners: Tensor, angle: float
    ) -> tuple[Tensor, Tensor]:
        """Rotate image and corners around center."""
        import math

        # Convert angle to radians
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # Rotate corners around center (0.5, 0.5)
        corners_2d = corners.reshape(4, 2)
        centered = corners_2d - 0.5

        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ])
        rotated = centered @ rotation_matrix.T
        corners_new = (rotated + 0.5).flatten()

        # Rotate image using affine grid
        c, h, w = image.shape
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
        ]).unsqueeze(0).float()

        grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
        image_rotated = F.grid_sample(
            image.unsqueeze(0), grid, align_corners=False, padding_mode="border"
        ).squeeze(0)

        return image_rotated, corners_new

    def _perturb_corners(self, corners: Tensor, strength: float) -> Tensor:
        """Add random perspective distortion to corners."""
        perturbation = (torch.rand(8) - 0.5) * 2 * strength
        return corners + perturbation


class DetectionAugmentation:
    """Combined augmentation pipeline for detection training.

    Applies photometric and geometric augmentations appropriately:
    - Photometric: Applied to all images
    - Geometric: Applied only to positive samples (with corners)

    Args:
        photometric_p: Probability of photometric augmentation
        geometric_p: Probability of geometric augmentation
    """

    def __init__(
        self,
        photometric_p: float = 0.5,
        geometric_p: float = 0.5,
        perspective_strength: tuple[float, float] = (0.0, 0.1),
        rotation_degrees: tuple[float, float] = (-15, 15),
    ) -> None:
        self.photometric = PhotometricAugmentation(p=photometric_p)
        self.geometric = GeometricAugmentation(
            perspective_strength=perspective_strength,
            rotation_degrees=rotation_degrees,
            p=geometric_p,
        )

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply augmentations to a training sample.

        Args:
            sample: Dict with keys:
                - image: (C, H, W) tensor
                - is_watermark: scalar tensor (0 or 1)
                - corners: (8,) tensor of corner coordinates
                - has_corners: scalar tensor (0 or 1)

        Returns:
            Augmented sample with same keys
        """
        image = sample["image"]
        corners = sample["corners"]
        is_positive = sample["has_corners"].item() > 0.5

        # Apply photometric augmentation to all images
        image = self.photometric(image)

        # Apply geometric augmentation only to positive samples
        if is_positive:
            image, corners = self.geometric(image, corners)

        return {
            "image": image,
            "is_watermark": sample["is_watermark"],
            "corners": corners,
            "has_corners": sample["has_corners"],
        }


class DomainRandomization:
    """Domain randomization for sim-to-real transfer.

    Simulates real-world camera capture conditions:
    - Screen moire patterns
    - Color banding / posterization
    - Chromatic aberration
    - Vignetting
    - Screen reflections / glare
    - Color temperature shifts
    - Gamma variations

    Args:
        moire_p: Probability of moire pattern
        banding_p: Probability of color banding
        chromatic_p: Probability of chromatic aberration
        vignette_p: Probability of vignetting
        reflection_p: Probability of screen reflections
        color_temp_p: Probability of color temperature shift
        gamma_p: Probability of gamma variation
    """

    def __init__(
        self,
        moire_p: float = 0.3,
        banding_p: float = 0.2,
        chromatic_p: float = 0.3,
        vignette_p: float = 0.4,
        reflection_p: float = 0.2,
        color_temp_p: float = 0.4,
        gamma_p: float = 0.3,
    ) -> None:
        self.moire_p = moire_p
        self.banding_p = banding_p
        self.chromatic_p = chromatic_p
        self.vignette_p = vignette_p
        self.reflection_p = reflection_p
        self.color_temp_p = color_temp_p
        self.gamma_p = gamma_p

    def __call__(self, image: Tensor) -> Tensor:
        """Apply domain randomization augmentations.

        Args:
            image: (C, H, W) tensor in [0, 1]

        Returns:
            Augmented image (C, H, W) in [0, 1]
        """
        # Moire pattern (screen capture artifact)
        if random.random() < self.moire_p:
            image = self._add_moire(image)

        # Color banding / posterization
        if random.random() < self.banding_p:
            image = self._add_banding(image)

        # Chromatic aberration
        if random.random() < self.chromatic_p:
            image = self._add_chromatic_aberration(image)

        # Vignetting
        if random.random() < self.vignette_p:
            image = self._add_vignette(image)

        # Screen reflections / glare
        if random.random() < self.reflection_p:
            image = self._add_reflection(image)

        # Color temperature shift
        if random.random() < self.color_temp_p:
            image = self._shift_color_temperature(image)

        # Gamma variation
        if random.random() < self.gamma_p:
            image = self._vary_gamma(image)

        return image.clamp(0.0, 1.0)

    def _add_moire(self, image: Tensor) -> Tensor:
        """Add moire pattern to simulate screen capture."""
        c, h, w = image.shape

        # Create interference pattern
        freq_x = random.uniform(0.1, 0.5)
        freq_y = random.uniform(0.1, 0.5)
        phase_x = random.uniform(0, 2 * math.pi)
        phase_y = random.uniform(0, 2 * math.pi)

        y_coords = torch.linspace(0, h * freq_y, h)
        x_coords = torch.linspace(0, w * freq_x, w)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        pattern = torch.sin(xx * 2 * math.pi + phase_x) * torch.sin(yy * 2 * math.pi + phase_y)
        pattern = pattern.unsqueeze(0).expand(c, -1, -1).to(image.device)

        # Apply subtle moire
        strength = random.uniform(0.02, 0.08)
        return image + pattern * strength

    def _add_banding(self, image: Tensor) -> Tensor:
        """Add color banding / posterization effect."""
        # Reduce bit depth simulation
        levels = random.randint(16, 64)
        quantized = torch.round(image * levels) / levels

        # Blend with original
        blend = random.uniform(0.3, 0.7)
        return image * (1 - blend) + quantized * blend

    def _add_chromatic_aberration(self, image: Tensor) -> Tensor:
        """Add chromatic aberration (color fringing)."""
        c, h, w = image.shape
        if c != 3:
            return image

        # Shift red and blue channels slightly
        shift = random.randint(1, 3)

        # Create shifted versions
        r_channel = image[0:1]
        g_channel = image[1:2]
        b_channel = image[2:3]

        # Shift red outward, blue inward (or vice versa)
        # Tensors are (1, H, W) — pad last dim, slice with 3D indexing
        if random.random() < 0.5:
            r_shifted = F.pad(r_channel, (shift, 0, 0, 0))[:, :, :w]
            b_shifted = F.pad(b_channel, (0, shift, 0, 0))[:, :, shift:]
        else:
            r_shifted = F.pad(r_channel, (0, shift, 0, 0))[:, :, shift:]
            b_shifted = F.pad(b_channel, (shift, 0, 0, 0))[:, :, :w]

        return torch.cat([r_shifted, g_channel, b_shifted], dim=0)

    def _add_vignette(self, image: Tensor) -> Tensor:
        """Add vignetting (darker corners)."""
        c, h, w = image.shape

        # Create radial gradient from center
        y = torch.linspace(-1, 1, h)
        x = torch.linspace(-1, 1, w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(xx**2 + yy**2)

        # Random vignette strength and falloff
        strength = random.uniform(0.1, 0.4)
        falloff = random.uniform(0.5, 1.5)

        vignette = 1 - strength * (dist ** falloff)
        vignette = vignette.clamp(0.3, 1.0)  # Don't make corners too dark
        vignette = vignette.unsqueeze(0).expand(c, -1, -1).to(image.device)

        return image * vignette

    def _add_reflection(self, image: Tensor) -> Tensor:
        """Add screen reflection / glare effect."""
        c, h, w = image.shape

        # Create gradient for reflection
        y = torch.linspace(0, 1, h)
        x = torch.linspace(0, 1, w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Random reflection position and angle
        angle = random.uniform(0, math.pi)
        offset = random.uniform(-0.5, 0.5)

        reflection = torch.cos(angle) * xx + torch.sin(angle) * yy + offset
        reflection = torch.sigmoid(reflection * random.uniform(2, 5))

        # Make it subtle and additive (like light reflection)
        strength = random.uniform(0.05, 0.15)
        reflection = reflection.unsqueeze(0).expand(c, -1, -1).to(image.device)

        return image + reflection * strength

    def _shift_color_temperature(self, image: Tensor) -> Tensor:
        """Shift color temperature (warm/cool)."""
        c, h, w = image.shape
        if c != 3:
            return image

        # Random temperature shift
        temp_shift = random.uniform(-0.1, 0.1)

        # Warm = more red/yellow, Cool = more blue
        r_mult = 1 + temp_shift
        b_mult = 1 - temp_shift

        result = image.clone()
        result[0] = image[0] * r_mult
        result[2] = image[2] * b_mult

        return result

    def _vary_gamma(self, image: Tensor) -> Tensor:
        """Apply gamma variation."""
        gamma = random.uniform(0.8, 1.2)
        return image.pow(gamma)


class BackgroundRandomization:
    """Generate random backgrounds for negative samples.

    Creates varied backgrounds that could appear in real-world photos:
    - Solid colors
    - Gradients
    - Noise patterns
    - Textures (synthetic)

    Args:
        size: Output image size (H, W)
    """

    def __init__(self, size: tuple[int, int] = (224, 224)) -> None:
        self.size = size

    def __call__(self) -> Tensor:
        """Generate a random background.

        Returns:
            Background image (3, H, W) in [0, 1]
        """
        bg_type = random.choice(["solid", "gradient", "noise", "texture"])

        if bg_type == "solid":
            return self._solid_background()
        elif bg_type == "gradient":
            return self._gradient_background()
        elif bg_type == "noise":
            return self._noise_background()
        else:
            return self._texture_background()

    def _solid_background(self) -> Tensor:
        """Generate solid color background."""
        h, w = self.size
        color = torch.rand(3, 1, 1)
        return color.expand(3, h, w)

    def _gradient_background(self) -> Tensor:
        """Generate gradient background."""
        h, w = self.size

        # Two random colors
        color1 = torch.rand(3)
        color2 = torch.rand(3)

        # Random gradient direction
        angle = random.uniform(0, math.pi)

        y = torch.linspace(0, 1, h)
        x = torch.linspace(0, 1, w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Interpolation factor based on angle
        t = (math.cos(angle) * xx + math.sin(angle) * yy + 1) / 2
        t = t.unsqueeze(0).expand(3, -1, -1)

        color1 = color1.view(3, 1, 1).expand(3, h, w)
        color2 = color2.view(3, 1, 1).expand(3, h, w)

        return color1 * (1 - t) + color2 * t

    def _noise_background(self) -> Tensor:
        """Generate noise pattern background."""
        h, w = self.size

        # Base noise
        noise = torch.rand(3, h, w)

        # Optional: blur for smoother appearance
        if random.random() < 0.5:
            noise = F.avg_pool2d(noise.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)

        return noise

    def _texture_background(self) -> Tensor:
        """Generate synthetic texture background."""
        h, w = self.size

        # Create base pattern
        freq = random.uniform(5, 20)
        y = torch.linspace(0, freq, h)
        x = torch.linspace(0, freq, w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Combine multiple patterns
        patterns = []
        for _ in range(random.randint(1, 3)):
            phase = random.uniform(0, 2 * math.pi)
            pattern = torch.sin(xx * random.uniform(0.5, 2) + yy * random.uniform(0.5, 2) + phase)
            patterns.append(pattern)

        combined = sum(patterns) / len(patterns)
        combined = (combined + 1) / 2  # Normalize to [0, 1]

        # Add color
        base_color = torch.rand(3, 1, 1)
        variation = combined.unsqueeze(0) * random.uniform(0.1, 0.3)

        return (base_color + variation - 0.15).clamp(0, 1).expand(3, h, w)


class DomainRandomizedAugmentation:
    """Full augmentation pipeline with domain randomization.

    Combines all augmentation strategies for maximum sim-to-real transfer:
    - Photometric augmentations
    - Geometric augmentations
    - Domain randomization (camera/screen artifacts)
    - Background randomization (for negative samples)

    Args:
        photometric_p: Probability of photometric augmentation
        geometric_p: Probability of geometric augmentation
        domain_random_p: Probability of domain randomization effects
    """

    def __init__(
        self,
        photometric_p: float = 0.7,
        geometric_p: float = 0.5,
        domain_random_p: float = 0.6,
        perspective_strength: tuple[float, float] = (0.0, 0.15),
        rotation_degrees: tuple[float, float] = (-30, 30),
    ) -> None:
        self.photometric = PhotometricAugmentation(
            p=photometric_p,
            brightness_range=(0.6, 1.4),  # More extreme
            contrast_range=(0.6, 1.4),
            saturation_range=(0.6, 1.4),
            noise_std=0.04,  # More noise
            blur_kernel_range=(3, 9),
            jpeg_quality_range=(30, 95),  # Lower quality possible
        )
        self.geometric = GeometricAugmentation(
            perspective_strength=perspective_strength,
            rotation_degrees=rotation_degrees,
            p=geometric_p,
        )
        self.domain = DomainRandomization()
        self.domain_p = domain_random_p
        self.background = BackgroundRandomization()

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply full augmentation pipeline.

        Args:
            sample: Dict with keys:
                - image: (C, H, W) tensor
                - is_watermark: scalar tensor (0 or 1)
                - corners: (8,) tensor of corner coordinates
                - has_corners: scalar tensor (0 or 1)

        Returns:
            Augmented sample with same keys
        """
        image = sample["image"]
        corners = sample["corners"]
        is_positive = sample["has_corners"].item() > 0.5

        # Apply photometric augmentation
        image = self.photometric(image)

        # Apply domain randomization
        if random.random() < self.domain_p:
            image = self.domain(image)

        # Apply geometric augmentation only to positive samples
        if is_positive:
            image, corners = self.geometric(image, corners)

        return {
            "image": image.clamp(0, 1),
            "is_watermark": sample["is_watermark"],
            "corners": corners.clamp(0, 1),
            "has_corners": sample["has_corners"],
        }
