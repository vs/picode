"""Color distortions.

Matches StegaStamp implementation:
- Brightness/Hue (utils.py:77-80): Additive per-channel + global shift
- Contrast (models.py:148-150): Simple multiplicative scaling
- Saturation (models.py:156-157): Lerp between color and luminance
"""

import torch
from torch import Tensor

from picode.distortions.base import Distortion


class BrightnessHue(Distortion):
    """Adjust brightness and hue together.

    Matches StegaStamp's get_rnd_brightness_tf exactly:
    - rnd_hue: per-channel additive shift (like hue rotation in RGB space)
    - rnd_brightness: global additive shift

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        rnd_bri: Max brightness shift (StegaStamp default: 0.3).
        rnd_hue: Max per-channel shift (StegaStamp default: 0.1).
    """

    name = "brightness_hue"

    def __init__(
        self,
        intensity: float = 0.5,
        rnd_bri: float = 0.3,
        rnd_hue: float = 0.1,
    ):
        super().__init__(intensity)
        self.rnd_bri = rnd_bri
        self.rnd_hue = rnd_hue

    def forward(self, x: Tensor) -> Tensor:
        """Apply brightness and hue adjustment.

        Matches StegaStamp utils.py:77-80 and models.py:152-153:
        1. Generate per-channel hue shift: uniform(-rnd_hue, rnd_hue) for each channel
        2. Generate global brightness shift: uniform(-rnd_bri, rnd_bri)
        3. Add both to image
        4. Clip to [0, 1]

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Adjusted tensor clamped to [0, 1].
        """
        if self.intensity == 0.0:
            return x

        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        # Scale ranges by intensity
        effective_hue = self.rnd_hue * self.intensity
        effective_bri = self.rnd_bri * self.intensity

        # Per-channel hue shift: (B, 3, 1, 1)
        rnd_hue = (torch.rand(b, c, 1, 1, device=device, dtype=dtype) * 2 - 1) * effective_hue

        # Global brightness shift: (B, 1, 1, 1)
        rnd_brightness = (
            torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
        ) * effective_bri

        # Add both (StegaStamp: rnd_hue + rnd_brightness added to image)
        adjusted = x + rnd_hue + rnd_brightness

        return torch.clamp(adjusted, 0.0, 1.0)

    def sample_parameters(self) -> dict:
        """Sample random parameters."""
        return {
            "rnd_bri": torch.rand(1).item() * self.rnd_bri,
            "rnd_hue": torch.rand(1).item() * self.rnd_hue,
        }


class Contrast(Distortion):
    """Adjust image contrast.

    Matches StegaStamp models.py:148-150:
    contrast_scale = uniform(contrast_low, contrast_high)
    encoded_image = encoded_image * contrast_scale

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        contrast_low: Min contrast factor (StegaStamp default: 0.5).
        contrast_high: Max contrast factor (StegaStamp default: 1.5).
    """

    name = "contrast"

    def __init__(
        self,
        intensity: float = 0.5,
        contrast_low: float = 0.5,
        contrast_high: float = 1.5,
    ):
        super().__init__(intensity)
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high

    def forward(self, x: Tensor) -> Tensor:
        """Apply contrast adjustment.

        Matches StegaStamp exactly: simple multiplicative scaling.
        At intensity=0, contrast_scale=1.0 (no change).
        At intensity=1, full range [contrast_low, contrast_high].

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Contrast-adjusted tensor (NOT clamped, done after brightness).
        """
        if self.intensity == 0.0:
            return x

        b = x.shape[0]

        # StegaStamp ramps the contrast range based on training step
        # We use intensity to interpolate the range
        effective_low = 1.0 - (1.0 - self.contrast_low) * self.intensity
        effective_high = 1.0 + (self.contrast_high - 1.0) * self.intensity

        # Sample contrast scale per batch element
        contrast_scale = effective_low + torch.rand(b, 1, 1, 1, device=x.device, dtype=x.dtype) * (
            effective_high - effective_low
        )

        return x * contrast_scale

    def sample_parameters(self) -> dict:
        """Sample random contrast scale."""
        return {
            "contrast_scale": self.contrast_low
            + torch.rand(1).item() * (self.contrast_high - self.contrast_low)
        }


class Saturation(Distortion):
    """Adjust image saturation using luminance-based desaturation.

    Matches StegaStamp models.py:156-157:
    encoded_image_lum = sum(encoded_image * [.3, .6, .1], axis=channels)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    Note: StegaStamp's formula is inverted - higher rnd_sat = LESS saturation.
    rnd_sat=0 means full color, rnd_sat=1 means grayscale.

    This luminance-based approach is ~10x faster than true HSV saturation
    (used by kornia backend) because it avoids color space conversions.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        rnd_sat: Max desaturation factor (StegaStamp default: 1.0).
    """

    name = "saturation"

    def __init__(self, intensity: float = 0.5, rnd_sat: float = 1.0):
        super().__init__(intensity)
        self.rnd_sat = rnd_sat

    def forward(self, x: Tensor) -> Tensor:
        """Apply saturation adjustment.

        Matches StegaStamp exactly:
        1. Compute luminance using [0.3, 0.6, 0.1] weights
        2. Lerp between color (original) and grayscale (luminance)
        3. rnd_sat controls interpolation (higher = more grayscale)

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Saturation-adjusted tensor.
        """
        if self.intensity == 0.0:
            return x

        # StegaStamp luminance weights
        weights = torch.tensor([0.3, 0.6, 0.1], device=x.device, dtype=x.dtype)
        weights = weights.view(1, 3, 1, 1)

        # Compute luminance (grayscale) - keep as 3 channels for broadcasting
        lum = (x * weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        lum = lum.expand_as(x)  # (B, 3, H, W)

        # Sample saturation factor
        effective_sat = self.rnd_sat * self.intensity
        rnd_sat = torch.rand(1, device=x.device, dtype=x.dtype).item() * effective_sat

        # Lerp: (1 - rnd_sat) * color + rnd_sat * grayscale
        adjusted = (1 - rnd_sat) * x + rnd_sat * lum

        return adjusted

    def sample_parameters(self) -> dict:
        """Sample random saturation factor."""
        return {"rnd_sat": torch.rand(1).item() * self.rnd_sat}
