"""Kornia-based distortion implementations.

These implementations use Kornia for GPU-optimized operations.
API is identical to native backend - this is a drop-in replacement.
"""

try:
    import kornia
except ImportError as e:
    raise ImportError(
        "Kornia backend requires kornia. Install with: pip install picode[kornia]"
    ) from e

from picode.distortions.kornia.blur import GaussianBlur, MotionBlur, RandomBlur
from picode.distortions.kornia.color import BrightnessHue, Contrast, Saturation
from picode.distortions.kornia.composite import Compose
from picode.distortions.kornia.compression import JPEGCompression
from picode.distortions.kornia.geometric import Crop, PerspectiveWarp, Rotation, Scale
from picode.distortions.kornia.noise import GaussianNoise

__all__ = [
    "BrightnessHue",
    "Compose",
    "Contrast",
    "Crop",
    "GaussianBlur",
    "GaussianNoise",
    "JPEGCompression",
    "MotionBlur",
    "PerspectiveWarp",
    "RandomBlur",
    "Rotation",
    "Saturation",
    "Scale",
]
