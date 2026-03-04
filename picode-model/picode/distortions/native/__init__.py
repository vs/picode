"""Native (pure PyTorch) distortion implementations.

These implementations match the StegaStamp paper and use no external libraries.
"""

from picode.distortions.native.blur import GaussianBlur, MotionBlur, RandomBlur
from picode.distortions.native.color import BrightnessHue, Contrast, Saturation
from picode.distortions.native.composite import Compose
from picode.distortions.native.compression import JPEGCompression
from picode.distortions.native.geometric import Crop, PerspectiveWarp, Rotation, Scale
from picode.distortions.native.noise import GaussianNoise

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
