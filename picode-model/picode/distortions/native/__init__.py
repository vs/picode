"""Native (pure PyTorch) distortion implementations.

These implementations match the StegaStamp paper and use no external libraries.
"""

from picode.distortions.native.blur import GaussianBlur, MotionBlur, RandomBlur, RandomBlurKernel
from picode.distortions.native.color import BrightnessHue, Contrast, Saturation
from picode.distortions.native.composite import Compose
from picode.distortions.native.compression import JPEGCompression
from picode.distortions.native.geometric import Crop, PerspectiveWarp, Rotation, Scale
from picode.distortions.native.noise import GaussianNoise
from picode.distortions.native.print_photo import (
    BarrelDistortion,
    ChromaticAberration,
    ResolutionLoss,
    ShotNoise,
    Vignetting,
)

__all__ = [
    "BarrelDistortion",
    "BrightnessHue",
    "ChromaticAberration",
    "Compose",
    "Contrast",
    "Crop",
    "GaussianBlur",
    "GaussianNoise",
    "JPEGCompression",
    "MotionBlur",
    "PerspectiveWarp",
    "RandomBlur",
    "RandomBlurKernel",
    "ResolutionLoss",
    "Rotation",
    "Saturation",
    "Scale",
    "ShotNoise",
    "Vignetting",
]
