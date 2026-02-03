"""Blur distortions using Kornia.

Uses kornia.filters for blur operations. API matches native blur.py exactly.
"""

import math

import kornia.filters
import torch
from torch import Tensor

from picode.distortions.base import Distortion


class GaussianBlur(Distortion):
    """Apply Gaussian blur to images using Kornia.

    API-compatible with native.GaussianBlur.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        kernel_size: Size of blur kernel (must be odd).
        sigma: Standard deviation of Gaussian (StegaStamp range: 1.0-3.0).
    """

    name = "gaussian_blur"

    def __init__(
        self,
        intensity: float = 0.5,
        kernel_size: int = 7,
        sigma: float = 2.0,
    ):
        super().__init__(intensity)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        """Apply Gaussian blur using kornia.filters.gaussian_blur2d."""
        if self.intensity == 0.0:
            return x

        effective_sigma = self.sigma * self.intensity
        if effective_sigma < 0.1:
            return x

        return kornia.filters.gaussian_blur2d(
            x,
            kernel_size=(self.kernel_size, self.kernel_size),
            sigma=(effective_sigma, effective_sigma),
        )

    def sample_parameters(self) -> dict:
        """Sample sigma in StegaStamp range [1.0, 3.0]."""
        return {"sigma": 1.0 + torch.rand(1).item() * 2.0}


class MotionBlur(Distortion):
    """Apply motion/line blur to images using Kornia.

    API-compatible with native.MotionBlur.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        kernel_size: Size of blur kernel.
        angle: Direction of motion in radians (default: random).
        sigma: Spread perpendicular to motion (StegaStamp: 0.25-1.0).
    """

    name = "motion_blur"

    def __init__(
        self,
        intensity: float = 0.5,
        kernel_size: int = 7,
        angle: float | None = None,
        sigma: float = 0.5,
    ):
        super().__init__(intensity)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.angle = angle  # None means random
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        """Apply motion blur using kornia.filters.motion_blur.

        Note: Kornia motion_blur uses degrees, native uses radians.
        """
        if self.intensity == 0.0:
            return x

        # Use provided angle or generate random (in radians)
        angle_rad = (
            self.angle if self.angle is not None else torch.rand(1).item() * 2.0 * math.pi
        )

        # Convert radians to degrees for Kornia
        angle_deg = math.degrees(angle_rad)

        # direction parameter: -1 to 1, controls blur bias along the motion direction
        # 0.0 gives symmetric motion blur
        direction = 0.0

        return kornia.filters.motion_blur(
            x,
            kernel_size=self.kernel_size,
            angle=angle_deg,
            direction=direction,
        )

    def sample_parameters(self) -> dict:
        """Sample angle and sigma in StegaStamp ranges."""
        return {
            "angle": torch.rand(1).item() * 2.0 * math.pi,
            "sigma": 0.25 + torch.rand(1).item() * 0.75,
        }


class RandomBlur(Distortion):
    """Probabilistic blur matching StegaStamp's random_blur_kernel.

    Randomly selects between identity, Gaussian, or line/motion blur.
    API-compatible with native.RandomBlur.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        kernel_size: Size of blur kernel (StegaStamp default: 7).
        prob_gauss: Probability of Gaussian blur (StegaStamp: 0.25).
        prob_line: Probability of line/motion blur (StegaStamp: 0.25).
        sigma_gauss_range: Sigma range for Gaussian (StegaStamp: [1.0, 3.0]).
        sigma_line_range: Sigma range for line blur (StegaStamp: [0.25, 1.0]).
    """

    name = "random_blur"

    def __init__(
        self,
        intensity: float = 0.5,
        kernel_size: int = 7,
        prob_gauss: float = 0.25,
        prob_line: float = 0.25,
        sigma_gauss_range: tuple[float, float] = (1.0, 3.0),
        sigma_line_range: tuple[float, float] = (0.25, 1.0),
    ):
        super().__init__(intensity)
        self.kernel_size = kernel_size
        self.prob_gauss = prob_gauss
        self.prob_line = prob_line
        self.sigma_gauss_range = sigma_gauss_range
        self.sigma_line_range = sigma_line_range

    def forward(self, x: Tensor) -> Tensor:
        """Apply random blur using Kornia.

        Matches StegaStamp: probabilistically selects blur type.
        """
        if self.intensity == 0.0:
            return x

        # For partial intensity, probabilistically skip
        if torch.rand(1).item() > self.intensity:
            return x

        # Probabilistic selection (matches StegaStamp logic)
        t = torch.rand(1).item()
        if t < self.prob_gauss:
            # Gaussian blur
            sigma = (
                self.sigma_gauss_range[0]
                + torch.rand(1).item() * (self.sigma_gauss_range[1] - self.sigma_gauss_range[0])
            )
            return kornia.filters.gaussian_blur2d(
                x,
                kernel_size=(self.kernel_size, self.kernel_size),
                sigma=(sigma, sigma),
            )
        elif t < self.prob_gauss + self.prob_line:
            # Motion/line blur
            angle_deg = torch.rand(1).item() * 360.0
            return kornia.filters.motion_blur(
                x,
                kernel_size=self.kernel_size,
                angle=angle_deg,
                direction=0.0,
            )
        else:
            # Identity (no blur)
            return x

    def sample_parameters(self) -> dict:
        """Sample which blur type will be used."""
        t = torch.rand(1).item()
        if t < self.prob_gauss:
            blur_type = "gaussian"
        elif t < self.prob_gauss + self.prob_line:
            blur_type = "line"
        else:
            blur_type = "identity"
        return {"blur_type": blur_type}
