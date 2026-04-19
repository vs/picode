"""Blur distortions.

Matches StegaStamp implementation in utils.py:8-43 (random_blur_kernel).
StegaStamp uses probabilistic selection between:
- Identity (no blur): 50% probability
- Gaussian blur: 25% probability, sigma in [1.0, 3.0]
- Line/motion blur: 25% probability, random angle, sigma in [0.25, 1.0]
"""

import math
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from picode.distortions.base import Distortion


class RandomBlurKernel(Distortion):
    """Random blur kernel matching StegaStamp's random_blur_kernel.

    Randomly selects between:
    - Gaussian blur (probs[0])
    - Line/motion blur (probs[1])
    - Identity/no blur (1 - probs[0] - probs[1])

    StegaStamp default: 25% Gaussian, 25% line, 50% identity.

    Args:
        probs: Tuple of (prob_gaussian, prob_line). Remaining probability is identity.
        kernel_size: Size of blur kernel (StegaStamp default: 7).
        sigma_range_gauss: Sigma range for Gaussian blur (StegaStamp: [1.0, 3.0]).
        sigma_range_line: Sigma range for line blur (StegaStamp: [0.25, 1.0]).
        min_line_width: Minimum pixels set in line kernel to avoid empty kernels.
    """

    name = "random_blur_kernel"

    def __init__(
        self,
        probs: tuple[float, float] = (0.25, 0.25),
        kernel_size: int = 7,
        sigma_range_gauss: tuple[float, float] = (1.0, 3.0),
        sigma_range_line: tuple[float, float] = (0.25, 1.0),
        min_line_width: int = 3,
    ):
        super().__init__(intensity=1.0)  # Always apply, probability handled internally
        self.probs = probs
        self.kernel_size = kernel_size
        self.sigma_range_gauss = sigma_range_gauss
        self.sigma_range_line = sigma_range_line
        self.min_line_width = min_line_width

    def forward(self, image: Tensor) -> Tensor:
        """Apply random blur kernel to image.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Blurred tensor.
        """
        kernel = self._random_kernel(image.device, image.dtype)
        # Apply same kernel to all channels via groups
        B, C, H, W = image.shape
        # Expand kernel for all channels: (C, 1, kernel_size, kernel_size)
        kernel_expanded = kernel.expand(C, 1, self.kernel_size, self.kernel_size)
        # Pad for SAME output size
        pad = self.kernel_size // 2
        image_padded = F.pad(image, [pad, pad, pad, pad], mode="reflect")
        # Conv2d with groups=C applies same kernel to each channel independently
        blurred = F.conv2d(image_padded, kernel_expanded, groups=C)
        return blurred

    def _random_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Generate random blur kernel.

        Returns:
            Kernel of shape (1, 1, kernel_size, kernel_size).
        """
        r = random.random()
        if r < self.probs[0]:
            return self._gaussian_kernel(device, dtype)
        elif r < self.probs[0] + self.probs[1]:
            return self._line_kernel(device, dtype)
        else:
            return self._identity_kernel(device, dtype)

    def _identity_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Identity kernel (no blur).

        Returns:
            Kernel of shape (1, 1, kernel_size, kernel_size).
        """
        kernel = torch.zeros(1, 1, self.kernel_size, self.kernel_size, device=device, dtype=dtype)
        center = self.kernel_size // 2
        kernel[0, 0, center, center] = 1.0
        return kernel

    def _gaussian_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Random Gaussian blur kernel.

        Returns:
            Kernel of shape (1, 1, kernel_size, kernel_size).
        """
        sigma = random.uniform(*self.sigma_range_gauss)

        # Create 1D Gaussian
        x = torch.arange(self.kernel_size, device=device, dtype=dtype) - self.kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Outer product for 2D kernel
        kernel = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        return kernel.unsqueeze(0).unsqueeze(0)

    def _line_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Random line/motion blur kernel matching StegaStamp.

        Returns:
            Kernel of shape (1, 1, kernel_size, kernel_size).
        """
        sigma = random.uniform(*self.sigma_range_line)
        angle = random.uniform(0, 2 * math.pi)

        N = self.kernel_size
        center = N // 2

        # Create coordinate grid centered at origin (matches StegaStamp)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(N, device=device, dtype=dtype),
                torch.arange(N, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        ) - (0.5 * (N - 1))

        manhat = coords.abs().sum(dim=-1)

        # Direction vector
        v = torch.tensor([math.cos(angle), math.sin(angle)], device=device, dtype=dtype)
        dists = (coords * v).sum(dim=-1)

        # Line width (StegaStamp uses random width)
        w_line = 3.0 + random.random() * (0.5 * (N - 1) - 3.0 + 0.1)

        # Gaussian weight perpendicular to line, with mask for line width
        kernel = torch.exp(-dists**2 / (2.0 * sigma**2)) * (manhat < w_line).float()

        # Ensure at least min_line_width pixels are set
        if kernel.sum() < self.min_line_width:
            kernel[center, center] = 1.0

        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample which blur type will be used."""
        r = random.random()
        if r < self.probs[0]:
            blur_type = "gaussian"
        elif r < self.probs[0] + self.probs[1]:
            blur_type = "line"
        else:
            blur_type = "identity"
        return {"blur_type": blur_type}


class RandomBlur(Distortion):
    """Probabilistic blur matching StegaStamp's random_blur_kernel.

    Randomly selects between identity, Gaussian, or line/motion blur.
    This matches the StegaStamp training augmentation exactly.

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

    def _create_blur_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create blur kernel matching StegaStamp's random_blur_kernel.

        Returns kernel of shape (3, 3, kernel_size, kernel_size) for conv2d.
        """
        N = self.kernel_size

        # Create coordinate grid centered at origin
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(N, device=device, dtype=dtype),
                torch.arange(N, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        ) - (0.5 * (N - 1))

        manhat = coords.abs().sum(dim=-1)

        # Identity kernel (no blur)
        vals_nothing = (manhat < 0.5).float()

        # Gaussian kernel
        sig_gauss = (
            self.sigma_gauss_range[0]
            + torch.rand(1, device=device).item()
            * (self.sigma_gauss_range[1] - self.sigma_gauss_range[0])
        )
        vals_gauss = torch.exp(-(coords**2).sum(dim=-1) / (2.0 * sig_gauss**2))

        # Line/motion blur kernel
        theta = torch.rand(1, device=device).item() * 2.0 * math.pi
        v = torch.tensor([math.cos(theta), math.sin(theta)], device=device, dtype=dtype)
        dists = (coords * v).sum(dim=-1)

        sig_line = (
            self.sigma_line_range[0]
            + torch.rand(1, device=device).item()
            * (self.sigma_line_range[1] - self.sigma_line_range[0])
        )
        w_line = 3.0 + torch.rand(1, device=device).item() * (0.5 * (N - 1) - 3.0 + 0.1)
        vals_line = torch.exp(-dists**2 / (2.0 * sig_line**2)) * (manhat < w_line).float()

        # Probabilistic selection (matches StegaStamp logic)
        t = torch.rand(1, device=device).item()
        if t < self.prob_gauss:
            vals = vals_gauss
        elif t < self.prob_gauss + self.prob_line:
            vals = vals_line
        else:
            vals = vals_nothing

        # Normalize kernel
        vals = vals / vals.sum()

        # Create 3-channel separable kernel for conv2d
        # Shape: (out_channels=3, in_channels=1, H, W) for groups=3
        kernel = vals.unsqueeze(0).unsqueeze(0).expand(3, 1, N, N)

        return kernel

    def forward(self, x: Tensor) -> Tensor:
        """Apply random blur.

        Matches StegaStamp: creates kernel, applies via conv2d with SAME padding.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Blurred tensor.
        """
        if self.intensity == 0.0:
            return x

        # For partial intensity, probabilistically skip
        if torch.rand(1).item() > self.intensity:
            return x

        kernel = self._create_blur_kernel(x.device, x.dtype)

        # Apply blur via conv2d with groups=3 (each channel separately)
        # Padding for SAME output size
        pad = self.kernel_size // 2
        x_padded = F.pad(x, [pad, pad, pad, pad], mode="reflect")

        # Conv2d with groups=3 applies each kernel slice to corresponding channel
        blurred = F.conv2d(x_padded, kernel, groups=3)

        return blurred

    def sample_parameters(self) -> dict[str, Any]:
        """Sample which blur type will be used."""
        t = torch.rand(1).item()
        if t < self.prob_gauss:
            blur_type = "gaussian"
        elif t < self.prob_gauss + self.prob_line:
            blur_type = "line"
        else:
            blur_type = "identity"
        return {"blur_type": blur_type}


class GaussianBlur(Distortion):
    """Apply Gaussian blur to images.

    Standalone Gaussian blur (not probabilistic like RandomBlur).
    Sigma range matches StegaStamp: [1.0, 3.0].

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

    def _create_gaussian_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create 2D Gaussian kernel."""
        N = self.kernel_size
        coords = torch.arange(N, device=device, dtype=dtype) - (N - 1) / 2.0

        # Effective sigma scales with intensity
        effective_sigma = self.sigma * self.intensity
        if effective_sigma < 0.1:
            # Return identity kernel
            kernel = torch.zeros(N, N, device=device, dtype=dtype)
            kernel[N // 2, N // 2] = 1.0
        else:
            g = torch.exp(-coords**2 / (2.0 * effective_sigma**2))
            kernel = g.outer(g)
            kernel = kernel / kernel.sum()

        return kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, N, N)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Gaussian blur."""
        if self.intensity == 0.0:
            return x

        kernel = self._create_gaussian_kernel(x.device, x.dtype)
        pad = self.kernel_size // 2
        x_padded = F.pad(x, [pad, pad, pad, pad], mode="reflect")
        return F.conv2d(x_padded, kernel, groups=3)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample sigma in StegaStamp range [1.0, 3.0]."""
        return {"sigma": 1.0 + torch.rand(1).item() * 2.0}


class MotionBlur(Distortion):
    """Apply motion/line blur to images.

    Matches StegaStamp's line blur: directional blur along random angle.

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

    def _create_motion_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create motion blur kernel matching StegaStamp's line blur."""
        N = self.kernel_size

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(N, device=device, dtype=dtype),
                torch.arange(N, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        ) - (0.5 * (N - 1))

        manhat = coords.abs().sum(dim=-1)

        # Use provided angle or random
        theta = self.angle if self.angle is not None else torch.rand(1).item() * 2.0 * math.pi
        v = torch.tensor([math.cos(theta), math.sin(theta)], device=device, dtype=dtype)
        dists = (coords * v).sum(dim=-1)

        effective_sigma = self.sigma * self.intensity
        w_line = 0.5 * (N - 1)

        if effective_sigma < 0.1:
            kernel = torch.zeros(N, N, device=device, dtype=dtype)
            kernel[N // 2, N // 2] = 1.0
        else:
            kernel = torch.exp(-dists**2 / (2.0 * effective_sigma**2)) * (manhat < w_line).float()
            kernel = kernel / kernel.sum()

        return kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, N, N)

    def forward(self, x: Tensor) -> Tensor:
        """Apply motion blur."""
        if self.intensity == 0.0:
            return x

        kernel = self._create_motion_kernel(x.device, x.dtype)
        pad = self.kernel_size // 2
        x_padded = F.pad(x, [pad, pad, pad, pad], mode="reflect")
        return F.conv2d(x_padded, kernel, groups=3)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample angle and sigma in StegaStamp ranges."""
        return {
            "angle": torch.rand(1).item() * 2.0 * math.pi,
            "sigma": 0.25 + torch.rand(1).item() * 0.75,
        }
