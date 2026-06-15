"""Print-to-photo distortions using PyTorch native operations.

Simulates the physical pipeline of printing an image and photographing it:
- Resolution loss (downscale + upscale)
- Shot noise (Poisson, brightness-dependent)
- Barrel distortion (lens radial warp)
- Vignetting (edge darkening)
- Chromatic aberration (color channel misalignment)
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from picode.distortions.base import Distortion


class ResolutionLoss(Distortion):
    """Simulate resolution loss from print + camera capture.

    Downscales to a lower resolution then upscales back, destroying
    high-frequency detail — the dominant effect of print-then-photograph.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        scale: Downscale factor. 0.5 = half resolution. Range [0.25, 1.0].
    """

    name = "resolution_loss"

    def __init__(self, intensity: float = 0.5, scale: float = 0.5) -> None:
        super().__init__(intensity)
        self._scale = scale

    def sample_parameters(self) -> dict[str, Any]:
        scale = 1.0 - self._intensity * (1.0 - self._scale)
        actual = max(0.1, scale + torch.empty(1).uniform_(-0.1, 0.1).item() * self._intensity)
        actual = min(1.0, max(0.1, actual))
        return {"scale": actual}

    def forward(self, x: Tensor) -> Tensor:
        if self._intensity == 0.0:
            return x
        params = self.sample_parameters()
        scale = params["scale"]
        if scale >= 0.99:
            return x
        _, _, h, w = x.shape
        small_h, small_w = max(4, int(h * scale)), max(4, int(w * scale))
        small = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


class ShotNoise(Distortion):
    """Brightness-dependent shot noise (Poisson-like).

    Real camera noise is proportional to sqrt(pixel brightness).
    More realistic than uniform Gaussian noise for photo simulation.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        scale: Noise scale factor.
    """

    name = "shot_noise"

    def __init__(self, intensity: float = 0.5, scale: float = 0.05) -> None:
        super().__init__(intensity)
        self._scale = scale

    def sample_parameters(self) -> dict[str, Any]:
        scale = self._intensity * self._scale
        return {"scale": scale}

    def forward(self, x: Tensor) -> Tensor:
        if self._intensity == 0.0 or not self.training:
            return x
        params = self.sample_parameters()
        scale = params["scale"]
        if scale <= 0:
            return x
        # Shot noise: std proportional to sqrt(brightness)
        std = scale * torch.sqrt(x.clamp(min=1e-6))
        noise = torch.randn_like(x) * std
        return (x + noise).clamp(0.0, 1.0)


class BarrelDistortion(Distortion):
    """Radial barrel/pincushion lens distortion.

    Applies radial polynomial warp: r' = r * (1 + k * r^2)
    k > 0 = barrel, k < 0 = pincushion.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        k_range: Range of distortion coefficient.
    """

    name = "barrel_distortion"

    def __init__(
        self, intensity: float = 0.5, k_range: tuple[float, float] = (-0.3, 0.3),
    ) -> None:
        super().__init__(intensity)
        self._k_range = k_range

    def sample_parameters(self) -> dict[str, Any]:
        k = torch.empty(1).uniform_(self._k_range[0], self._k_range[1]).item()
        k *= self._intensity
        return {"k": k}

    def forward(self, x: Tensor) -> Tensor:
        if self._intensity == 0.0:
            return x
        params = self.sample_parameters()
        k = params["k"]
        if abs(k) < 1e-6:
            return x

        b, c, h, w = x.shape
        # Create normalized grid [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        r2 = grid_x ** 2 + grid_y ** 2
        factor = 1.0 + k * r2
        grid_x_d = grid_x * factor
        grid_y_d = grid_y * factor
        grid = torch.stack([grid_x_d, grid_y_d], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class Vignetting(Distortion):
    """Radial brightness falloff at image edges.

    Simulates lens vignetting: output = input * (1 - strength * r^2)

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        strength: Maximum darkening at corners.
    """

    name = "vignetting"

    def __init__(self, intensity: float = 0.5, strength: float = 0.3) -> None:
        super().__init__(intensity)
        self._strength = strength

    def sample_parameters(self) -> dict[str, Any]:
        s = self._intensity * self._strength
        s *= (0.7 + 0.6 * torch.rand(1).item())  # random variation
        return {"strength": s}

    def forward(self, x: Tensor) -> Tensor:
        if self._intensity == 0.0:
            return x
        params = self.sample_parameters()
        s = params["strength"]
        if s <= 0:
            return x

        _, _, h, w = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        r2 = grid_x ** 2 + grid_y ** 2
        # Normalize so corners (r2=2) get full effect
        mask = (1.0 - s * r2 / 2.0).clamp(min=0.0)
        return (x * mask.unsqueeze(0).unsqueeze(0)).clamp(0.0, 1.0)


class ChromaticAberration(Distortion):
    """Color channel spatial misalignment.

    Shifts R, G, B channels by slightly different amounts,
    simulating lens chromatic aberration or print registration error.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        max_shift: Maximum pixel shift as fraction of image size.
    """

    name = "chromatic_aberration"

    def __init__(self, intensity: float = 0.5, max_shift: float = 0.005) -> None:
        super().__init__(intensity)
        self._max_shift = max_shift

    def sample_parameters(self) -> dict[str, Any]:
        shift = self._intensity * self._max_shift
        # R and B shift in opposite directions, G stays centered
        angle = torch.rand(1).item() * 2 * math.pi
        r_dx = shift * math.cos(angle)
        r_dy = shift * math.sin(angle)
        b_dx = -r_dx
        b_dy = -r_dy
        return {"r_shift": (r_dx, r_dy), "b_shift": (b_dx, b_dy)}

    def forward(self, x: Tensor) -> Tensor:
        if self._intensity == 0.0:
            return x
        params = self.sample_parameters()
        r_dx, r_dy = params["r_shift"]
        b_dx, b_dy = params["b_shift"]
        if abs(r_dx) < 1e-6 and abs(r_dy) < 1e-6:
            return x

        b, c, h, w = x.shape
        # Base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

        # Shifted grids for R and B channels
        r_grid = base_grid.clone()
        r_grid[..., 0] += r_dx * 2  # *2 because grid is [-1, 1]
        r_grid[..., 1] += r_dy * 2

        b_grid = base_grid.clone()
        b_grid[..., 0] += b_dx * 2
        b_grid[..., 1] += b_dy * 2

        r_ch = F.grid_sample(
            x[:, 0:1], r_grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        g_ch = x[:, 1:2]  # Green stays centered
        b_ch = F.grid_sample(
            x[:, 2:3], b_grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return torch.cat([r_ch, g_ch, b_ch], dim=1).clamp(0.0, 1.0)
