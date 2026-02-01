"""Base class for all distortions."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Distortion(nn.Module, ABC):
    """Abstract base class for differentiable image distortions.

    All distortions operate on tensors of shape (B, C, H, W) with values in [0, 1].

    Args:
        intensity: Strength of the distortion effect, from 0.0 (none) to 1.0 (full).
    """

    name: str

    def __init__(self, intensity: float = 0.5):
        super().__init__()
        self._intensity = intensity

    @property
    def intensity(self) -> float:
        """Get current intensity value."""
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        """Set intensity value."""
        self._intensity = max(0.0, min(1.0, value))

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply distortion to input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Distorted tensor of same shape with values clamped to [0, 1].
        """
        pass

    @abstractmethod
    def sample_parameters(self) -> dict:
        """Randomly sample distortion parameters.

        Called during training to introduce variation. Parameters are sampled
        within ranges scaled by the current intensity.

        Returns:
            Dictionary of parameter names to sampled values.
        """
        pass

    def set_parameters(self, **kwargs) -> None:
        """Set specific parameter values.

        Used for deterministic application (e.g., CLI visualization).

        Args:
            **kwargs: Parameter names and values to set.
        """
        if "intensity" in kwargs:
            self.intensity = kwargs["intensity"]
