"""Composite distortions for chaining multiple effects.

Note: This is identical to native since it's just sequential application.
"""

from torch import Tensor, nn

from picode.distortions.base import Distortion


class Compose(nn.Module):
    """Chain multiple distortions together.

    API-compatible with native.Compose.

    Args:
        distortions: List of Distortion instances to apply.
    """

    def __init__(self, distortions: list[Distortion]):
        super().__init__()
        self.distortions = nn.ModuleList(distortions)

    def forward(self, x: Tensor) -> Tensor:
        """Apply all distortions in sequence."""
        for distortion in self.distortions:
            x = distortion(x)
        return x

    def __len__(self) -> int:
        """Return number of distortions."""
        return len(self.distortions)

    def __getitem__(self, idx: int) -> Distortion:
        """Get distortion by index."""
        return self.distortions[idx]
