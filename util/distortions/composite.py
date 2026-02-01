"""Composite distortions for chaining multiple effects."""

from torch import Tensor, nn

from distortions.base import Distortion


class Compose(nn.Module):
    """Chain multiple distortions together.

    Applies distortions in sequence, passing output of each to the next.

    Args:
        distortions: List of Distortion instances to apply.
    """

    def __init__(self, distortions: list[Distortion]):
        super().__init__()
        self.distortions = nn.ModuleList(distortions)

    def forward(self, x: Tensor) -> Tensor:
        """Apply all distortions in sequence.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Tensor with all distortions applied.
        """
        for distortion in self.distortions:
            x = distortion(x)
        return x

    def __len__(self) -> int:
        """Return number of distortions."""
        return len(self.distortions)

    def __getitem__(self, idx: int) -> Distortion:
        """Get distortion by index."""
        return self.distortions[idx]
