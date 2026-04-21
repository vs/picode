"""PicodeLite encoder network.

Placeholder for Task 2 - will be implemented with:
- 800x800 encoder resolution
- Learned 2x upsampling for reduced artifacts
- U-Net architecture
"""

# Placeholder - will be implemented in Task 2
# For now, just import to satisfy __init__.py

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """Placeholder for PicodeLite encoder - to be implemented in Task 2."""

    def __init__(self) -> None:
        raise NotImplementedError("PicodeLite encoder not yet implemented - see Task 2")

    def forward(self, image, message):  # type: ignore[no-untyped-def]
        raise NotImplementedError("PicodeLite encoder not yet implemented - see Task 2")
