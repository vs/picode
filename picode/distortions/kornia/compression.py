"""Compression distortions using Kornia.

Note: This is currently a stub that re-exports the native implementation.
A Kornia-based JPEG compression implementation could use kornia.contrib.image_codec
when available, but for now we use the native PyTorch implementation.
"""

# Re-export from native as Kornia doesn't have a direct equivalent
from picode.distortions.native.compression import JPEGCompression

__all__ = ["JPEGCompression"]
