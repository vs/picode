"""Harvester utility functions."""

import imagehash
import numpy as np
from PIL import Image


def compute_phash(image: np.ndarray) -> bytes:
    """Compute perceptual hash for an image.

    Args:
        image: OpenCV image (BGR format, HWC shape)

    Returns:
        8-byte perceptual hash
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = image[:, :, ::-1]
    else:
        rgb = image

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb)

    # Compute perceptual hash (64 bits = 8 bytes)
    phash = imagehash.phash(pil_image)

    # Pack 64 boolean bits into 8 bytes
    packed = np.packbits(phash.hash.flatten())
    return packed.tobytes()
