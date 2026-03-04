"""Generate sample training images."""

import os
from PIL import Image
import numpy as np

# Create data directory
os.makedirs("data/train", exist_ok=True)

# Generate 20 random color images (400x400)
np.random.seed(42)
for i in range(20):
    # Create random image with some structure (not pure noise)
    img = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
    # Add some gradient to make it more image-like
    gradient = np.linspace(0, 50, 400).reshape(1, 400, 1).astype(np.uint8)
    img = np.clip(img + gradient, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(f"data/train/sample_{i:03d}.png")

print(f"Created 20 sample images in data/train/")
