# picode-distortions

Differentiable image distortions for training steganography models. All distortions support gradient backpropagation, making them suitable for end-to-end neural network training.

Part of the [picode](../README.md) project.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
import torch
from distortions import GaussianBlur, GaussianNoise, JPEGCompression, Compose

# Single distortion
blur = GaussianBlur(intensity=0.5)
blurred = blur(image_tensor)

# Chain multiple distortions
transform = Compose([
    GaussianNoise(intensity=0.3),
    JPEGCompression(intensity=0.5),
])
output = transform(image_tensor)

# Gradients flow through all distortions
output.sum().backward()  # Works!
```

## Available Distortions

### Blur

```python
from distortions import GaussianBlur, MotionBlur

GaussianBlur(intensity=0.5, kernel_size=7, sigma_range=(1.0, 3.0))
MotionBlur(intensity=0.5, kernel_size=7, angle_range=(0, 360))
```

### Noise

```python
from distortions import GaussianNoise

GaussianNoise(intensity=0.5, std=0.02)
```

### Color

```python
from distortions import BrightnessHue, Contrast, Saturation

BrightnessHue(intensity=0.5, rnd_bri=0.3, rnd_hue=0.1)
Contrast(intensity=0.5, contrast_low=0.5, contrast_high=1.5)
Saturation(intensity=0.5, rnd_sat=1.0)
```

### Geometric

```python
from distortions import PerspectiveWarp, Rotation, Scale, Crop

PerspectiveWarp(intensity=0.5, scale=0.1)
Rotation(intensity=0.5, max_angle=15.0)
Scale(intensity=0.5, scale_range=(0.8, 1.2))
Crop(intensity=0.5, min_crop=0.7)
```

### Compression

```python
from distortions import JPEGCompression

JPEGCompression(intensity=0.5, quality=50)
```

### Composition

```python
from distortions import Compose

pipeline = Compose([
    GaussianNoise(intensity=0.3),
    JPEGCompression(intensity=0.5),
    PerspectiveWarp(intensity=0.2),
])
```

## Intensity Parameter

All distortions accept an `intensity` parameter (0.0 to 1.0) that scales the distortion strength:

- `intensity=0.0`: No distortion (identity transform)
- `intensity=1.0`: Full distortion

This enables gradual ramp-up during training, matching StegaStamp's curriculum learning approach.

## Tensor Conventions

- **Format**: NCHW (batch, channels, height, width)
- **Range**: [0, 1]
- **Output**: Always clamped to [0, 1]

## CLI

```bash
# List available distortions
distort --list

# Apply distortion to an image
distort input.png output.png gaussian-blur --intensity 0.5
distort input.png output.png jpeg --intensity 0.7

# Apply all distortions (creates multiple outputs)
distort input.png output_dir/ all
```

## Creating Custom Distortions

```python
from distortions import Distortion
from torch import Tensor

class MyDistortion(Distortion):
    def __init__(self, intensity: float = 0.5, my_param: float = 1.0):
        super().__init__(intensity)
        self.my_param = my_param

    def forward(self, x: Tensor) -> Tensor:
        if self.intensity == 0.0:
            return x
        # Apply transformation scaled by intensity
        result = x + self.my_param * self.intensity * torch.randn_like(x)
        return torch.clamp(result, 0.0, 1.0)
```
