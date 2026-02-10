# Picode

PyTorch steganography framework for encoding and decoding hidden messages in images. Based on [StegaStamp](https://github.com/tancik/StegaStamp) (Tancik et al., CVPR 2020).

## Features

- **U-Net Encoder**: Embeds binary messages into images as imperceptible perturbations
- **CNN Decoder**: Extracts hidden messages even from distorted images
- **Differentiable Distortions**: Blur, noise, color, geometric, and JPEG compression for training robustness
- **Swappable Backends**: Support for multiple implementations (native, kornia) for benchmarking
- **Error Correction Codes**: BCH and LDPC stubs for message robustness

## Project Structure

```
picode/
├── distortions/           # Differentiable image distortions
│   ├── base.py            # Distortion ABC
│   ├── native/            # Pure PyTorch implementations
│   └── kornia/            # Kornia-based implementations
├── ecc/                   # Error correction codes
│   ├── base.py            # ECC ABC
│   ├── bch/               # BCH implementation (stub)
│   └── ldpc/              # LDPC implementation (stub)
├── models/                # Encoder/decoder models
│   ├── base.py            # Encoder/Decoder ABC
│   └── stegastamp/        # StegaStamp implementation
└── tests/                 # Test suite
```

## Installation

```bash
# Clone and set up virtual environment
git clone <repo-url> && cd picode
python -m venv venv && source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Encoding and Decoding

```python
import torch
from picode.models.stegastamp import Encoder, Decoder

# Initialize models
encoder = Encoder(num_bits=100)
decoder = Decoder(num_bits=100)

# Encode a message into an image
image = torch.rand(1, 3, 400, 400)  # NCHW, [0, 1] range
message = torch.randint(0, 2, (1, 100)).float()  # Binary message

encoded_image = encoder(image, message)
recovered = decoder(encoded_image)
binary_message = (recovered > 0.5).float()
```

### Training with Distortions

```python
from picode.models.stegastamp import StegaStampTrainer
from picode.distortions.native import Compose, GaussianNoise, JPEGCompression, PerspectiveWarp

# Create trainer
trainer = StegaStampTrainer(num_bits=100)
optimizer = torch.optim.Adam(
    list(trainer.encoder.parameters()) + list(trainer.decoder.parameters()),
    lr=1e-4
)

# Distortion pipeline for robustness
distortion = Compose([
    GaussianNoise(intensity=0.3),
    JPEGCompression(intensity=0.5),
    PerspectiveWarp(intensity=0.2),
])

# Training loop
for images in dataloader:
    messages = torch.randint(0, 2, (images.size(0), 100)).float()
    encoded = trainer.encode(images, messages)
    distorted = distortion(encoded)
    decoded = trainer.decode(distorted)
    # ... compute loss and optimize
```

### Distortions CLI

```bash
# List available distortions
distort --list

# Apply a distortion to an image
distort gaussian-blur input.png -o output/ --intensity 0.5
distort perspective-warp input.png -o output/ --intensity 0.3
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Image + Message ──► [Encoder] ──► Encoded ──► [Distortions] ──►        │
│                                    Image                                 │
│                                                                          │
│                      ──► [Decoder] ──► Recovered Message                 │
│                                                                          │
│  Loss = BCE(message) + L2(image) + LPIPS(perceptual)                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Encoder** (U-Net): Takes an image and binary message, outputs an encoded image that looks identical to the original
2. **Distortions**: Simulates real-world degradation (printing, compression, camera capture, lighting)
3. **Decoder** (CNN): Extracts the message from the (possibly distorted) encoded image
4. **Training**: End-to-end optimization balances message recovery accuracy with image quality

## Available Distortions

| Category | Distortions |
|----------|-------------|
| Blur | `GaussianBlur`, `MotionBlur`, `RandomBlur` |
| Noise | `GaussianNoise` |
| Color | `BrightnessHue`, `Contrast`, `Saturation` |
| Geometric | `PerspectiveWarp`, `Rotation`, `Scale`, `Crop` |
| Compression | `JPEGCompression` |
| Composite | `Compose` (chain multiple distortions) |

All distortions are differentiable and support an `intensity` parameter (0.0-1.0) for gradual training ramp-up.

## Using the Kornia Backend

For GPU-optimized performance, install and use the Kornia backend:

```bash
# Install with Kornia support
pip install picode[kornia]
```

```python
# Just change the import - API is identical
from picode.distortions.kornia import GaussianBlur, Compose, JPEGCompression

# Same usage as native backend
blur = GaussianBlur(intensity=0.5, kernel_size=7)
output = blur(image)
```

### Benchmarking Backends

```python
from picode.distortions import native, kornia

# Compare performance
backends = [
    ("native", native.GaussianBlur),
    ("kornia", kornia.GaussianBlur),
]
for name, BlurClass in backends:
    blur = BlurClass(intensity=0.5)
    result = benchmark(blur, test_images)
```

## Documentation

- [Distortions Implementation](docs/distortions_implementation.md) - Technical comparison with original StegaStamp
- [Model Implementation](docs/model_implementation.md) - Encoder, decoder, and training details
- [Project Structure Design](docs/plans/2026-02-03-restructure-design.md) - Architecture decisions

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.
