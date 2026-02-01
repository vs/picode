# Picode

PyTorch steganography framework for encoding and decoding hidden messages in images. Based on [StegaStamp](https://github.com/tancik/StegaStamp) (Tancik et al., CVPR 2020).

## Features

- **U-Net Encoder**: Embeds binary messages into images as imperceptible perturbations
- **CNN Decoder**: Extracts hidden messages even from distorted images
- **Differentiable Distortions**: Blur, noise, color, geometric, and JPEG compression for training robustness
- **Training Utilities**: Loss functions and trainer class for end-to-end training

## Project Structure

```
picode/
├── model/
│   └── stegastamp/       # Encoder, decoder, loss, and training
├── util/
│   └── distortions/      # Differentiable image distortions
└── docs/                 # Technical documentation
```

## Installation

```bash
# Clone and set up virtual environment
git clone <repo-url> && cd picode
python -m venv venv && source venv/bin/activate

# Install both packages
cd util && pip install -e ".[dev]"
cd ../model && pip install -e ".[dev]"
```

## Quick Start

### Encoding and Decoding

```python
import torch
from stegastamp import Encoder, Decoder

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
from stegastamp import StegaStampTrainer
from distortions import Compose, GaussianNoise, JPEGCompression, PerspectiveWarp

# Create trainer
trainer = StegaStampTrainer(num_bits=100)
optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-4)

# Distortion pipeline for robustness
distortion = Compose([
    GaussianNoise(intensity=0.3),
    JPEGCompression(intensity=0.5),
    PerspectiveWarp(intensity=0.2),
])

# Training loop
for images in dataloader:
    messages = torch.randint(0, 2, (images.size(0), 100)).float()
    losses = trainer.train_step(images, messages, distortion, optimizer)
    print(f"Loss: {losses['loss']:.4f}, Message: {losses['message_loss']:.4f}")
```

### Distortions CLI

```bash
# List available distortions
distort --list

# Apply a distortion to an image
distort input.png output.png gaussian-blur --intensity 0.5
distort input.png output.png perspective --intensity 0.3
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
| Blur | `GaussianBlur`, `MotionBlur` |
| Noise | `GaussianNoise` |
| Color | `BrightnessHue`, `Contrast`, `Saturation` |
| Geometric | `PerspectiveWarp`, `Rotation`, `Scale`, `Crop` |
| Compression | `JPEGCompression` |
| Composite | `Compose` (chain multiple distortions) |

All distortions are differentiable and support an `intensity` parameter (0.0-1.0) for gradual training ramp-up.

## Documentation

- [Distortions Library](util/README.md) - Detailed API for image distortions
- [StegaStamp Model](model/README.md) - Encoder, decoder, and training API
- [Implementation Comparison](docs/distortions_implementation.md) - Technical comparison with original StegaStamp

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.
