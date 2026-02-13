# Picode

PyTorch steganography framework for encoding and decoding hidden messages in images. Based on [StegaStamp](https://github.com/tancik/StegaStamp) (Tancik et al., CVPR 2020).

## Features

- **U-Net Encoder**: Embeds binary messages into images as imperceptible perturbations
- **CNN Decoder**: Extracts hidden messages even from distorted images
- **Differentiable Distortions**: Blur, noise, color, geometric, and JPEG compression with swappable backends (native PyTorch, Kornia)
- **Error Correction Codes**: BCH and LDPC implementations for message robustness
- **Training Infrastructure**: YAML config, curriculum learning, checkpointing, TensorBoard logging

## Project Structure

```
picode/
├── distortions/           # Differentiable image distortions
│   ├── base.py            # Distortion ABC
│   ├── native/            # Pure PyTorch implementations
│   └── kornia/            # Kornia-based implementations
├── ecc/                   # Error correction codes
│   ├── base.py            # ECC ABC
│   ├── bch/               # BCH implementation (galois library)
│   └── ldpc/              # LDPC implementation (pyldpc library)
├── models/                # Encoder/decoder models
│   ├── base.py            # Encoder/Decoder ABC
│   └── stegastamp/        # StegaStamp implementation
├── training/              # Training infrastructure
│   ├── config.py          # YAML config loading
│   ├── trainer.py         # Trainer with loss ramping
│   ├── evaluation.py      # Evaluator with robustness sweeps
│   ├── checkpointing.py   # Checkpoint management
│   └── logging/           # Console and TensorBoard loggers
└── tests/                 # Test suite
```

## Installation

```bash
# Clone and set up virtual environment
git clone <repo-url> && cd picode
python -m venv venv && source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Optional: Install with Kornia backend support
pip install -e ".[kornia]"
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
from picode.models.stegastamp import Encoder, Decoder, train_step
from picode.distortions.native import Compose, GaussianNoise, JPEGCompression, PerspectiveWarp

# Initialize models
encoder = Encoder(num_bits=100)
decoder = Decoder(num_bits=100)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
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
    losses = train_step(
        encoder=encoder,
        decoder=decoder,
        images=images,
        distortion=distortion,
        optimizer=optimizer,
    )
    print(f"Loss: {losses['loss']:.4f}, Accuracy: {losses['accuracy']:.2%}")
```

### Using Error Correction

```python
from picode.ecc import BCH
from picode.ecc.ldpc import LDPC

# BCH: corrects up to 10 bit errors in 127-bit codewords
bch = BCH(n=127, k=64)
message = torch.randint(0, 2, (4, 64)).float()
codeword = bch.encode(message)
decoded, success = bch.decode(codeword)

# LDPC: soft-decision belief propagation decoding
ldpc = LDPC(n=200, d_v=3, d_c=6)
message = torch.randint(0, 2, (4, ldpc.message_length)).float()
codeword = ldpc.encode(message)
decoded = ldpc.decode(codeword.float())
```

### Training with Config File

```bash
# Train with default config
picode-train --config configs/stegastamp_baseline.yaml

# Override specific settings
picode-train --config configs/stegastamp_baseline.yaml --lr 0.0002 --num-steps 50000
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

### Backend Differences

The native and Kornia backends have identical APIs but may produce slightly different results:

- **Saturation**: Native uses StegaStamp's RGB luminance weights (0.3, 0.6, 0.1); Kornia uses standard Rec.601 weights
- **JPEG**: Kornia uses `jpeg_codec_differentiable` for more accurate compression simulation

### Benchmarking Backends

```bash
# Run backend benchmarks
pytest picode/tests/benchmarks/ -v
```

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

## Training Configuration

Training is configured via YAML files (see `configs/stegastamp_baseline.yaml`):

```yaml
experiment_name: my_experiment

data:
  path: ./data/train
  batch_size: 4
  num_workers: 4

training:
  num_steps: 140000
  lr: 0.0001
  num_bits: 100
  image_size: 400

loss:
  message: { scale: 1.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 20000 }
  lpips: { scale: 1.0, ramp_steps: 20000 }

distortion:
  strategy: curriculum  # curriculum, fixed, random, none
  perspective: { strength: 0.1, ramp_steps: 10000 }
  noise: { strength: 0.02, ramp_steps: 1000 }
  jpeg_quality: { strength: 25, ramp_steps: 1000 }

checkpoint:
  dir: checkpoints
  save_every_steps: 10000

logging:
  backends: [console, tensorboard]
  tensorboard_dir: runs
```

### Distortion Strategies

- **curriculum**: Gradually increase distortion strength during training (recommended)
- **fixed**: Apply distortions at constant strength
- **random**: Randomly sample distortion strength each step
- **none**: No distortions (for baseline comparison)

## Documentation

- [Distortions Implementation](docs/distortions_implementation.md) - Technical comparison with original StegaStamp
- [Model Implementation](docs/model_implementation.md) - Encoder, decoder, and training details
- [Project Structure Design](docs/plans/2026-02-03-restructure-design.md) - Architecture decisions

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.
