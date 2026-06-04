# Picode

Steganography framework for encoding and decoding hidden messages in images. Evolved from [StegaStamp](https://github.com/tancik/StegaStamp) (Tancik et al., CVPR 2020) into PicoTrust, a production-quality architecture achieving 32.82 dB PSNR with zero colour shifts and strong JPEG robustness.

## Sub-Projects

This monorepo contains multiple sub-projects:

- **[picode-model/](picode-model/)** - PyTorch training framework for model development
- **[picode-ios/](picode-ios/)** - iOS application for mobile steganography
- **[picode-scraper/](picode-scraper/)** - Distributed web scraper for collecting paired image datasets

## Features

- **PicoTrust Architecture**: 512x512 U-Net encoder with grayscale E_post refinement and strength annealing, delivering 32.82 dB PSNR with zero colour shifts
- **JPEG Robustness**: 98.6% bit accuracy at JPEG quality 10 -- messages survive extreme compression
- **Multiple Model Variants**: StegaStamp, PicodeLite, PicodeFrame, and PicoTrust architectures for different use cases
- **Blind Detection**: Multi-scale sliding window detector for finding steganographic images in photos/videos
- **Differentiable Distortions**: Blur, noise, color, geometric, and JPEG compression with swappable backends (native PyTorch, Kornia)
- **Error Correction Codes**: BCH and LDPC implementations for message robustness
- **Training Infrastructure**: YAML config, curriculum learning, strength annealing, WGAN discriminator, checkpointing, TensorBoard logging
- **Cloud Training**: GCE (Google Cloud) and Modal deployment scripts for GPU training
- **Data Collection**: Distributed web scraper for collecting paired image datasets
- **iOS App**: Mobile implementation for real-world steganography

## Results

| Model | Resolution | PSNR | Bit Accuracy | JPEG Q10 | Colour Shifts |
|-------|-----------|------|-------------|----------|---------------|
| PicoTrust v2 | 512x512 | 32.82 dB | 98.4% | 98.6% | None |
| PicoTrust v1 | 256x256 | 26.54 dB | 99.8% | 99.4% | Yes |
| TrustMark-Q | 256x256 | ~42 dB | ~98% | Fails | None |
| TrustMark-P | 256x256 | ~49 dB | ~98% | Fails | None |

PicoTrust v2 is the recommended production model. It eliminates colour shifts by construction (grayscale 1-channel residual) and maintains strong accuracy under JPEG compression -- an area where TrustMark variants fail.

## Project Structure

```
/
├── docs/                    # Shared documentation
├── picode-ios/              # iOS application
├── picode-model/            # PyTorch training framework
│   ├── picode/              # Python package
│   │   ├── distortions/     # Differentiable image distortions
│   │   ├── ecc/             # Error correction codes (BCH, LDPC)
│   │   ├── models/          # Encoder/decoder models
│   │   │   ├── stegastamp/  #   StegaStamp baseline (CVPR 2020)
│   │   │   ├── picodelite/  #   Lightweight variant
│   │   │   ├── picodeframe/ #   Frame-border encoding variant
│   │   │   └── picotrust/   #   Best model (U-Net + E_post + WGAN)
│   │   ├── detection/       # Blind steganographic image detection
│   │   ├── training/        # Training infrastructure
│   │   └── tests/           # Test suite
│   ├── configs/             # Training configs
│   ├── scripts/             # Utility scripts
│   └── pyproject.toml       # Package config
├── picode-scraper/          # Distributed image dataset scraper
│   ├── picode_scraper/      # Python package
│   ├── configs/             # Scraper configs
│   └── pyproject.toml       # Package config
├── venv/                    # Shared Python virtual environment
├── CLAUDE.md                # Claude Code instructions
├── README.md                # This file
└── LICENSE                  # Apache 2.0
```

## Installation

```bash
# Clone and set up virtual environment
git clone <repo-url> && cd picode
python -m venv venv && source venv/bin/activate

# Install picode-model in development mode
cd picode-model
pip install -e ".[dev]"

# Optional: Install with Kornia backend support
pip install -e ".[kornia]"
```

## Quick Start

### Encoding and Decoding (PicoTrust v2)

```python
import torch
from picode.models.picotrust import Encoder, Decoder

# Initialize models (PicoTrust v2 architecture)
encoder = Encoder(num_bits=100, image_size=512, strength=0.03)
decoder = Decoder(num_bits=100, image_size=256)

# Encode a message into an image
image = torch.rand(1, 3, 512, 512)  # NCHW, [0, 1] range
message = torch.randint(0, 2, (1, 100)).float()  # Binary message

encoded_image = encoder(image, message)

# Decoder works on 256x256 (bilinear downsample from 512)
import torch.nn.functional as F
decoded_input = F.interpolate(encoded_image, size=256, mode='bilinear', align_corners=False)
logits = decoder(decoded_input)  # Returns logits (pre-sigmoid)
binary_message = (torch.sigmoid(logits) > 0.5).float()
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
cd picode-model

# Train PicoTrust v2 (recommended)
picode-train --config configs/picotrust_v2.yaml

# Train StegaStamp baseline
picode-train --config configs/stegastamp_baseline.yaml

# Override specific settings
picode-train --config configs/picotrust_v2.yaml --lr 0.0002 --num-steps 100000
```

### Distortions CLI

```bash
# List available distortions
distort --list

# Apply a distortion to an image
distort gaussian-blur input.png -o output/ --intensity 0.5
distort perspective-warp input.png -o output/ --intensity 0.3

# Apply all distortions sequentially (combine command)
distort combine input.png -o output/ --intensity 0.5
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PicoTrust v2 Training Pipeline                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Image (512) + Message ──► [U-Net Encoder + E_post] ──► Encoded        │
│                              grayscale residual          Image (512)    │
│                              softsign + strength                        │
│                                                                         │
│  Encoded (512) ──► [Distortions] ──► [Downsample 256] ──► [Decoder]   │
│                                                             ──► Msg    │
│                                                                         │
│  Loss = MSE(message) + L2(image) + LPIPS(perceptual)                   │
│        + FFL(frequency) + WGAN(adversarial)                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Encoder** (U-Net + E_post): Takes a 512x512 image and binary message, produces a 1-channel grayscale residual bounded by softsign with strength annealing (1.0 -> 0.03), added to the original image
2. **Distortions**: Simulates real-world degradation (printing, compression, camera capture, lighting)
3. **Decoder** (CNN + compact STN): Extracts the message from a 256x256 (bilinear downsampled) version of the encoded image
4. **Training**: End-to-end optimization with MSE message loss, L2/LPIPS/FFL image losses, and WGAN adversarial loss

## Model Architectures

### PicoTrust (recommended)
The best model. U-Net encoder at 512x512 with E_post refinement layer that outputs a 1-channel grayscale residual. Uses softsign activation with strength annealing (starts at 1.0, anneals to 0.03) to gradually constrain the residual during training. The grayscale residual eliminates colour shifts by construction. Decoder operates at 256x256 with a compact STN (AdaptiveAvgPool2d). Trained with WGAN discriminator, MSE message loss, L2, LPIPS, and FFL losses.

- Best checkpoint: `checkpoints/picotrust_v2/checkpoint_00200000_grayscale.pt`
- Config: `configs/picotrust_v2.yaml`

### StegaStamp
The original baseline architecture from Tancik et al., CVPR 2020. U-Net encoder with BatchNorm and ReLU, CNN decoder with 7 conv layers. Well-understood architecture with reliable gradient flow and training stability.

### PicodeLite
Lightweight variant designed for faster inference and smaller model size. Suitable for resource-constrained deployment.

### PicodeFrame
Frame-border encoding variant that concentrates the steganographic signal in image borders rather than the full image area. Useful when encoding should be confined to a narrow frame region.

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
# Run backend benchmarks (from picode-model/)
cd picode-model
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

Training is configured via YAML files. The recommended config is `picode-model/configs/picotrust_v2.yaml`:

```yaml
experiment_name: picotrust_v2

model:
  type: picotrust
  encoder_size: 512
  decoder_size: 256

data:
  source: folder
  path: ./data/train
  batch_size: 4

training:
  num_steps: 200000
  lr: 0.0001
  num_bits: 100
  image_size: 512
  warmup_steps: 500
  no_im_loss_steps: 10000
  residual_strength: 1.0
  residual_strength_anneal_target: 0.03
  residual_strength_anneal_start: 10000
  residual_strength_anneal_steps: 60000

loss:
  message: { scale: 5.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 50000 }
  lpips: { scale: 1.0, ramp_steps: 50000 }
  ffl: { scale: 1.0, ramp_steps: 50000, delay_steps: 30000 }
  message_loss_type: mse
  gan_config:
    enabled: true
    discriminator_lr: 0.00001
    g_loss_scale: 1.0
    g_loss_ramp_steps: 20000

distortion:
  strategy: curriculum
  perspective: { strength: 0.1, ramp_steps: 10000 }
  noise: { strength: 0.02, ramp_steps: 1000 }
  jpeg_quality: { strength: 25, ramp_steps: 1000 }

checkpoint:
  dir: checkpoints
  save_every_steps: 10000

logging:
  backends: [console, tensorboard]
```

### Distortion Strategies

- **curriculum**: Gradually increase distortion strength during training (recommended)
- **fixed**: Apply distortions at constant strength
- **random**: Randomly sample distortion strength each step
- **none**: No distortions (for baseline comparison)

## Cloud Training

### Google Cloud (GCE)

Train on GCE with T4 GPU:

```bash
cd picode-model

# Setup and start training
./scripts/gce_setup.sh setup
./scripts/gce_setup.sh train

# Resume from checkpoint
./scripts/gce_setup.sh resume
```

### Modal

Train on cloud GPUs using [Modal](https://modal.com/):

```bash
cd picode-model

# Initial setup (one-time)
./scripts/modal_setup.sh setup

# Upload training data (uses tarball for large datasets)
./scripts/modal_setup.sh upload-data ./data/coco/coco2017/train2017

# Start training
./scripts/modal_setup.sh train

# Resume from checkpoint
./scripts/modal_setup.sh resume

# Download checkpoints
./scripts/modal_setup.sh download ./checkpoints_modal
```

## Documentation

- [Distortions Implementation](docs/distortions_implementation.md) - Technical comparison with original StegaStamp
- [Model Implementation](docs/model_implementation.md) - Encoder, decoder, and training details
- [Models Comparison](docs/models_comparison.md) - Comparison of all model architectures
- [Gradient Flow Analysis](docs/gradient_flow_analysis.md) - Analysis of gradient behavior in different architectures
- [Model Improvements](docs/model_improvements.md) - Architecture improvements and optimizations
- [Detection Improvements](docs/detection_improvements.md) - Blind detection system design
- [Mobile Deployment](docs/mobile_deployment.md) - iOS and mobile deployment guide
- [Data Scraper](docs/data_scraper.md) - Distributed image collection pipeline
- [Wisdom](docs/wisdom.md) - Lessons learned from training experiments

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.
