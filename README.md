# Picode

Neural image steganography framework for encoding hidden messages in photographs. Built on [StegaStamp](https://github.com/tancik/StegaStamp) (Tancik et al., CVPR 2020), evolved into **PicoTrust** — a content-adaptive architecture that hides information in image textures while leaving smooth regions untouched.

## Sub-Projects

- **[picode-model/](picode-model/)** - PyTorch training framework for model development
- **[picode-ios/](picode-ios/)** - iOS application for mobile steganography
- **[picode-scraper/](picode-scraper/)** - Distributed web scraper for collecting training image datasets

## Results

### PicoTrust v14 (best content-adaptivity)

64 bits, 512→512 decoder, `blur(encoded)` σ=1.0, exponential strength annealing.

| Variant | PSNR | Bit Accuracy | JPEG Q10 | TRC |
|---------|------|-------------|----------|-----|
| v14 (strength 0.025) | 33.85 dB | 98.4% | 97.7% | 0.591 |
| v14 s020 (post-annealed) | 35.15 dB | 98.2% | 96.8% | 0.572 |

With LDPC soft decoding (33 payload bits from 64 coded): 99.8% clean, 99.8% JPEG Q10, 99.8% blur σ=6.

### PicoTrust v12 (production model)

64 bits, 512→416 decoder, decoder-side blur σ=0.8, blurred LPIPS/GAN.

| Variant | PSNR | Bit Accuracy | JPEG Q10 |
|---------|------|-------------|----------|
| v12 (strength 0.014) | 38.11 dB | 95.5% | 93.2% |
| v12 s010 (de-annealed) | 40.72 dB | 94.0% | 89.8% |

### PicoTier v1 (multi-tier)

4-tier model based on v14 techniques. One model, multiple capacity/quality tradeoffs:

| Tier | Bits | Payload (LDPC) | Strength | Decoder Blur σ |
|------|------|---------------|----------|----------------|
| UHQ | 30 | 17 data bits | 0.010 | 1.4 |
| HQ | 48 | 26 data bits | 0.011 | 1.0 |
| MQ | 72 | 38 data bits | 0.012 | 0.8 |
| LQ | 96 | 50 data bits | 0.013 | 0.6 |

## Key Innovations

### Content-adaptive encoding (v12/v14)

The encoder learns to concentrate residual energy in textured image regions and avoid smooth areas. This makes modifications perceptually invisible without post-processing masks.

**How it works:**
1. **Decoder-side blur**: Gaussian blur on decoder input during training forces the decoder to learn low-frequency patterns
2. **Blurred LPIPS/GAN**: Perceptual losses computed on `blur(encoded)` enable differential penalization — more in smooth regions, less in textured regions
3. **Clean output**: The encoder's actual output is never blurred — sharp, clean images at inference

### Exponential strength annealing

Residual strength anneals from 1.0 to target (e.g., 0.025) using exponential schedule: `strength = initial × (target/initial)^progress`. Spends equal training time per order of magnitude, unlike linear annealing which rushes through the critical low-strength regime.

### v14 vs v12 architecture

| | v12 | v14 |
|---|-----|-----|
| Decoder resolution | 416 | 512 |
| Decoder blur σ | 0.8 | 1.0 |
| Blur target | decoder input only | `blur(encoded)` for decoder+LPIPS+GAN |
| Target strength | 0.014 | 0.025 |
| Content-adaptivity | moderate | strong (TRC=0.591) |
| Best PSNR | 40.72 dB (s010) | 35.15 dB (s020) |
| Use case | max PSNR | max adaptivity |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PicoTrust v14 Training Pipeline                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Image (512) + Message (64b) → [U-Net Encoder + E_post]            │
│                                  grayscale 1ch residual             │
│                                  softsign × strength                │
│                                          ↓                          │
│                              Encoded Image (512)                    │
│                                          ↓                          │
│                              [Distortions]                          │
│                                          ↓                          │
│                              blur(encoded) σ=1.0                    │
│                                          ↓                          │
│                              [Decoder (512)] → Message              │
│                                                                     │
│  Losses:                                                            │
│    L2, FFL     on real encoded image (pixel quality)                │
│    LPIPS, GAN  on blur(encoded) (content-adaptive gradients)        │
│    MSE         on message (bit recovery)                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Encoding and Decoding

```bash
cd picode-model

# Encode a message into an image
picode -c checkpoints/best.pt encode input.jpg output.png -m "Hello World" \
    --save-original original.png \
    --save-residual residual.png

# Decode a message from an encoded image
picode -c checkpoints/best.pt decode encoded.png
```

### Training

```bash
cd picode-model
pip install -e ".[dev]"

# Train PicoTrust v14 (best content-adaptivity)
picode-train --config configs/picotrust_v14.yaml

# Train PicoTrust v12 (production, high PSNR)
picode-train --config configs/picotrust_v12.yaml

# Train PicoTier v1 (multi-tier)
picode-train --config configs/picotier_v1.yaml
```

### Error Correction

```python
from picode.ecc.ldpc import LDPC

# LDPC soft-decision decoding with decoder logit probabilities
ldpc = LDPC(n=64, d_v=2, d_c=4)
codeword = ldpc.encode(message)
decoded = ldpc.decode(soft_probabilities)  # Uses sigmoid(logits), not hard bits
```

## Documentation

- [PicoTrust Architecture & Results](docs/picotrust.md) — Complete architecture, all version results, training lessons 1-34
- [Training Best Practices](docs/training_best_practices.md) — Distilled recipe for production models
- [Training Wisdom](docs/wisdom.md) — Hard-learned lessons from training experiments

## Installation

```bash
git clone <repo-url> && cd picode
python -m venv venv && source venv/bin/activate
cd picode-model && pip install -e ".[dev]"
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) file.
