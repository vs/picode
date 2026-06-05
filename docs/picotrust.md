# PicoTrust

PicoTrust is a neural image steganography model that hides binary messages in images with near-invisible modifications. Based on the StegaStamp architecture (Tancik et al., CVPR 2020), PicoTrust introduces grayscale residuals, softsign amplitude bounding with strength annealing, and a compact decoder — achieving high image quality (38-39 dB PSNR) with robust message recovery.

## Architecture

### Overview

```
Image (512x512) + Message (80 bits)
    ↓
  Encoder (U-Net + E_post)
    ↓
  Grayscale residual → softsign bound → strength × r/(1+|r|)
    ↓
  Encoded image = Original + residual
    ↓
  Bilinear downsample to 256x256
    ↓
  Decoder (CNN + compact STN)
    ↓
  80-bit logits → sigmoid → recovered message
```

### Encoder (1.6M params)

StegaStamp-style U-Net with 4 downsampling stages and skip connections, plus a refinement block (E_post):

1. **Message preparation**: 80 bits → Linear(80, 7500) + ReLU → reshape to (3, 50, 50) → upsample to (3, 512, 512)
2. **Input**: Concatenated image + message (6 channels), each normalized by subtracting 0.5
3. **U-Net encoder**: 5 convolutions with stride-2 downsampling, saving activations for skip connections
4. **U-Net decoder**: 4 upsampling stages with skip concatenation, producing 32-channel feature maps
5. **E_post refinement**: 3-layer block replacing StegaStamp's single Conv(32→3):
   ```
   Conv2d(32, 32, 3×3, pad=1) + ReLU
   Conv2d(32, 16, 1×1) + SiLU
   Conv2d(16, 1, 1×1)              ← outputs 1 grayscale channel
   ```
6. **Grayscale broadcast**: 1-channel residual expanded to 3 channels (R=G=B by construction, zero colour shift)
7. **Softsign bounding**: `residual = strength × raw / (1 + |raw|)` — never saturates, always has gradient
8. **Output**: `encoded = image + residual`

**Key design choices:**
- **Grayscale residual** eliminates colour shifts architecturally (no loss function needed)
- **Softsign** instead of tanh: gradient `1/(1+|x|)^2` is small but never zero, preventing gradient death
- **Zero-initialized** E_post final layer: residual starts at exactly zero, grows gradually
- **Strength annealing**: starts at 1.0 (unbounded), anneals to target (0.012-0.015) over training

### Decoder (4.7M params)

StegaStamp-style CNN with a compact Spatial Transformer Network (STN):

1. **Compact STN** (<1M params, vs StegaStamp's 41M):
   ```
   3 stride-2 convs (3→32→64→128) + ReLU
   AdaptiveAvgPool2d(1)        ← replaces massive Flatten+FC
   FC(128, 128) + ReLU
   FC weight (128, 6) → 2×3 affine matrix
   ```
   - Identity-initialized bias: [1, 0, 0, 0, 1, 0]
   - Predicts perspective correction for robustness

2. **Main CNN**:
   ```
   5 stride-2 convs (3→32→64→64→128→128) with stride-1 refinement convs
   Flatten(128 × 8 × 8 = 8192)    ← preserves spatial structure
   FC(8192, 512) + ReLU
   FC(512, 80) → raw logits
   ```
   - No global average pooling (critical: GAP destroys spatial information needed for steganography)

**Total model size**: 6.3M params (vs StegaStamp's ~54M)

### Discriminator

WGAN PatchGAN discriminator with conservative learning rate (1e-5, 0.1× encoder LR). Ramps in over 20k steps.

## Training

### Strength Annealing

The key innovation enabling high-PSNR steganography. The residual amplitude is bounded by `strength × softsign(raw_residual)`, where `strength` anneals during training:

```
Steps 0-10k:     strength = 1.0 (unbounded — encoder-decoder bootstrap)
Steps 10k-90k:   linear anneal from 1.0 → target (e.g., 0.013)
Steps 90k+:      strength = target (fine-tuning at fixed budget)
```

Without annealing, bounded residuals from step 0 prevent encoder-decoder bootstrapping — the encoder can't create patterns strong enough for an untrained decoder to learn from.

### Loss Function

| Loss | Scale | Ramp | Purpose |
|------|-------|------|---------|
| Message (MSE) | 5.0 | 1 step | Bit accuracy — must dominate total image losses |
| L2 | 1.5 | 50k steps | Pixel-level image quality |
| LPIPS | 1.0 | 50k steps | Perceptual image quality |
| FFL | 1.0 | 50k steps (30k delay) | Frequency-domain quality |
| GAN (WGAN) | 1.0 | 20k steps | Adversarial realism |

- **MSE > BCE** for message loss: BCE has a trivial equilibrium at 0.5; MSE has stronger gradients
- **Message scale must dominate**: 5.0 vs ~3.5 total image losses ensures the encoder prioritizes message capacity
- **Slow ramps** prevent instability: image losses ramp over 50k steps, FFL delayed 30k steps

### Distortion Curriculum

During training, encoded images pass through a distortion pipeline before decoding:

| Distortion | Strength | Ramp |
|------------|----------|------|
| Perspective | 0.1 | 10k steps |
| Brightness | ±0.3 | 1k steps |
| Saturation | ±1.0 | 1k steps |
| Hue | ±0.1 | 1k steps |
| Gaussian noise | σ=0.02 | 1k steps |
| JPEG compression | Q25 | 1k steps |
| Gaussian blur | Enabled | 1k steps |

### Training Phases

1. **Bootstrap** (steps 0-10k): Unbounded residual, message loss only. Encoder-decoder establish communication. ~50% bootstrap failure rate — if `prob_std < 0.01` by step 1000, restart.
2. **Squeeze** (steps 10k-90k): Strength anneals from 1.0 to target. Image losses ramp in. Encoder learns to encode efficiently in shrinking budget.
3. **Phase 2** (steps 60k+): Decoder LR drops to 10% of base. Stabilizes decoder while encoder continues adapting.
4. **Fine-tuning** (steps 90k-200k+): Fixed strength, all losses active. Model converges to final quality/accuracy tradeoff.

### De-annealing (v6 Experiment)

A novel technique: after training at one strength, resume at a slightly higher strength to relax the residual budget:

```
v5 (0.012) → v6a (0.013) → v6b (0.014) → v6c (0.015)
```

Each phase runs 30k steps at fixed strength. This maps the PSNR-accuracy curve precisely and avoids training from scratch for each strength level. Relaxing the constraint is easier than tightening — the encoder already knows efficient encoding.

## Results

### Training History

| Version | Bits | Strength | PSNR | Bit Acc | JPEG Q10 | Notes |
|---------|------|----------|------|---------|----------|-------|
| v1 | 100 | unbounded | 26.54 dB | 99.8% | 99.4% | 256x256, colour shifts |
| v2 | 100 | 0.030 | 32.82 dB | 98.4% | 98.6% | 512→256, grayscale residual |
| v3 | 100 | 0.010 | 40.48 dB | 67.6% | 64.0% | Too tight — accuracy collapsed |
| v4 | 100 | 0.020 | 35.56 dB | 97.8% | 97.6% | Good balance for 100 bits |
| v5 | 80 | 0.012 | 39.07 dB | 85.5% | 85.0% | High PSNR, accuracy too low |
| v6a | 80 | 0.013 | 38.84 dB | 91.7% | 90.5% | De-annealed from v5 |
| v6b | 80 | 0.014 | 38.13 dB | 93.6% | 92.1% | De-annealed from v6a |

### Strength-PSNR-Accuracy Relationship

Empirical curve from 80-bit models (50-image evaluation):

| Strength | PSNR (dB) | Raw Accuracy | PSNR delta | Accuracy delta |
|----------|-----------|-------------|------------|----------------|
| 0.012 | 39.07 | 85.5% | — | — |
| 0.013 | 38.84 | 91.7% | -0.23 dB | +6.2% |
| 0.014 | 38.13 | 93.6% | -0.71 dB | +1.9% |
| 0.015 | ~37.5* | ~95-96%* | ~-0.6 dB* | ~+2%* |

*v6c (0.015) estimated, training in progress.

The relationship is nonlinear: the first 0.001 increment (0.012→0.013) gives the best accuracy-per-PSNR tradeoff.

### Robustness (v6b, strength 0.014, 50 images)

| Distortion | Bit Accuracy |
|------------|-------------|
| Clean | 93.6% |
| JPEG Q10 | 92.1% |
| JPEG Q50 | 93.8% |
| JPEG Q90 | 93.7% |
| Gaussian noise σ=0.05 | 93.8% |
| Gaussian noise σ=0.10 | 92.1% |
| Gaussian blur σ=1.0 | 94.5% |
| Gaussian blur σ=3.0 | 93.4% |
| Brightness ±0.1 | 93.4% |
| Brightness ±0.3 | 90.9% |
| Brightness ±0.5 | 88.4% |
| Contrast ±0.3 | 94.1% |
| Contrast ±0.5 | 93.6% |

### Error Correction (LDPC)

Raw bit accuracy of 91-94% translates to ~5-7 errors per 80-bit message. With LDPC soft-decision decoding:

- **LDPC(80, d_v=2, d_c=5)**: 49 payload bits from 80 coded bits (rate 0.613)
- Soft-decision belief propagation uses raw decoder probabilities (not hard-thresholded bits)
- At 94% raw accuracy: LDPC corrects to near-perfect recovery
- At 91% raw accuracy: LDPC achieves ~93% payload accuracy (edge case)

Comparison of ECC approaches (tested on v4, 100-bit model):

| Method | Input | Payload | Approach |
|--------|-------|---------|----------|
| No ECC | 100 raw bits | 100 bits | Hard threshold at 0.5 |
| BCH(63,36) + BCH(31,21) | 94 coded bits | 57 bits | Hard-decision, corrects 5+2 errors |
| LDPC(100, d_v=2, d_c=5) | 100 coded bits | 61 bits | Soft-decision belief propagation |

LDPC with soft decoding is strictly superior: more payload bits (61 vs 57) and better correction from using decoder confidence information rather than discarding it with hard thresholding.

## Comparison with Other Models

### 100-bit Methods

| Model | Year | Bits | Resolution | PSNR (dB) | Clean Acc | JPEG | Params |
|-------|------|------|-----------|-----------|-----------|------|--------|
| HiDDeN | 2018 | 30 | 128x128 | ~38 | 95.5% | — | ~1M |
| StegaStamp | 2020 | 100 | 400x400 | ~30-36* | 99.8% | 99.8% | ~54M |
| TrustMark-Q | 2023 | 100 | 256x256 | 42-45 | ~100% | 89.7% | — |
| TrustMark-B | 2023 | 100 | 256x256 | 40-44 | ~100% | — | — |
| InvisMark | 2024 | 100 | 2K | 51.4 | 100% | 99.5% | ~87M+ |
| WAM | 2024 | 32 | 256x256 | 38.8 | 100% | — | — |
| **PicoTrust v2** | 2026 | 100 | 512x512 | 32.82 | 98.4% | 98.6% (Q10) | **6.3M** |
| **PicoTrust v4** | 2026 | 100 | 512x512 | 35.56 | 97.8% | 97.6% (Q10) | **6.3M** |
| **PicoTrust v6b** | 2026 | 80 | 512x512 | 38.13 | 93.6% | 92.1% (Q10) | **6.3M** |

*StegaStamp PSNR varies 30-37 dB across evaluations; lower numbers reflect aggressive encoding for physical print-and-photograph robustness.

### Analysis

**PicoTrust strengths:**
- **Extreme JPEG robustness**: 92-98% accuracy at JPEG Q10 (most methods test at Q50+). This matters for real-world scenarios where images are heavily recompressed.
- **Compact model**: 6.3M params — 8.5× smaller than StegaStamp, suitable for mobile deployment.
- **No colour shifts**: Grayscale residual guarantees R=G=B perturbation. No hue-based artifacts.
- **Robust across all distortions**: No single failure mode (unlike TrustMark failing JPEG, or StegaStamp failing flips).

**PicoTrust limitations:**
- **PSNR gap**: 38 dB (v6b) vs 42-51 dB for TrustMark/InvisMark. Modern methods using pretrained backbones (ConvNeXT, etc.) achieve higher PSNR with comparable robustness.
- **Bootstrap fragility**: ~50% failure rate per attempt. The model either bootstraps within 1000 steps or collapses permanently.
- **Strength-accuracy cliff**: Below strength ~0.015, accuracy drops sharply. The useful operating range is narrow (0.013-0.03).

**vs StegaStamp (direct ancestor):**
PicoTrust v4 achieves +6 dB PSNR over StegaStamp with comparable accuracy, in an 8.5× smaller model. Key improvements: grayscale residual, softsign bounding, strength annealing, compact STN.

**vs TrustMark:**
TrustMark achieves 42+ dB PSNR but has notably weaker JPEG robustness (89.7%) and struggles with noise (69.9%). PicoTrust prioritizes robustness over PSNR — different design goals. TrustMark is better for pristine digital distribution; PicoTrust is better for real-world scenarios involving compression, social media re-encoding, and physical capture.

**vs InvisMark (SOTA):**
InvisMark (51.4 dB, 99.5% JPEG) represents the current state of the art using a much larger model (ConvNeXT-base decoder, 3-stage training). PicoTrust trades ~13 dB PSNR for 8.5× fewer parameters and a simpler training procedure.

## Checkpoints

Available at [huggingface.co/vadishev/picotrust](https://huggingface.co/vadishev/picotrust):

| Checkpoint | Strength | PSNR | Accuracy | Bits |
|------------|----------|------|----------|------|
| v2/picotrust_v2_200k.pt | 0.030 | 32.82 dB | 98.4% | 100 |
| v4/picotrust_v4_200k.pt | 0.020 | 35.56 dB | 97.8% | 100 |

## Usage

### Encode
```bash
picode -c checkpoints/best.pt encode input.jpg output.png -m "Hello World" \
    --save-original original.png \
    --save-residual residual.png
```

### Decode
```bash
picode -c checkpoints/best.pt decode encoded.png
```

### Evaluate
```bash
python scripts/evaluate.py checkpoints/best.pt --dir data/samples --robustness
```

### ECC Test
```bash
python scripts/test_ecc.py checkpoints/best.pt --dir data/samples --max-images 10
```

## Configuration

Example configuration (v6b — best PSNR/accuracy balance for 80 bits):

```yaml
experiment_name: picotrust_v6b

model:
  type: picotrust
  encoder_size: 512
  decoder_size: 256

training:
  num_bits: 80
  num_steps: 230000
  lr: 0.0001
  residual_strength: 0.014
  residual_strength_anneal_target: 0.014
  phase2_step: 60000
  phase2_decoder_lr_scale: 0.1

loss:
  message: { scale: 5.0 }
  l2: { scale: 1.5, ramp_steps: 50000 }
  lpips: { scale: 1.0, ramp_steps: 50000 }
  ffl: { scale: 1.0, ramp_steps: 50000, delay_steps: 30000 }
  message_loss_type: mse
  gan_config: { enabled: true, discriminator_lr: 0.00001 }

distortion:
  strategy: curriculum
```

See `picode-model/configs/` for all training configurations (v1-v6).

## Training Lessons

Hard-won insights from 6 model versions:

1. **Bounded residuals from step 0 kill bootstrap** — must anneal from unbounded
2. **Grayscale residual eliminates colour shifts** — architectural guarantee, no loss needed
3. **Softsign > tanh** for residual bounding — gradient never reaches zero
4. **Zero-init E_post final layer** — residual starts at zero, grows gradually
5. **MSE > BCE** for message loss — no trivial equilibrium
6. **Message scale must dominate** total image losses (5.0 vs ~3.5)
7. **ResNet50 decoder fails** — AdaptiveAvgPool2d destroys spatial information
8. **GAN LR must be conservative** — 0.1× encoder LR, not 2× (causes collapse)
9. **De-annealing works** — relaxing strength from a converged model is faster than training from scratch
10. **Bootstrap detection** — if `prob_std < 0.01` by step 1000, kill and restart
