# PicoTrust

PicoTrust is a neural image steganography model that hides binary messages in images with near-invisible modifications. Based on the StegaStamp architecture (Tancik et al., CVPR 2020), PicoTrust introduces grayscale residuals, softsign amplitude bounding with strength annealing, and a compact decoder — achieving high image quality (39+ dB PSNR) with robust message recovery.

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
| **v6c** | **80** | **0.015** | **37.71 dB** | **94.6%** | **92.4%** | **De-annealed from v6b — best 80-bit balance** |
| v7 | 128 | 0.015 | 38.17 dB | 93.8% | 88.6% | 512→512 decoder, 19.2M params |
| v8 | 96 | 0.014 | 37.93 dB | 78.3%* | 76.0%* | 512→416, LPIPS 1.5, GAN 1.5, stopped early |
| **v9** | **32** | **0.014** | **38.56 dB** | **99.2%** | **98.4%** | **512→416, LPIPS 1.5, GAN 1.5 — best accuracy** |
| v9 s013 | 32 | 0.013 | 39.29 dB | 98.9% | 97.7% | De-annealed from v9 |
| v9 s012 | 32 | 0.012 | 39.75 dB | 99.1% | 97.6% | De-annealed from s013 |
| **v9 s011** | **32** | **0.011** | **40.28 dB** | **98.8%** | **97.4%** | **De-annealed from s012 — 40 dB milestone** |

*v8 evaluated at 100k steps (only 10k past annealing). Accuracy was still recovering.

### Decoder Resolution Experiment (v7)

v7 tested whether a full-resolution 512→512 decoder (no downsampling) could encode more bits at the same PSNR. Key findings:

**Results (128 bits, 512→512, strength 0.015, 150k steps):**
- PSNR: 38.17 dB — **higher** than v6c (37.71 dB) despite 60% more bits
- Raw accuracy: 93.8% — slightly lower than v6c (94.6%)
- LDPC(128, d_v=2, d_c=4): **65 payload bits** at 99.9% clean correction (+33% vs v6c's 49 bits)
- Model size: 19.2M params (3× larger due to FC(32768→512) layer)

**Perceptual quality issue:** Despite higher PSNR, v7's encoded images had more noticeable artifacts. The 512→512 decoder removes the low-pass filtering effect of downsampling, allowing the encoder to use high-frequency spatial patterns. These have low per-pixel amplitude (high PSNR) but are perceptually conspicuous.

**Lesson:** The 512→256 downsampling in earlier versions wasn't just a bottleneck — it was an implicit perceptual quality constraint. It forced the encoder to use low-frequency, smooth residual patterns that are less visible to the human eye. PSNR doesn't capture this; LPIPS is a better metric for encoding invisibility.

**Robustness comparison (50 images):**

| Distortion | v6c (80b, 512→256) | v7 (128b, 512→512) |
|------------|--------------------|--------------------|
| Clean | 94.6% | 93.8% |
| JPEG Q10 | 92.4% | 88.6% |
| Blur σ=3 | 94.4% | 91.5% |
| Noise σ=0.1 | 93.2% | 90.3% |
| Brightness ±0.5 | 88.8% | 89.2% |

v7 is weaker on JPEG and blur (distortions that destroy high-frequency patterns) but comparable on brightness (a low-frequency distortion). This confirms the encoder is relying on high-frequency patterns that don't survive common distortions.

### Content-Adaptive Encoding Experiment (v8)

v8 tested 96 bits with a 512→416 decoder (mild 1.23× downsampling) and stronger perceptual losses (LPIPS 1.5, GAN g_loss 1.5, disc_lr 1.5e-5) at strength 0.014.

**Key finding: content-adaptive residual placement.** The increased LPIPS and GAN weights taught the encoder to concentrate residuals in textured/complex regions where changes are perceptually invisible, and avoid smooth areas (sky, walls) where artifacts are conspicuous. This is the first PicoTrust version to show strongly content-adaptive encoding.

**Why it works:**
- **LPIPS (1.5×)** uses VGG features that are more sensitive to changes in smooth/perceptually important regions. Hiding in texture costs less LPIPS penalty, so the encoder learns to exploit textured areas.
- **GAN (1.5×)** reinforces this — the discriminator easily spots artifacts in uniform regions but struggles to detect changes in texture.
- **416 decoder** provides enough spatial resolution for locally selective encoding, unlike the 256 decoder which forces coarse low-frequency patterns.

**Accuracy issue:** 96 bits at strength 0.014 pushed the capacity limit too hard. After annealing completed at step 90k, accuracy dropped to 64-71% and recovered slowly to ~77% by step 101k. The model learned excellent spatial strategy but couldn't encode enough information in the tight budget. Stopped early — the content-adaptive behavior is the valuable finding, not the accuracy.

**Implications for future training:** The LPIPS 1.5 + GAN 1.5 loss combination should be applied to models with proven capacity (e.g., 80 bits at strength 0.015) to get content-adaptive encoding without sacrificing accuracy.

### Minimal Bits Experiment (v9)

v9 tested the capacity floor: 32 bits with the v8 loss design (LPIPS 1.5, GAN 1.5) at 512→416 decoder, strength 0.014. Then de-annealed to 0.013.

**Results (32 bits, 512→416, 50-image evaluation):**

| Strength | Steps | PSNR | Accuracy | JPEG Q10 | Blur σ=3 | Bright ±0.5 |
|----------|-------|------|----------|----------|----------|-------------|
| 0.014 | 160k | 38.56 dB | 99.2% | 98.4% | 99.1% | 96.1% |
| 0.013 | 180k | 39.29 dB | 98.9% | 97.7% | 98.7% | 96.1% |
| **0.012** | **200k** | **39.75 dB** | **99.1%** | **97.6%** | **98.5%** | **94.6%** |

**Key findings:**
- 32 bits at 99.2% accuracy = ~0.25 bit errors on average — essentially perfect without ECC
- Bootstrapped instantly (99% at step 2100 vs 50% failure rate at higher bit counts)
- De-annealing from 0.014→0.013 traded only -0.3% accuracy for +0.73 dB PSNR
- Content-adaptive encoding visible: residuals concentrate in textured regions
- **Perceptual artifacts still visible** despite high PSNR — the 416 decoder allows higher-frequency patterns than 256

**Capacity analysis:** 32 bits is well within the encoder's capacity even at strength 0.012. The model handles the tight residual budget with ease, suggesting the capacity limit at this strength is somewhere between 64-96 bits.

**De-annealing to 0.011:** Each 0.001 strength reduction costs only ~0.3% accuracy while gaining ~0.5 dB PSNR. At 0.011, v9 achieves **40.28 dB** — surpassing v3's PSNR (40.48 dB) with 98.8% accuracy vs v3's 67.6%. The difference: v3 had 100 bits, v9 has 32. This proves the capacity theory — fewer bits at the same strength yields dramatically better results.

**Trade-off: bits vs perceptual quality.** With 32 bits the encoder has excess capacity, so the real constraint isn't accuracy but artifact visibility. High-frequency patterns remain visible with the 416 decoder despite 39+ dB PSNR. Techniques to suppress them: residual blurring, TV loss, lower-res residual generation, or 512→256 decoder.

### Strength-PSNR-Accuracy Relationship

Empirical curves from 50-image evaluations:

**80-bit models (512→256 decoder):**

| Strength | PSNR (dB) | Raw Accuracy | PSNR delta | Accuracy delta |
|----------|-----------|-------------|------------|----------------|
| 0.012 | 39.07 | 85.5% | — | — |
| 0.013 | 38.84 | 91.7% | -0.23 dB | +6.2% |
| 0.014 | 38.13 | 93.6% | -0.71 dB | +1.9% |
| 0.015 | 37.71 | 94.6% | -0.42 dB | +1.0% |

The first 0.001 increment (0.012→0.013) gives the best accuracy-per-PSNR tradeoff. Diminishing returns above 0.014.

**32-bit models (512→416 decoder, LPIPS 1.5, GAN 1.5):**

| Strength | PSNR (dB) | Raw Accuracy | PSNR delta | Accuracy delta |
|----------|-----------|-------------|------------|----------------|
| 0.014 | 38.56 | 99.2% | — | — |
| 0.013 | 39.29 | 98.9% | +0.73 dB | -0.3% |
| 0.012 | 39.75 | 99.1% | +0.46 dB | +0.2% |
| 0.011 | 40.28 | 98.8% | +0.53 dB | -0.3% |

With 32 bits, de-annealing barely costs accuracy. Each 0.001 step gains ~0.5 dB PSNR for ~0.3% accuracy. At 0.011, the model crosses 40 dB — matching TrustMark-B territory while maintaining 98.8% accuracy and 97.4% JPEG Q10 robustness.

In linear terms (RMS residual amplitude on 0-255 scale):
- 0.013 strength → 39.3 dB → ~2.8 pixel levels modified per pixel
- 0.015 strength → 37.7 dB → ~3.3 pixel levels modified per pixel

### Robustness (v9 s012, 32 bits, strength 0.012, 50 images — best model)

| Distortion | Bit Accuracy |
|------------|-------------|
| Clean | 99.1% |
| JPEG Q10 | 97.6% |
| JPEG Q50 | 98.9% |
| JPEG Q90 | 99.1% |
| Gaussian noise σ=0.05 | 99.1% |
| Gaussian noise σ=0.10 | 98.7% |
| Gaussian blur σ=1.0 | 99.1% |
| Gaussian blur σ=3.0 | 98.5% |
| Brightness ±0.1 | 98.9% |
| Brightness ±0.3 | 96.6% |
| Brightness ±0.5 | 94.6% |
| Contrast ±0.3 | 99.3% |
| Contrast ±0.5 | 99.1% |

### Robustness (v6c, 80 bits, strength 0.015, 50 images)

| Distortion | Bit Accuracy |
|------------|-------------|
| Clean | 94.6% |
| JPEG Q10 | 92.4% |
| JPEG Q50 | 94.5% |
| JPEG Q90 | 94.7% |
| Gaussian noise σ=0.05 | 94.7% |
| Gaussian noise σ=0.10 | 93.2% |
| Gaussian blur σ=1.0 | 95.1% |
| Gaussian blur σ=3.0 | 94.4% |
| Brightness ±0.1 | 93.8% |
| Brightness ±0.3 | 91.9% |
| Brightness ±0.5 | 88.8% |
| Contrast ±0.3 | 94.2% |
| Contrast ±0.5 | 93.7% |

### Error Correction (LDPC)

Raw bit accuracy of 94.6% translates to ~4-5 errors per 80-bit message. With LDPC soft-decision decoding:

- **LDPC(80, d_v=2, d_c=5)**: 49 payload bits from 80 coded bits (rate 0.613)
- Soft-decision belief propagation uses raw decoder probabilities (not hard-thresholded bits)
- At 94.6% raw accuracy (v6c): LDPC corrects to **99.8% payload accuracy** on clean images

#### ECC Results on v6c (strength 0.015, 5 images × 10 trials)

| Distortion | Raw (80b) | BCH hard (36b) | LDPC soft (49b) |
|------------|-----------|-----------------|-----------------|
| Clean | 96.1% | 99.6% | **99.8%** |
| JPEG Q5 | 94.9% | 99.6% | **99.8%** |
| JPEG Q10 | 95.6% | 99.7% | **99.6%** |
| Blur σ=2 | 96.4% | 100% | **100%** |
| Blur σ=6 | 96.2% | 99.9% | **99.9%** |
| Noise σ=0.05 | 95.5% | 99.3% | **99.7%** |
| Noise σ=0.10 | 93.8% | 98.8% | **99.1%** |
| Noise σ=0.15 | 91.1% | 93.6% | **97.5%** |
| Noise σ=0.20 | 87.0% | 88.1% | 92.4% |
| Brightness ±0.3 | 92.7% | 96.0% | **96.6%** |
| Brightness ±0.5 | 90.0% | 93.0% | 92.8% |
| Brightness ±0.7 | 88.0% | 87.9% | 90.0% |

LDPC achieves near-perfect correction (99.6-100%) on all common distortions (JPEG, blur, moderate noise). Only extreme stress tests (noise σ>0.15, brightness ±0.5+) remain below 95%.

#### ECC Comparison

| Method | Input | Payload | Approach |
|--------|-------|---------|----------|
| No ECC | 80 raw bits | 80 bits | Hard threshold at 0.5 |
| BCH(63,36) | 63 coded bits + 17 padding | 36 bits | Hard-decision, corrects 5 errors |
| LDPC(80, d_v=2, d_c=5) | 80 coded bits | 49 bits | Soft-decision belief propagation |

LDPC with soft decoding is strictly superior: more payload bits (49 vs 36) and better correction from using decoder confidence information rather than discarding it with hard thresholding.

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
| **PicoTrust v6c** | 2026 | 80 | 512x512 | 37.71 | 94.6% | 92.4% (Q10) | **6.3M** |
| **PicoTrust v7** | 2026 | 128 | 512x512 | 38.17 | 93.8% | 88.6% (Q10) | **19.2M** |
| PicoTrust v8 | 2026 | 96 | 512→416 | 37.93 | 78.3%* | 76.0%* (Q10) | 13.3M |
| **PicoTrust v9** | 2026 | 32 | 512→416 | 40.28 | 98.8% | 97.4% (Q10) | 12.8M |

*v8 stopped early (100k steps). Content-adaptive encoding but 96 bits exceeded capacity at strength 0.014.

*StegaStamp PSNR varies 30-37 dB across evaluations; lower numbers reflect aggressive encoding for physical print-and-photograph robustness.

### Analysis

**PicoTrust strengths:**
- **Extreme JPEG robustness**: 92-98% accuracy at JPEG Q10 (most methods test at Q50+). This matters for real-world scenarios where images are heavily recompressed.
- **Compact model**: 6.3M params — 8.5× smaller than StegaStamp, suitable for mobile deployment.
- **No colour shifts**: Grayscale residual guarantees R=G=B perturbation. No hue-based artifacts.
- **Robust across all distortions**: No single failure mode (unlike TrustMark failing JPEG, or StegaStamp failing flips).

**PicoTrust limitations:**
- **PSNR gap**: 40.3 dB (v9) vs 42-51 dB for TrustMark/InvisMark. Gap is closing — within ~2 dB of TrustMark-Q. Modern methods use pretrained backbones (ConvNeXT, etc.) for higher PSNR.
- **Bootstrap fragility**: ~50% failure rate per attempt. The model either bootstraps within 1000 steps or collapses permanently.
- **Strength-accuracy cliff**: Depends on bit count. For 80 bits, below ~0.013 accuracy drops below LDPC threshold. For 32 bits, strength 0.013 gives 98.9% accuracy — the cliff is much lower.

**vs StegaStamp (direct ancestor):**
PicoTrust v4 achieves +6 dB PSNR over StegaStamp with comparable accuracy, in an 8.5× smaller model. Key improvements: grayscale residual, softsign bounding, strength annealing, compact STN.

**vs TrustMark:**
TrustMark achieves 42+ dB PSNR but has notably weaker JPEG robustness (89.7%) and struggles with noise (69.9%). PicoTrust prioritizes robustness over PSNR — different design goals. TrustMark is better for pristine digital distribution; PicoTrust is better for real-world scenarios involving compression, social media re-encoding, and physical capture.

**vs InvisMark (SOTA):**
InvisMark (51.4 dB, 99.5% JPEG) represents the current state of the art using a much larger model (ConvNeXT-base decoder, 3-stage training). PicoTrust trades ~14 dB PSNR for 8.5× fewer parameters and a simpler training procedure.

**Physical robustness (print-to-scan):**
Neither TrustMark nor InvisMark tests physical robustness — they only evaluate digital distortions. StegaStamp remains the only model designed for print-to-photograph survival. PicoTrust inherits StegaStamp's STN for geometric correction and trains with perspective/JPEG/noise distortions. At v4's strength (0.02, 35.6 dB), the residual signal (~4.2 pixel levels RMS) is comparable to StegaStamp's and could plausibly survive physical capture. Higher-PSNR variants (v6a-c at 37-39 dB) trade physical robustness for digital invisibility.

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

Example configuration (v6c — best PSNR/accuracy balance for 80 bits):

```yaml
experiment_name: picotrust_v6c

model:
  type: picotrust
  encoder_size: 512
  decoder_size: 256

training:
  num_bits: 80
  num_steps: 260000
  lr: 0.0001
  residual_strength: 0.015
  residual_strength_anneal_target: 0.015
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

See `picode-model/configs/` for all training configurations (v1-v9).

## Training Lessons

Hard-won insights from 9 model versions:

1. **Bounded residuals from step 0 kill bootstrap** — must anneal from unbounded
2. **Grayscale residual eliminates colour shifts** — architectural guarantee, no loss needed
3. **Softsign > tanh** for residual bounding — gradient never reaches zero
4. **Zero-init E_post final layer** — residual starts at zero, grows gradually
5. **MSE > BCE** for message loss — no trivial equilibrium
6. **Message scale must dominate** total image losses (5.0 vs ~3.5)
7. **ResNet50 decoder fails** — AdaptiveAvgPool2d destroys spatial information
8. **GAN LR must be conservative** — 0.1× encoder LR, not 2× (causes collapse)
9. **De-annealing works** — relaxing strength from a converged model is faster than training from scratch
10. **Bootstrap detection** — if `prob_std < 0.02` by step 2000, kill and restart (~50% failure rate)
11. **Downsampling is a feature, not a bottleneck** — 512→256 forces low-frequency residual patterns that are perceptually invisible. 512→512 allows high-frequency patterns that have higher PSNR but are more visible and less robust to JPEG/blur
12. **More bits don't proportionally cost accuracy** — 128 bits at 512→512 achieved 93.8% vs 94.6% for 80 bits at 512→256, only -0.8% despite 60% more bits. Encoder capacity is underutilized at 80 bits
13. **LPIPS 1.5 + GAN 1.5 enables content-adaptive encoding** — higher perceptual loss weights teach the encoder to concentrate residuals in textured regions where changes are invisible, avoiding smooth areas. This is the right spatial strategy but must be paired with sufficient bits-per-strength budget
14. **32 bits is the sweet spot for accuracy** — at 32 bits with strength 0.013, the model achieves 98.9% accuracy and 39.3 dB PSNR. The encoder has excess capacity, bootstraps instantly (no collapse), and de-anneals gracefully. Tradeoff: only 32 raw bits (4 bytes) of payload
15. **High-frequency artifacts persist with 416/512 decoder** — even at 39+ dB PSNR with content-adaptive LPIPS/GAN losses, artifacts remain visible when the decoder resolution is close to the encoder's. The 512→256 downsampling remains the best perceptual quality mechanism found so far. Future work: residual blurring, TV loss, or low-res residual generation to suppress HF without downsampling
