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
Steps 10k-130k:  exponential anneal from 1.0 → target (e.g., 0.014)
                  strength = 1.0 × (0.014)^progress
Steps 130k-200k: strength = target (fine-tuning at fixed budget)
```

Exponential annealing (v11+) is preferred over linear for ranges spanning multiple orders of magnitude. Linear annealing rushes through the critical low-strength regime.

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
| **v10** | **32** | **0.014** | **38.70 dB** | **99.4%** | **98.8%** | **512→256, bilinear upsample, dilated E_post, Laplacian loss, early FFL — smoothest residuals** |
| v10 s012 | 32 | 0.012 | — | — | — | De-annealed from v10 |
| **v10 s010** | **32** | **0.010** | **41.06 dB** | **97.5%** | **95.6%** | **De-annealed from v10 — 41 dB milestone, highest PSNR** |
| v11 | 64 | 0.014 | 38.09 dB | 95.0%* | 92.8%* | 512→416, encoder-side blur σ=1.0, exponential annealing — superseded by v12 |
| **v12** | **64** | **0.014** | **38.11 dB** | **95.5%** | **93.2%** | **512→416, decoder-side blur σ=0.8, blurred LPIPS/GAN, clean output, content-adaptive — production model** |
| **v12 s010** | **64** | **0.010** | **40.72 dB** | **94.0%** | **89.8%** | **De-annealed from v12 — 40 dB at 64 bits** |
| v13 | 64 | 0.020 | 35.84 dB | 98.5% | 97.3% | Learned mask, no blurred LPIPS — stopped early, unsuccessful |

*v11 evaluated at 130k (5 images only). v12 evaluated at 200k (50 images). v13 evaluated at 150k (50 images), stopped early.

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

### HF Artifact Reduction Experiment (v10)

v10 targeted the high-frequency artifact problem identified in v9. Five changes were applied simultaneously, training from scratch on 32 bits:

1. **Bilinear message upsampling** — `F.interpolate(mode="bilinear")` replaces nearest-neighbor in `prepare_message`, eliminating 50×50 grid discontinuities
2. **Dilated E_post** — added `Conv2d(32, 32, 3, dilation=2)` layer, increasing receptive field from ~7×7 to ~11×11
3. **Decoder 256** — back to 512→256 downsampling (from 416), forcing low-frequency residual patterns
4. **Early FFL** — Focal Frequency Loss activated from step 10k (was delayed to 40k in v9)
5. **Laplacian loss** — new loss term penalizing `|Laplacian(residual)|`, directly suppresses high-frequency spatial patterns. Scale 1.0, ramped over 50k steps.

**Results (32 bits, 512→256, 5-image evaluation):**

| Strength | Steps | PSNR | Accuracy | JPEG Q10 | Blur σ=3 | Bright ±0.5 |
|----------|-------|------|----------|----------|----------|-------------|
| 0.014 | 200k | 38.70 dB | 99.4% | 98.8% | 99.4% | 90.0% |

**Key findings:**
- **Significantly smoother residuals** than v9. High-frequency grid artifacts are eliminated. Residual patterns are visibly lower-frequency and more uniform.
- **Accuracy matches v9** (99.4% vs 99.2%) despite the additional constraints — 32 bits has enough capacity headroom.
- **PSNR slightly lower** than v9 s012 (38.70 vs 39.75 dB at comparable accuracy) because the 256 decoder forces larger spatial patterns.
- **Robustness excellent**: JPEG Q10 98.8%, blur σ=3 99.4% — better than v9 on blur (low-frequency patterns survive better).
- **Brightness robustness weaker**: 90.0% vs 94.6% for v9 s012. The 256 decoder's coarser patterns are more affected by global brightness shifts.
- **Bootstrap fastest ever**: 100% accuracy by step 900 (bilinear upsampling gives smoother gradients for the decoder to learn from).
- **Recovery slower than v9**: After annealing completed at step 90k, accuracy dropped to ~65-75% and recovered to ~92% average by step 200k (vs v9 reaching 99% by 160k). The combined constraints make the squeeze phase harder.
- **Not content-adaptive**: Unlike v9 which concentrated residuals in textured regions, v10 spreads residuals uniformly. The 256 decoder can't resolve fine spatial variations, so the encoder has no incentive for selective placement. Artifacts are smoother but equally visible in smooth regions (sky, walls) and textured regions.

**Training dynamics:**
- Steps 0-10k: Bootstrap phase (100% accuracy by step 900)
- Steps 10k-90k: Annealing squeeze (strength 1.0→0.014), Laplacian and FFL active from step 10k
- Steps 90k-200k: Recovery phase (65%→92% average accuracy, prob_std 0.25→0.43)
- Laplacian loss: ~0.01 during squeeze, ~0.008-0.010 at convergence (fading as residual amplitude shrinks)

**Implication:** The 256 decoder solves HF artifacts but prevents content-adaptive encoding. Future directions: 416 decoder + residual blurring (smooth patterns with spatial selectivity), or learned spatial mask (`use_mask=True`) with 256 decoder.

### Content-Adaptive Clean Output (v12 — Production Model)

v12 solved the central tension of the v9-v11 line: how to get content-adaptive residual placement (encoder concentrates signal in textured regions) with clean, sharp encoded images (no blur artifacts).

**Architecture**: 64 bits, 512→416 decoder, exponential strength annealing, decoder-side Gaussian blur (σ=0.8).

**Key innovations:**

1. **Exponential strength annealing** — `strength = initial × (target/initial)^progress`. Spends equal time per order of magnitude, much smoother than linear for the 1.0→0.014 range. Anneal window: steps 10k-130k.

2. **Decoder-side blur** — Gaussian blur (σ=0.8) applied to the decoder's input during training. The decoder learns to decode from blurred images, forcing the encoder to create low-frequency patterns that survive the blur. At inference, no blur is applied — the encoded image stays clean and sharp.

3. **Blurred LPIPS/GAN for content-adaptive gradients** — LPIPS and GAN discriminator receive `blur(encoded)` instead of `encoded`. The blur converts the residual into large-scale patterns that LPIPS can differentially penalize by image region: high penalty in smooth areas (sky, walls), low penalty in textured areas (grass, fabric). This drives the encoder to concentrate residuals in texture. L2 and FFL see the real (unblurred) encoded image to maintain pixel-level quality.

**Why blurred LPIPS drives content-adaptivity:**
- Without blur, the encoder can hide signal in fine per-pixel noise that LPIPS can't detect well → no spatial selectivity
- With blur, the residual patterns are visible to LPIPS at a coarse scale → LPIPS penalizes more in smooth regions → encoder avoids smooth regions
- The blur doesn't affect the actual output — it only shapes the gradient signal

**Results (50-image evaluation):**

| Strength | Steps | PSNR | Accuracy | JPEG Q10 | Blur σ=3 | Bright ±0.5 |
|----------|-------|------|----------|----------|----------|-------------|
| 0.014 | 200k | 38.11 dB | 95.5% | 93.2% | 95.7% | 92.1% |
| 0.012 | 230k | 39.11 dB | 93.2% | 90.9% | 94.5% | 89.8% |
| **0.010** | **260k** | **40.72 dB** | **94.0%** | **89.8%** | **93.6%** | **87.3%** |

**Per-image adaptive strength:** The trained strength (0.010-0.014) can be adjusted at inference. The encoder produces a raw residual bounded by softsign — changing strength just scales the amplitude. Use lower strength (0.008-0.010) for easy images with lots of texture (better PSNR), bump to 0.015-0.020 for hard images with smooth regions (better accuracy). No retraining needed.

**Texture masking:** Post-processing step that attenuates the residual in smooth image regions based on local variance of the input image. Computes a spatial mask in `[floor, 1.0]` where smooth regions get `floor` and textured regions get ~1.0, then multiplies the residual by the mask before adding to the image. No retraining needed — the decoder tolerates partial attenuation. Available via `picode encode --texture-mask --mask-floor 0.3`.

v12 at 200k steps (strength 0.014), evaluated on 20 images:

| Mode | PSNR | Bit acc | Clean msg | JPEG Q50 | JPEG Q10 | Noise 0.05 | Blur 2.0 |
|------|------|---------|-----------|----------|----------|------------|----------|
| No mask | 38.0 dB | 98.1% | — | — | — | — | — |
| floor=0.5 | 39.4 dB | — | — | — | — | — | — |
| floor=0.3 | 40.0 dB | 95.9% | — | — | — | — | — |

With BCH(63,36,t=5) hard decoding (36 data bits):

| Mode | Clean | JPEG Q50 | JPEG Q10 | Noise 0.05 | Blur 2.0 |
|------|-------|----------|----------|------------|----------|
| No mask | 85% | 90% | 75% | 65% | 85% |
| floor=0.5 | 80% | 75% | 45% | 50% | 75% |
| floor=0.3 | 50% | 40% | 35% | 20% | 55% |

With LDPC(60,32) soft decoding (32 data bits) — **much better**:

| Mode | Clean | JPEG Q50 | JPEG Q10 | Noise 0.05 | Blur 2.0 |
|------|-------|----------|----------|------------|----------|
| No mask | 95% | 95% | 100% | 90% | 95% |
| floor=0.5 | 100% | 95% | 80% | 75% | 100% |
| floor=0.3 | 90% | 95% | 65% | 50% | 100% |

**Key findings:**
- Texture masking trades accuracy for PSNR: +2.0 dB at floor=0.3, +1.4 dB at floor=0.5
- BCH hard decoding suffers disproportionately because the mask pushes some images past the correction threshold
- LDPC soft decoding exploits decoder logit confidence and tolerates the mask much better — at floor=0.5 it's near-perfect on clean/JPEG Q50 while gaining +1.4 dB PSNR
- **Recommended production config:** LDPC soft decoding + texture mask floor=0.5 (39.4 dB, 100% clean recovery, 95% JPEG Q50)

**Bootstrap at 64 bits:** ~9% success rate per attempt (1 in 11). Much harder than 32 bits (~50%). Auto-retry loop needed. The decoder-side blur does not interfere with bootstrapping since the encoder's residual is unblurred during the bootstrap phase.

### Learned Spatial Mask Experiment (v13 — Unsuccessful)

v13 attempted to replace the blurred LPIPS/GAN approach with a learned per-pixel spatial mask. The mask head (4,641 params, 0.3% of encoder) predicted `mask(x,y) ∈ [0, 1]` from U-Net features, with `residual = strength × softsign(raw) × mask`. All losses (L2, LPIPS, GAN, FFL) received the real encoded image — no blurred LPIPS/GAN. Strength target 0.020 (higher than v12's 0.014 to give the mask headroom for selectivity).

**Results (50-image evaluation at 150k steps):** 98.5% accuracy, 35.84 dB PSNR, JPEG Q10 97.3%.

**Why it failed:**
- **Diagonal curve artifacts**: the encoder learned structured spatial patterns (diagonal curves) visible in the encoded images. Without blurred LPIPS, nothing suppressed these HF structured patterns — LPIPS on the real output couldn't penalize them effectively.
- **Mask not selective enough**: `mask_mean=0.83, mask_std=0.20` — the mask reduced residual to ~83% on average but didn't sharply separate textured from smooth regions. With effective residual of `0.020 × 0.83 = 0.017`, artifacts were more visible than v12's 0.014.
- **High accuracy, poor visual quality**: 98.5% accuracy was excellent (better than v12's 95.5%), but the encoded images looked worse due to the structured artifacts.

**Key lesson**: the learned mask controls WHERE to place residual, but not WHAT SHAPE the residual takes. Without blurred LPIPS, the encoder is free to create structured HF patterns that are perceptually conspicuous. The mask and blurred LPIPS solve different problems — the mask handles spatial selectivity, blurred LPIPS handles pattern smoothness. Both may be needed together.

**Implication for future work**: a v14 could combine the learned mask (spatial control) with blurred LPIPS (pattern smoothness). The mask would route residual to textured regions while blurred LPIPS ensures the residual patterns in those regions are smooth and invisible.

### Exponential vs Linear Annealing (v11/v12)

Linear annealing spends most of its time in the high-strength regime and rushes through the critical low-strength transition. Exponential annealing treats each order of magnitude equally:

| Step | Progress | Exponential | Linear |
|------|----------|-------------|--------|
| 10k  | 0.00 | 1.000 | 1.000 |
| 50k  | 0.33 | 0.242 | 0.675 |
| 70k  | 0.50 | 0.118 | 0.507 |
| 90k  | 0.67 | 0.057 | 0.340 |
| 130k | 1.00 | 0.014 | 0.014 |

The exponential schedule gives the encoder more time in the 0.1→0.014 range where fine-tuning efficiency matters most. Implemented as `anneal_schedule: exponential` in the config.

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

**32-bit models (512→256 decoder, v10 architecture):**

| Strength | PSNR (dB) | Raw Accuracy | PSNR delta | Accuracy delta |
|----------|-----------|-------------|------------|----------------|
| 0.014 | 38.70 | 99.4% | — | — |
| 0.010 | 41.06 | 97.5% | +2.36 dB | -1.9% |

With 32 bits, de-annealing barely costs accuracy. Each 0.001 step gains ~0.5 dB PSNR for ~0.3% accuracy. At 0.011, the v9 model crosses 40 dB — matching TrustMark-B territory. The v10 architecture (256 decoder, Laplacian loss, smooth residuals) pushes further: v10 s010 reaches **41.06 dB** with 97.5% accuracy and 95.6% JPEG Q10 robustness.

**64-bit models (512→416 decoder, LPIPS 1.5, GAN 1.5, decoder blur σ=0.8):**

| Strength | PSNR (dB) | Raw Accuracy | PSNR delta | Accuracy delta |
|----------|-----------|-------------|------------|----------------|
| 0.014 | 38.11 | 95.5% | — | — |
| 0.012 | 39.11 | 93.2% | +1.0 dB | -2.3% |
| 0.010 | 40.72 | 94.0% | +1.6 dB | +0.8% |

With 64 bits, de-annealing from 0.014→0.010 trades ~1.5% accuracy for +2.6 dB PSNR. The accuracy recovery at 0.010 (94.0% vs 93.2% at 0.012) suggests the model benefits from more fine-tuning time at the target strength.

In linear terms (RMS residual amplitude on 0-255 scale):
- 0.010 strength → 40.7 dB → ~2.3 pixel levels modified per pixel
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

### Robustness (v12, 64 bits, strength 0.014, 50 images — production model)

| Distortion | Bit Accuracy |
|------------|-------------|
| Clean | 95.5% |
| JPEG Q10 | 93.2% |
| JPEG Q50 | 95.7% |
| JPEG Q90 | 95.8% |
| Gaussian noise σ=0.05 | 95.2% |
| Gaussian noise σ=0.10 | 92.6% |
| Gaussian blur σ=1.0 | 96.6% |
| Gaussian blur σ=3.0 | 95.7% |
| Brightness ±0.1 | 95.1% |
| Brightness ±0.3 | 93.3% |
| Brightness ±0.5 | 92.1% |
| Contrast ±0.3 | 95.6% |
| Contrast ±0.5 | 95.2% |

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
| **PicoTrust v10** | 2026 | 32 | 512→256 | 38.70 | 99.4% | 98.8% (Q10) | **6.3M** |
| **PicoTrust v10 s010** | 2026 | 32 | 512→256 | 41.06 | 97.5% | 95.6% (Q10) | **6.3M** |
| **PicoTrust v12** | 2026 | 64 | 512→416 | 38.11 | 95.5% | 93.2% (Q10) | 12.8M |
| **PicoTrust v12 s010** | 2026 | 64 | 512→416 | 40.72 | 94.0% | 89.8% (Q10) | 12.8M |

*v8 stopped early (100k steps). Content-adaptive encoding but 96 bits exceeded capacity at strength 0.014.

*StegaStamp PSNR varies 30-37 dB across evaluations; lower numbers reflect aggressive encoding for physical print-and-photograph robustness.

### Analysis

**PicoTrust strengths:**
- **Extreme JPEG robustness**: 92-98% accuracy at JPEG Q10 (most methods test at Q50+). This matters for real-world scenarios where images are heavily recompressed.
- **Compact model**: 6.3M params — 8.5× smaller than StegaStamp, suitable for mobile deployment.
- **No colour shifts**: Grayscale residual guarantees R=G=B perturbation. No hue-based artifacts.
- **Robust across all distortions**: No single failure mode (unlike TrustMark failing JPEG, or StegaStamp failing flips).

**PicoTrust limitations:**
- **PSNR gap**: 38-41 dB (v9/v10/v12) vs 42-51 dB for TrustMark/InvisMark. Gap is closing — v12 s010 at 40.7 dB is within ~1 dB of TrustMark-Q at 64 bits. Modern methods use pretrained backbones (ConvNeXT, etc.) for higher PSNR.
- **Bootstrap fragility**: ~50% failure rate at 32 bits, ~9% at 64 bits. Auto-retry loop needed for higher bit counts.
- **Strength-accuracy cliff**: Depends on bit count. For 64 bits at 0.010, accuracy is 94%. For 32 bits at 0.010, accuracy is 97.5%. Higher bit counts have steeper cliffs.

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

Example configuration (v12 — production model, 64 bits, content-adaptive):

```yaml
experiment_name: picotrust_v12

model:
  type: picotrust
  encoder_size: 512
  decoder_size: 416

training:
  num_bits: 64
  num_steps: 200000
  lr: 0.0001
  residual_strength: 1.0
  residual_strength_anneal_target: 0.014
  residual_strength_anneal_start: 10000
  residual_strength_anneal_steps: 120000
  anneal_schedule: exponential
  decoder_blur_sigma: 0.8
  phase2_step: 60000
  phase2_decoder_lr_scale: 0.1

loss:
  message: { scale: 5.0 }
  l2: { scale: 1.5, ramp_steps: 50000 }
  lpips: { scale: 1.5, ramp_steps: 50000 }
  ffl: { scale: 1.0, ramp_steps: 50000, delay_steps: 0 }
  message_loss_type: mse
  gan_config: { enabled: true, discriminator_lr: 0.000015, g_loss_scale: 1.5 }

distortion:
  strategy: curriculum
```

See `picode-model/configs/` for all training configurations (v1-v12).

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
15. **High-frequency artifacts persist with 416/512 decoder** — even at 39+ dB PSNR with content-adaptive LPIPS/GAN losses, artifacts remain visible when the decoder resolution is close to the encoder's. The 512→256 downsampling remains the best perceptual quality mechanism found so far
16. **Bilinear upsampling + dilated E_post + Laplacian loss produce smoother residuals** — v10 combined all three with 256 decoder and early FFL. Grid artifacts eliminated, residuals visibly smoother. But the 256 decoder prevents content-adaptive encoding — residuals are uniform rather than texture-concentrated
17. **256 decoder and content-adaptivity are mutually exclusive** — the 256 decoder can't resolve fine spatial detail, so the encoder has no incentive for selective placement. Getting both smooth AND content-adaptive requires a different approach: either 416 decoder + residual blurring, or learned spatial masks with 256 decoder
18. **Laplacian loss fades under strength annealing** — once residual amplitude anneals to 0.014, the Laplacian values become small (~0.01). The loss is most useful during the squeeze phase (steps 10k-90k) when residuals are still large. After annealing, the strength bound itself constrains HF patterns
19. **Encoder-side blur prevents bootstrap** — even σ=0.1 at step 0 collapses bootstrap (0/15 attempts at 64 bits). The blur destroys the spatial structure the decoder needs to initially learn from. Fix: use decoder-side blur only, or delay encoder blur until after bootstrap
20. **Exponential annealing > linear** — for multiplicative parameters like strength spanning 2 orders of magnitude (1.0→0.014), exponential decay spends equal time per order of magnitude. Linear wastes time at high strength and rushes through the critical low-strength regime
21. **Decoder-side blur teaches low-frequency decoding** — blurring the decoder's input during training forces the decoder to rely only on low-frequency patterns. The encoder naturally learns to create patterns that survive the blur (low-frequency) without any architectural constraint on its output. Clean encoded images at inference
22. **Blurred LPIPS drives content-adaptivity without modifying output** — computing LPIPS on `blur(encoded)` makes the residual visible to LPIPS at a coarse scale, enabling differential penalization by image region (more in smooth areas, less in textured). The actual encoded image remains unblurred. The blur scale for LPIPS and decoder should match (same σ) since LPIPS guides patterns at the scale the decoder uses
23. **64 bits bootstraps ~9% of the time** — vs ~50% at 32 bits. The decoder has 2× more outputs to learn, making random initialization less likely to produce useful signal. Auto-retry loop essential
24. **Strength is adjustable at inference** — the softsign bound `strength × raw/(1+|raw|)` can be evaluated at any strength without retraining. Use lower strength for easy (textured) images, higher for hard (smooth) images. Per-image adaptive strength for production use
25. **L2 on real output + LPIPS on blurred output = clean + adaptive** — L2 maintains pixel quality of the actual encoded image. LPIPS on the blurred version provides content-adaptive spatial guidance. Using the same blur for both would give stronger adaptivity but no pixel-level quality control on the real output
26. **LDPC soft decoding >> BCH hard decoding** — at ~96% bit accuracy, BCH treats every bit as equally certain and fails when error count exceeds `t`. LDPC exploits soft probabilities from decoder logits (via sigmoid), gaining ~20% message recovery over BCH at comparable code rates. The confidence information in logits is valuable — bits the decoder is uncertain about get less weight in belief propagation. Use LDPC for production
27. **Texture masking + LDPC is the right production combo** — post-processing texture mask (floor=0.5) attenuates residuals in smooth regions, gaining +1.4 dB PSNR. BCH can't tolerate the accuracy drop, but LDPC soft decoding handles it gracefully: 100% clean message recovery, 95% at JPEG Q50, 80% at JPEG Q10
28. **Learned mask alone doesn't solve visual quality** — a learned spatial mask (v13) successfully controls WHERE residual goes (mask_mean=0.83, high accuracy) but doesn't control WHAT SHAPE the residual takes. Without blurred LPIPS, the encoder creates structured HF patterns (diagonal curves) that are perceptually conspicuous. The mask and blurred LPIPS address different problems: mask = spatial selectivity, blurred LPIPS = pattern smoothness. Both are needed for high visual quality
29. **High accuracy ≠ good visual quality** — v13 achieved 98.5% accuracy (better than v12's 95.5%) but worse visual quality due to structured artifacts. Optimizing for bit accuracy alone doesn't produce invisible encoding — perceptual loss design matters as much as the spatial strategy
