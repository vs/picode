# Sobel Adaptive Encoding

Production encoding strategy for PicoTrust that achieves near-invisible steganography with reliable LDPC message recovery. Combines three techniques: scale-from-trained-strength residual reuse, Sobel gradient texture masking, and batch ID selection.

## Core Idea

Train the model at a comfortable strength (s=0.020) where it learns strong content-adaptive patterns. At inference, scale the residual down to the minimum strength where LDPC can still recover the message. The encoder never sees the low strength during training — the spatial pattern is preserved exactly, just attenuated.

## Why This Works

1. **Post-annealing degrades content-adaptivity.** Training at low strength (s=0.010) forces the encoder to spread residuals uniformly to maintain accuracy. The model trained at s=0.020 has excess capacity and concentrates residuals in textured regions — exactly what we want for invisibility.

2. **LDPC bridges the accuracy gap.** At s=0.012–0.015 (scaled from s=0.020), raw bit accuracy drops to ~88–92%. LDPC(48,26) or LDPC(72,38) with soft-decision belief propagation corrects the remaining errors. The model doesn't need to be perfect — just good enough for ECC.

3. **Batch ID selection eliminates hard messages.** Different bit patterns produce different residual shapes. Some align well with the image's texture (easy to decode), others don't (hard). By trying multiple random IDs and picking the best, we avoid hard messages entirely.

## Algorithm

```
encode(image):
    # 1. Get raw residual at trained strength
    raw_residual = encoder(image, ldpc_encode(candidate_id), strength=1.0) - image

    # 2. Compute Sobel texture mask
    mask = sobel_texture_mask(image, blur_sigma=5.0, floor=0.85)

    # 3. Try batch of random IDs at escalating strengths
    for strength in [0.010, 0.012, 0.015, 0.020]:
        for id in random_ids(batch_size=64):
            ldpc_msg = ldpc_encode(id)
            raw = encoder(image, ldpc_msg, strength=1.0) - image
            encoded = image + strength * raw * mask
            if ldpc_decode(decoder(encoded)) succeeds:
                return encoded, id, strength

    # 4. Fallback: use trained strength without mask
    return image + 0.020 * raw_residual, fallback_id, 0.020
```

### Batch ID Selection (Key Innovation)

The ID assigned to an image is a random database key — semantically meaningless. Instead of assigning one random ID and hoping it decodes at low strength, try 64 candidates in one batched GPU forward pass and pick the one with highest decode confidence:

```python
# 64 candidates, one forward pass (~0.5s on mobile GPU)
candidates = torch.randint(0, 2, (64, ldpc_k))
ldpc_msgs = stack([ldpc.encode(c) for c in candidates])
encoded_batch = encoder(img.expand(64, -1, -1, -1), ldpc_msgs)
residuals = encoded_batch - img
scaled = img + target_strength * residuals * mask
probs = decoder(scaled)

# Pick message with highest minimum bit confidence
confidence = (probs - 0.5).abs()
best = confidence.min(dim=1).values.argmax()
```

With 64 candidates at s=0.012, success rate goes from ~18% per trial to ~99.9% across the batch.

### Sobel Texture Mask

Attenuates the residual in smooth regions (sky, walls) while preserving full signal in textured regions (foliage, fabric, hair).

```python
def sobel_texture_mask(image, blur_sigma=5.0, floor=0.85):
    gray = image.mean(dim=1, keepdim=True)
    gx = conv2d(gray, sobel_x, padding=1)
    gy = conv2d(gray, sobel_y, padding=1)
    grad_mag = (gx**2 + gy**2).sqrt()
    # Smooth with large Gaussian for coherent regions
    grad_smooth = gaussian_blur(grad_mag, sigma=blur_sigma)
    grad_norm = grad_smooth / grad_smooth.max()
    return floor + (1 - floor) * grad_norm
```

**Default: σ=5.** Matches training. Good balance of spatial coherence and boundary precision.

**Sigma is a pure inference-time parameter** — no retraining needed to change it per image.

| Sigma | RF (pixels) | Character | Best for |
|-------|-------------|-----------|----------|
| 1.0 | ~6 | Sharp, pixel-level | Images with fine texture/smooth transitions (e.g. sky gradients next to trees) |
| 3.0 | ~18 | Moderate | General purpose |
| **5.0** | **~30** | **Smooth, default** | **Most images — trained with this value** |
| 8.0 | ~48 | Very smooth | Large uniform regions |
| 15.0 | ~90 | Regional | Big flat backgrounds, but can bleed into adjacent smooth areas |

**Sigma sweep results (50 images, composite distortion gate):**

| Sigma | Success | Avg PSNR | % at s≤0.015 |
|-------|---------|----------|-------------|
| 1.0 | 94.8% | 41.0 dB | 41.6% |
| 3.0 | 93.2% | 41.1 dB | 40.8% |
| 5.0 | 95.2% | 41.0 dB | 41.2% |
| 8.0 | 95.6% | 41.0 dB | 38.4% |
| 15.0 | 95.2% | 41.1 dB | 40.8% |

Metrics are nearly identical — the difference is purely visual. Large σ creates smoother masks but can bleed texture detection across boundaries (e.g. sky gradient near trees gets marked as textured with σ=15, showing artifacts in the sky). Small σ follows boundaries tightly but can create sharper mask transitions.

**Recommendation:** Use σ=5 as default. Try σ=15 for images with large uniform backgrounds. Try σ=1 for images with fine texture/smooth boundaries (gradient skies).

**Why not local variance (previous approach)?**
- 7×7 box filter has tiny receptive field — noisy, pixel-level decisions
- Sharp transitions at texture/smooth boundaries create visible artifacts
- Sobel+blur produces spatially coherent masks matching human perception of texture

### Strength Ladder

| Strength | With mask (floor=0.85) | Without mask | Typical PSNR |
|----------|----------------------|-------------|-------------|
| 0.010 | First try | — | 44–46 dB |
| 0.012 | Second try | — | 43–45 dB |
| 0.015 | Third try | Fourth try | 41–43 dB |
| 0.020 | Fifth try (floor=0.75) | Sixth try | 39–41 dB |
| 0.025 | — | Fallback | 37–38 dB |

## Results

### b7238s20p00d00_sobel — 72 bits, Sobel mask in training (BEST MODEL)

**The breakthrough model.** No blur at all — Sobel mask applied during training directly teaches the encoder where to put signal. The encoder-decoder pair optimizes end-to-end with the mask constraint.

| Metric | Raw s=0.020 | Sobel adaptive (composite gate) |
|--------|------------|-------------------------------|
| Accuracy | 98.5% | — |
| PSNR | 35.90 dB | 41.0 dB avg |
| TRC | 0.579 | — |
| JPEG Q10 | 98.6% | — |
| LDPC success (5 trials) | — | **93.6%** |
| % at s≤0.015 | — | **39.6%** |

**Why it's the best:** The model was trained WITH the Sobel mask, so the encoder already knows smooth regions get attenuated. It doesn't waste capacity there — it concentrates signal in textures from the start. Previous models were trained without the mask, so applying it at inference was fighting the encoder's learned pattern.

### Comparison (all models, composite distortion gate, 5 trials/image)

| Model | Success | Avg PSNR | % at s≤0.015 |
|-------|---------|----------|-------------|
| **b7238s20p00d00_sobel** | **93.6%** | **41.0 dB** | **39.6%** |
| b7238s20p03d03 | 83.6% | 40.5 dB | 18.4% |
| v22 (48b, σ=0.7) | 81.2% | 39.6 dB | 13.6% |
| b7238s20p03d00 | 75.6% | 39.6 dB | 12.0% |

The Sobel-trained model crushes all blur-based models — +10% success rate, +0.5 dB PSNR, 2× more trials at low strength.

### Previous blur-based models (for reference)

#### v22 (b4826s20p07d07) — 48 bits, σ=0.7

Best visual quality among blur-based models. 48 bits with LDPC(48,26) = 26 payload bits (67M IDs). Fewer bits = less encoding pressure = smoother residuals. But superseded by the Sobel-trained model on both metrics and visual quality.

#### b7238s20p03d00 — 72 bits, LPIPS blur only (no decoder blur)

Best visual quality among 72-bit blur-based models. Fine-grained residuals blend into texture. But without the Sobel mask in training, it can't match the Sobel-trained model's content-adaptivity.

#### b7238s20p03d03 — 72 bits, both blur σ=0.3

Highest LDPC success among blur-based models (83.6%) but coarser residual patterns visible as "halos" when attenuated.

## Key Lessons

1. **Train with the Sobel mask, not blur.** Blur was a proxy for "penalize smooth regions." The Sobel mask does it directly — the encoder learns end-to-end where signal is allowed. This produces 93.6% LDPC success vs 83.6% for the best blur model.

2. **Don't post-anneal if you have LDPC.** Post-annealing trades content-adaptivity for low-strength accuracy. With LDPC + batch ID selection, you don't need high accuracy at low strength — you need a high-quality spatial pattern that hides well. The s=0.020 checkpoint has that pattern; post-annealing destroys it.

3. **Message selection is free.** The ID is a random database key. Choosing the "easy" ID for a given image costs nothing semantically but dramatically improves encode quality.

4. **No blur needed at all.** The Sobel-trained model uses no decoder blur and no perceptual blur. The mask alone provides content-adaptivity. Blur-based models produce coarser patterns that are more visible when attenuated.

5. **Sobel > local variance for texture masks.** Large-kernel Sobel gradient produces smooth, perceptually coherent masks. Local variance with small kernels creates noisy, artifact-prone masks.

6. **Scale, don't retrain.** The content-adaptive pattern learned at s=0.020 is valuable. Preserve it by linear scaling at inference rather than retraining at lower strength.

7. **Sigma is tunable at inference.** σ=5 is the default (matches training). σ=15 for large uniform backgrounds, σ=1 for fine texture/smooth boundaries (gradient skies). No retraining needed.

## Production Architecture

```
Client uploads photo
  → Server encodes at s=1.0 with 64 candidate IDs (batched)
  → For each candidate, scales residual × Sobel mask at s=0.012
  → Picks ID with highest decode confidence
  → If none pass LDPC at s=0.012, escalates to s=0.015, then s=0.020
  → Stores ID → content mapping in database
  → Returns encoded image (PSNR 40–45 dB, visually clean)

Client scans encoded photo
  → Decoder extracts bit probabilities
  → LDPC soft decode → 26 payload bits → ID
  → API lookup: ID → content (short link, metadata, etc.)
```

## Recommended Model

**b7238s20p00d00_sobel** — the production model:
- 72 channel bits, LDPC(72,38) = 38 payload bits, 274B IDs
- No blur at all — Sobel mask in training provides content-adaptivity
- 93.6% single-message LDPC success through composite distortions
- With batch ID selection (64 candidates): ~99.9%+ projected success
- Sobel mask σ=5 default, tunable per image (σ=1–15) at inference
- Trained at s=0.020, adaptive inference at s=0.012–0.020

**For smaller namespace (67M IDs):** v22 (b4826s20p07d07)
- 48 channel bits, LDPC(48,26) = 26 payload bits
- Fewer bits = less encoding pressure, but superseded by Sobel model on metrics
