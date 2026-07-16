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

**Why Sobel + large blur (σ=5)?**
- Sobel detects edges/texture at pixel level
- Gaussian blur σ=5 (RF ≈ 30 pixels) creates smooth, coherent mask regions
- No sharp transitions between textured and smooth areas — avoids mask artifacts
- Floor=0.85 means smooth regions still get 85% of signal (not zeroed out)

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

### v22 (b4826s20p07d07) — 48 bits, σ=0.7

The best visual quality model. 48 bits with LDPC(48,26) = 26 payload bits (67M IDs).

| Metric | Single message | With batch ID (projected) |
|--------|---------------|--------------------------|
| LDPC success | 80% | ~99%+ |
| Avg PSNR | 40.0 dB | ~42 dB |
| % at s≤0.015 | 18% | ~60%+ |
| Visual quality | Outstanding | Outstanding |

**Why v22 looks best:** Fewer bits (48 vs 72) = less encoding pressure. Higher blur σ=0.7 = smoother residual patterns. The combination produces residuals that are nearly invisible even without the texture mask.

### b7238s20p03d00 — 72 bits, LPIPS blur only (no decoder blur)

Best visual quality among 72-bit models. No decoder blur means the encoder uses the full frequency spectrum — fine-grained residuals that blend into image texture naturally. LPIPS blur σ=0.3 still provides content-adaptivity during training.

| Metric | Single message | With batch ID (projected) |
|--------|---------------|--------------------------|
| LDPC success | 82% | ~99%+ |
| Avg PSNR | 39.9 dB | ~41 dB |
| % at s≤0.015 | 16% | ~55%+ |
| Visual quality | Excellent — smooth, blends into texture | Excellent |

### b7238s20p03d03 — 72 bits, both blur σ=0.3

Higher LDPC success rate but more visible residuals. Decoder blur σ=0.3 forces coarser low-frequency patterns that appear as visible "halos" when attenuated by the Sobel mask.

| Metric | Single message | With batch ID (projected) |
|--------|---------------|--------------------------|
| LDPC success | 90% | ~99%+ |
| Avg PSNR | 40.5 dB | ~42 dB |
| % at s≤0.015 | 22% | ~65%+ |
| Visual quality | Good — but coarser patterns more noticeable | Good |

### Visual Quality Ranking

With batch ID selection pushing all models to ~99%+ success, visual quality becomes the differentiator:

1. **v22 (48b, σ=0.7)** — best overall, fewest bits + smoothest patterns
2. **b7238s20p03d00 (72b, LPIPS only)** — best 72-bit, fine residuals blend into texture
3. **b7238s20p03d03 (72b, both blur)** — coarser patterns, more visible at low strength

**Why d00 beats d03 visually:** Without decoder blur, the encoder learns to use fine spatial frequencies that match natural image texture. When scaled down, these fine patterns become imperceptible. With decoder blur σ=0.3, the encoder is forced to use coarser patterns (the decoder can't read fine detail after blur) — these low-frequency residuals are more visible to the human eye as smooth "waves" or "halos" in flat regions, even after Sobel masking.

## Key Lessons

1. **Don't post-anneal if you have LDPC.** Post-annealing trades content-adaptivity for low-strength accuracy. With LDPC + batch ID selection, you don't need high accuracy at low strength — you need a high-quality spatial pattern that hides well. The s=0.020 checkpoint has that pattern; post-annealing destroys it.

2. **Message selection is free.** The ID is a random database key. Choosing the "easy" ID for a given image costs nothing semantically but dramatically improves encode quality.

3. **Fewer bits = better visual quality.** 48 bits (26 payload) is enough for 67M IDs. Don't use 72 bits unless you need the larger namespace.

4. **No decoder blur = better visual quality with Sobel adaptive.** Decoder blur forces low-frequency residuals that look like visible halos when attenuated. Without decoder blur (d00), the encoder uses fine frequencies that blend into texture. LPIPS blur alone is sufficient for content-adaptivity.

5. **Sobel > local variance for texture masks.** Large-kernel Sobel gradient produces smooth, perceptually coherent masks. Local variance with small kernels creates noisy, artifact-prone masks.

6. **Scale, don't retrain.** The content-adaptive pattern learned at s=0.020 is valuable. Preserve it by linear scaling at inference rather than retraining at lower strength.

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

## Recommended Models

**For best visual quality:** v22 (b4826s20p07d07)
- 48 channel bits, LDPC(48,26) = 26 payload bits, 67M IDs
- σ=0.7 decoder + perceptual blur
- Best overall visual quality — fewest bits, smoothest patterns

**For 72-bit namespace with best visuals:** b7238s20p03d00
- 72 channel bits, LDPC(72,38) = 38 payload bits, 274B IDs
- No decoder blur, LPIPS σ=0.3 only
- Fine-grained residuals blend into texture — best 72-bit visual quality

**For 72-bit with highest reliability:** b7238s20p03d03
- 72 channel bits, LDPC(72,38) = 38 payload bits, 274B IDs
- Both blur σ=0.3
- 90% single-message success (vs 82% for d00) — less batch search needed
- Slightly more visible residuals

All three use trained strength s=0.020, Sobel mask, and batch ID selection at inference.
