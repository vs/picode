# PicoComposite

PicoComposite is a multi-tier steganography model that supports 4 quality/capacity levels in a single trained model. Based on PicoTrust v14/v15 techniques (512→512, exponential annealing, content-adaptive encoding via blurred LPIPS/GAN), with split decoder/perceptual blur per tier.

## Tiers

| Tier | Name | Channel bits | LDPC config | Data bits | IDs | Short link | Strength | Dec σ | Perc σ |
|------|------|-------------|-------------|-----------|-----|------------|----------|-------|--------|
| 0 | **UHQ** | 30 | LDPC(30,17) | 17 | 131K | 6 chars | 0.008 | 0.5 | 1.0 |
| 1 | **HQ** | 48 | LDPC(48,26) | 26 | 67M | 9 chars | 0.010 | 0.5 | 0.7 |
| 2 | **MQ** | 72 | LDPC(72,38) | 38 | 274B | 13 chars | 0.012 | 0.5 | 0.5 |
| 3 | **LQ** | 96 | LDPC(96,50) | 50 | 1.1Q | 17 chars | 0.014 | 0.5 | 0.5 |

All LDPC codes use d_v=3, d_c=6, rate ~0.5. LDPC decoding uses soft probabilities from decoder logits (sigmoid), which dramatically outperforms BCH hard decoding.

## Architecture

Based on PicoTrust v14 with tier conditioning:

```
Image (512×512) + Message (up to 96 bits, zero-padded) + Tier index
    ↓
  Encoder (U-Net + E_post + tier embedding + bottleneck projection)
    ↓
  Grayscale residual → softsign bound with per-tier strength
    ↓
  Encoded image = Original + residual
    ↓
  Distortions (training only)
    ↓
  Decoder: original + blur(residual, σ=decoder_blur_sigma)   [shared 0.5]
    ↓
  Perceptual: blur(encoded, σ=perceptual_blur_sigma)         [per-tier]
    ↓
  Decoder (CNN + compact STN + tier classifier + tier-conditioned bit head)
    ↓
  Tier logits (4-class) + bit logits (96 bits, truncated to tier's count)
```

### Encoder conditioning

The encoder receives the tier index through three pathways:
1. **Message pathway**: tier embedding (32-dim) concatenated with message before projection
2. **Feature pathway**: tier embedding projected to 256-dim, added to U-Net bottleneck features
3. **Strength pathway**: per-sample softsign with `tier_strengths[tier_idx]`

### Decoder auto-detection

At inference, the decoder classifies the tier from the image alone (no tier input needed), then uses the detected tier's embedding to condition bit extraction. The bit output is truncated to the detected tier's bit count.

### Split blur design

Each tier has two blur sigmas — learned from v14-v20 experiments:

- **`decoder_blur_sigma`** (shared 0.5 for all tiers): Applied to decoder input as `original + blur(residual, σ=0.5)`. Shared across tiers so one decoder works on mobile without per-tier blur logic at inference.
- **`perceptual_blur_sigma`** (per-tier): Applied to LPIPS/GAN losses as `blur(encoded, σ)`. Controls content-adaptivity during training — higher σ = stronger LPIPS guidance = cleaner smooth regions.
- **L2/FFL losses**: computed on real encoded image — pixel-level quality on unblurred output.

Higher tiers (UHQ, fewer bits) get higher perceptual σ (1.0), maximizing content-adaptivity with the most headroom. Lower tiers (MQ/LQ, more bits) use lower perceptual σ (0.5), matching decoder blur.

**Key constraint from v18/v19**: σ > 1.0 for LPIPS creates diagonal artifacts — all perceptual sigmas capped at 1.0.

## Training

### Key parameters (v1)

```yaml
model:
  type: picomposite
  encoder_size: 512
  decoder_size: 512

training:
  num_bits: 96            # MAX_BITS — all tiers pad to this
  num_steps: 200000
  lr: 0.0001
  residual_strength: 1.0
  residual_strength_anneal_target: 1.0   # Per-tier targets in tiers.py
  residual_strength_anneal_start: 10000
  residual_strength_anneal_steps: 120000
  anneal_schedule: exponential
  phase2_step: 60000
  phase2_decoder_lr_scale: 0.1

loss:
  message: { scale: 5.0 }
  tier_classifier: { scale: 1.0 }
  l2: { scale: 1.5, ramp_steps: 50000 }
  lpips: { scale: 1.5, ramp_steps: 50000 }
  ffl: { scale: 1.0, ramp_steps: 50000, delay_steps: 0 }
  message_loss_type: mse
  gan_config: { enabled: true, discriminator_lr: 0.000015, g_loss_scale: 1.5 }
```

### Strength annealing

All tiers start at `residual_strength: 1.0` and anneal exponentially to their per-tier targets over 120k steps (starting at step 10k). UHQ anneals to 0.008, LQ to 0.014 — strength and perceptual blur both differentiate tiers.

### Training schedule

| Phase | Steps | What happens |
|-------|-------|-------------|
| Bootstrap | 0–500 | Message + tier classifier loss only (warmup) |
| Image loss ramp | 500–10k | Image losses ramp in (no_im_loss_steps) |
| Annealing | 10k–130k | Strength anneals from 1.0 → per-tier targets (exponential) |
| Phase 2 | 60k+ | Decoder LR reduced 10× for fine-tuning |
| Fine-tuning | 130k–200k | Training at final strengths, all losses active |

### Masked message loss

Each sample's message is padded to 96 bits. A binary mask tracks which positions are active for the sample's tier. The message loss is `(loss_per_bit * mask).sum() / mask.sum()` — only active bits contribute.

## ECC Integration

LDPC is applied at the application layer, not during training. The model encodes/decodes raw channel bits; LDPC encoding/decoding happens on-device (mobile app).

**Encode flow:**
```
API assigns random data-bit ID → LDPC encode → channel bits → encoder → image
```

**Decode flow:**
```
Image → decoder → channel bit logits → sigmoid → soft LDPC decode → data-bit ID → API fetch
```

LDPC soft decoding exploits confidence from decoder logits. Bits the decoder is uncertain about get less weight in belief propagation, yielding ~20% better message recovery than BCH hard decoding at comparable code rates.

### Security

With random ID assignment, enumeration resistance comes from the namespace size and API rate limiting:
- UHQ (131K IDs): suitable for premium/limited use, rate limiting essential
- HQ (67M IDs): practical for v1 product
- MQ (274B IDs): infeasible to enumerate at rate-limited API speeds
- LQ (1.1Q IDs): effectively infinite namespace

## Config and Code

- **Tier definitions**: `picode-model/picode/models/picomposite/tiers.py`
- **Encoder**: `picode-model/picode/models/picomposite/encoder.py`
- **Decoder**: `picode-model/picode/models/picomposite/decoder.py`
- **Training**: `picode-model/picode/training/trainer.py` (`_train_step_picomposite`)
- **Config**: `picode-model/configs/picomposite_v1.yaml`
- **Tests**: `picode-model/picode/tests/models/picomposite/`
