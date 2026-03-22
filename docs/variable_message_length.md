# Variable Message Length Support

This document explores approaches for implementing a single adaptive model that adjusts visual artifacts based on message length, rather than maintaining multiple specialized models.

## Problem Statement

Users have different needs:
- **High-quality mode**: Minimal visual artifacts, shorter messages (e.g., short URLs)
- **High-capacity mode**: More artifacts acceptable, longer messages (e.g., full metadata)

Rather than training and deploying multiple models, we explore whether a single model can adapt its artifact level based on message length.

## Why Current Models Have Fixed Artifacts

In the current StegaStamp/Picode architecture, the encoder always:

1. Expands the message to a spatial map via `Linear(num_bits, 7500)` → reshape to `(3, 50, 50)` → upsample to `(400, 400)`
2. Produces a residual that gets added to the original image
3. The residual magnitude is roughly constant regardless of message content

Even padding a 50-bit message to 100 bits with zeros doesn't reduce artifacts - the model still "works" to encode all 100 bits without knowing that half are meaningless padding.

## Proposed Approaches

### Option A: Capacity-Conditioned Encoder

**Core idea:** Add a "capacity signal" that tells the encoder how many bits are meaningful, allowing it to scale the residual accordingly.

**Architecture changes:**

```python
class AdaptiveEncoder(nn.Module):
    def forward(self, image: Tensor, message: Tensor, capacity_ratio: float) -> Tensor:
        """
        Args:
            image: (B, 3, H, W) input image
            message: (B, max_bits) zero-padded message
            capacity_ratio: float in (0, 1] indicating fraction of bits used
        """
        # Tile capacity_ratio spatially and concatenate with inputs
        capacity_map = torch.full((B, 1, H, W), capacity_ratio)
        # ... rest of encoder with capacity awareness
```

**Training approach:**

1. Randomly sample message lengths during training (e.g., uniform from 20 to 100 bits)
2. Pad shorter messages with zeros to max length
3. Include `capacity_ratio` as an additional input channel
4. Modify loss function:
   ```python
   loss = message_loss + l2_loss + lpips_loss + alpha * (1 - capacity_ratio) * residual_penalty
   ```
   This penalizes large residuals more heavily when capacity is low.

**Decoder:** Unchanged - always outputs `max_bits`. The message itself encodes its length in the first few bits.

**Pros:**
- Single model for all capacity levels
- Intuitive "quality slider" at encoding time
- Decoder remains simple

**Cons:**
- Requires retraining from scratch
- May not achieve the same quality as a model specialized for a single capacity
- Capacity ratio must be chosen at encode time

---

### Option B: Progressive Bit Encoding

**Core idea:** Encode bits in priority order. Early bits are encoded most robustly with minimal artifacts; later bits add incremental distortion.

**Conceptual model:**

```
Bits 1-30:   Encoded with minimal residual (high priority, survives heavy distortion)
Bits 31-60:  Adds moderate residual (medium priority)
Bits 61-100: Adds more residual (low priority, full capacity mode)
```

**Architecture approach:**

Use a multi-scale or hierarchical encoder where each "level" adds more bits:

```python
class ProgressiveEncoder(nn.Module):
    def __init__(self):
        self.encoder_level1 = EncoderBlock(bits=30)   # Coarse, robust
        self.encoder_level2 = EncoderBlock(bits=30)   # Medium
        self.encoder_level3 = EncoderBlock(bits=40)   # Fine, fragile

    def forward(self, image, message, levels=3):
        residual = self.encoder_level1(image, message[:, :30])
        if levels >= 2:
            residual += self.encoder_level2(image, message[:, 30:60])
        if levels >= 3:
            residual += self.encoder_level3(image, message[:, 60:100])
        return image + residual
```

**Training approach:**

- Train with variable `levels` parameter
- Weight early bits more heavily in message loss
- Each level's residual is trained to be minimal while achieving its bit accuracy target

**Decoder:** Could be similarly progressive, outputting confidence per-bit tier. This enables graceful degradation - if the image is heavily distorted, level 3 bits may be lost but levels 1-2 survive.

**Pros:**
- Graceful degradation under distortion
- Natural quality/capacity trade-off
- Could enable "enhancement" - start with low-capacity, add more later (if original is preserved)

**Cons:**
- More complex architecture
- Decoder complexity increases
- Levels are discrete, not continuous

---

### Option C: Learned Rate-Distortion Trade-off

**Core idea:** Borrow from neural image compression research. Train with a rate-distortion objective where a hyperparameter controls the quality/capacity trade-off.

**Loss function:**

```python
loss = lambda_quality * distortion_loss(original, encoded) + message_loss
```

Where `lambda_quality` varies:
- High `lambda_quality` → prioritize image quality → fewer artifacts → may sacrifice some bit accuracy
- Low `lambda_quality` → prioritize message accuracy → more artifacts acceptable

**Training approach:**

Train multiple times with different `lambda_quality` values, or use a conditional approach where `lambda_quality` is an input to the network (similar to Option A).

**Challenge:** In practice, different `lambda_quality` values often require different model weights to achieve optimal results. This risks recreating the multi-model problem, unless the conditioning approach works well.

**Pros:**
- Well-studied in compression literature
- Continuous quality control

**Cons:**
- May require multiple model weights (defeating the purpose)
- Message accuracy may degrade unpredictably at high quality settings

---

### Option D: Attention-Based Sparse Encoding

**Core idea:** Use attention mechanisms to selectively modify only the image regions that can best hide information, and use fewer regions for shorter messages.

**How it works:**

1. A learned attention network identifies "hiding-friendly" regions (textures, edges, complex areas)
2. For shorter messages, only the best regions are used
3. For longer messages, more regions are recruited

**Pros:**
- Artifacts concentrated in less-visible areas
- Natural scaling with message length

**Cons:**
- Significantly more complex architecture
- May create predictable patterns attackers could detect
- Region selection adds latency

---

## Decoder Considerations

Regardless of encoder approach, the decoder must handle variable-length messages. Options:

### Length in Header

Reserve the first N bits (e.g., 7 bits = 0-127 length) to encode message length:

```
| Length (7 bits) | Payload (up to 93 bits) | Padding |
```

**Decoder logic:**
1. Decode all bits
2. Read length from first 7 bits
3. Extract payload of that length
4. Ignore remaining bits

### Fixed Maximum with Convention

Always decode to `max_bits`. The message format itself indicates length:
- Null-terminated strings
- Length-prefixed binary data
- Self-delimiting encoding (like UTF-8)

### ECC-Based Detection

Use different ECC configurations for different message lengths. Try decoding with each; the one that succeeds indicates the length.

**Example:**
- 50-bit mode: BCH(127, 50) with strong error correction
- 100-bit mode: BCH(127, 100) with weaker error correction

---

## Recommended Approach

For Picode's use case (two quality levels: high-quality/short vs. lower-quality/long), **Option A (Capacity-Conditioned Encoder)** is recommended:

1. **Simplicity:** Single model, minimal architecture changes
2. **Flexibility:** Continuous capacity ratio, not discrete levels
3. **Decoder unchanged:** Mobile app complexity stays low
4. **Training feasibility:** Straightforward extension of current training

**Implementation outline:**

1. Modify `Encoder.forward()` to accept `capacity_ratio` parameter
2. Create capacity-aware spatial conditioning (tile ratio into channel)
3. Add residual magnitude penalty weighted by `(1 - capacity_ratio)` to loss
4. Train with randomly sampled message lengths
5. Encode message length in first 7 bits of payload

**Expected outcome:**
- Short messages (capacity_ratio ~0.3): Minimal visible artifacts
- Full messages (capacity_ratio ~1.0): Current artifact level

---

## Open Questions

1. **Quality floor:** What's the minimum artifact level achievable for very short messages? There's a fundamental limit to steganographic invisibility.

2. **Decoder robustness:** Does the decoder maintain accuracy across all capacity ratios, or does it need capacity-awareness too?

3. **Distortion interaction:** Do low-capacity encodings survive print-scan and JPEG better (since they're encoded more robustly), or worse (since the signal is weaker)?

4. **Perceptual vs. measured quality:** Should we optimize for PSNR/SSIM, or perceptual metrics like LPIPS? Users care about perceived quality.

---

## Next Steps

1. Prototype capacity-conditioned encoder with current architecture
2. Train on variable message lengths with residual penalty
3. Evaluate quality-capacity curve empirically
4. Compare against baseline (fixed 100-bit model with padding)
