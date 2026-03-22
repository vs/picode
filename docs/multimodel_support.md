# Multi-Model Support for Mobile Decoding

This document explores approaches for enabling a mobile app to decode messages from images encoded with any supported model, without prior knowledge of which model was used.

## Problem Statement

Picode may support multiple encoder models optimized for different use cases:

| Model Type | Visual Artifacts | Message Capacity | Best For |
|------------|------------------|------------------|----------|
| High-quality | Minimal | Shorter (50-60 bits) | Professional photos, art |
| High-capacity | Noticeable | Longer (100+ bits) | Metadata-heavy use cases |
| Text-optimized | Tuned for text regions | Variable | Images with text overlays |
| Photo-optimized | Tuned for natural images | Variable | Photographs |

The mobile app must decode any encoded image without knowing which model created it.

## Approaches

### Approach 1: Embedded Model ID (Header Bits)

**How it works:** Reserve the first N bits (e.g., 8 bits) of every message to encode a model identifier.

**Encoding:**
```
| Model ID (8 bits) | Payload (remaining bits) |
```

**Decoding logic:**
1. Decode header bits to identify model
2. Use correct decoder for full message extraction

**Analysis:**

| Aspect | Assessment |
|--------|------------|
| Payload capacity | Reduced by header size (8 bits) |
| Decode latency | Single pass (if header decodes correctly) |
| Implementation | Simple bit convention |
| Adding new models | Easy (just assign new ID) |

**Critical flaw - Chicken-and-egg problem:**

To read the model ID, you need to decode the header. But different models encode bits differently - a header encoded by Model A won't decode correctly with Model B's decoder.

This approach only works if:
- All models produce identical encodings for the header region (requires architectural constraints), OR
- A "universal header decoder" exists that works across all models (adds complexity)

**Verdict:** Not recommended as primary approach due to the chicken-and-egg problem.

---

### Approach 2: Try-All-Models with Early Exit

**How it works:** Run all available decoders on the image. Use confidence scores and ECC success to determine the correct result.

**Decoding logic:**
```
for decoder in available_decoders:
    result = decoder.decode(image)
    confidence = compute_confidence(result)
    ecc_success = try_ecc_decode(result)

    if ecc_success and confidence > threshold:
        return result  # Early exit

return best_result or NoWatermarkFound
```

**Analysis:**

| Aspect | Assessment |
|--------|------------|
| Payload capacity | Full (no bits reserved) |
| Decode latency | O(N) where N = number of models |
| Implementation | Straightforward |
| Adding new models | Easy (add to decoder list) |
| Robustness | Good (ECC validates correctness) |

**Performance characteristics:**

- CoreML decoder inference: ~50-100ms per model on modern iPhone
- With 2-3 models: 100-300ms total (acceptable for camera scanning)
- With 5+ models: May need optimization

**Optimizations:**
1. **Parallel execution:** Run decoders concurrently on GPU
2. **Early exit:** Stop when ECC succeeds with high confidence
3. **Prioritization:** Try most common model first
4. **Batch inference:** Some CoreML optimizations for multiple models

**False positive mitigation:**

A wrong decoder might produce random bits that accidentally pass ECC. Mitigations:
- Require confidence threshold (e.g., >0.8 average bit confidence)
- Use stronger ECC with lower false-positive rate
- Add checksum/magic bytes to message format

**Verdict:** Recommended as practical starting point. Simple, robust, acceptable performance for 2-3 models.

---

### Approach 3: Visual Sync Pattern / Model Signature

**How it works:** Embed a subtle visual pattern that identifies the model, separate from the message payload.

**Possible implementations:**

1. **Corner markers:** QR-code-style finder patterns with model-specific variations
2. **Frequency signature:** Embed model ID in specific frequency bands (DCT/DFT domain)
3. **Learned fingerprint:** Train models to embed a recognizable "signature" in the residual

**Decoding logic:**
1. Detect and analyze visual signature
2. Identify model from signature
3. Run appropriate decoder

**Analysis:**

| Aspect | Assessment |
|--------|------------|
| Payload capacity | Full (signature is separate) |
| Decode latency | Fast (signature detection + single decode) |
| Implementation | Complex (requires model retraining) |
| Adding new models | Hard (must design new signature, retrain) |
| Robustness | Unknown (signature must survive distortion) |

**Challenges:**
- Signature detection adds failure mode
- Visible patterns may increase artifacts
- Requires retraining all existing models
- Signature must survive JPEG, print-scan, screenshots

**Verdict:** Not recommended for initial implementation. Consider for future if try-all becomes too slow.

---

### Approach 4: Discriminator Network

**How it works:** Train a small, fast classifier that predicts which encoder was used (or "none").

**Architecture:**
```
Image → Discriminator CNN → [Model A, Model B, Model C, None]
```

**Decoding logic:**
1. Run discriminator (~10-20ms)
2. If "None" predicted, return NoWatermarkFound
3. Otherwise, run predicted decoder

**Training:**
- Generate synthetic dataset: encode images with each model
- Train discriminator to classify by model
- Include unencoded images for "None" class

**Analysis:**

| Aspect | Assessment |
|--------|------------|
| Payload capacity | Full |
| Decode latency | Fast (discriminator + single decode) |
| Implementation | Medium (requires training additional model) |
| Adding new models | Medium (retrain discriminator) |
| Robustness | Good (learned from data) |

**Advantages:**
- Handles "no watermark" case naturally
- Fast inference path
- Can be trained on distorted images for robustness

**Disadvantages:**
- Another model to maintain and deploy
- Must retrain when adding new encoder models
- Discriminator errors cascade to wrong decoder

**Verdict:** Recommended evolution path once multiple production models exist.

---

## Comparison Summary

| Approach | Capacity | Latency | Complexity | Extensibility | Recommended? |
|----------|----------|---------|------------|---------------|--------------|
| Header bits | Reduced | Fast* | Low | Easy | No (chicken-egg) |
| Try-all | Full | O(N) | Low | Easy | **Yes (start here)** |
| Visual signature | Full | Fast | High | Hard | No (too complex) |
| Discriminator | Full | Fast | Medium | Medium | **Yes (evolution)** |

*If chicken-and-egg problem is solved

---

## Recommended Implementation Path

### Phase 1: Try-All with Early Exit (Now)

Implement for 2-3 models:

```swift
protocol PicodeDecoder {
    func decode(image: UIImage) async throws -> DecodeResult
    var modelId: String { get }
}

class MultiModelDecoder {
    let decoders: [PicodeDecoder]

    func decode(image: UIImage) async throws -> DecodeResult {
        var bestResult: DecodeResult?
        var bestConfidence: Double = 0

        for decoder in decoders {
            do {
                let result = try await decoder.decode(image: image)

                // Early exit on high-confidence ECC success
                if result.eccSuccess && result.confidence > 0.9 {
                    return result
                }

                if result.confidence > bestConfidence {
                    bestConfidence = result.confidence
                    bestResult = result
                }
            } catch {
                continue  // Try next decoder
            }
        }

        guard let result = bestResult, result.confidence > 0.5 else {
            throw DecodeError.noWatermarkFound
        }
        return result
    }
}
```

### Phase 2: Add Discriminator (When N > 3)

When model count grows:

1. Train lightweight discriminator (MobileNet-scale)
2. Deploy alongside decoders
3. Use discriminator prediction to skip unnecessary decode attempts

```swift
class DiscriminatorGuidedDecoder {
    let discriminator: ModelDiscriminator
    let decoders: [String: PicodeDecoder]

    func decode(image: UIImage) async throws -> DecodeResult {
        let prediction = try await discriminator.predict(image: image)

        if prediction.modelId == "none" {
            throw DecodeError.noWatermarkFound
        }

        guard let decoder = decoders[prediction.modelId] else {
            throw DecodeError.modelError("Unknown model: \(prediction.modelId)")
        }

        return try await decoder.decode(image: image)
    }
}
```

### Phase 3: Consider Variable-Length Model (Future)

If the variable message length approach (see `variable_message_length.md`) proves viable, the multi-model problem may be simplified to a single adaptive model, eliminating the need for model detection entirely.

---

## Message Format Convention

Regardless of detection approach, standardize the message format:

```
| Version (4 bits) | Length (7 bits) | Payload (variable) | Checksum (8 bits) |
```

- **Version:** Message format version (allows future evolution)
- **Length:** Payload length in bytes
- **Payload:** Actual message content
- **Checksum:** CRC-8 for additional validation beyond ECC

This format works with any encoder model and provides defense-in-depth against false positives.

---

## iOS Implementation Notes

### CoreML Model Management

```swift
class ModelRegistry {
    private var loadedModels: [String: MLModel] = [:]

    func decoder(for modelId: String) async throws -> PicodeDecoder {
        if let model = loadedModels[modelId] {
            return CoreMLDecoder(model: model)
        }

        // Lazy load model
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let model = try await loadModel(modelId, configuration: config)
        loadedModels[modelId] = model
        return CoreMLDecoder(model: model)
    }
}
```

### Memory Considerations

- Each CoreML model: ~5-20MB in memory
- Keep frequently-used models loaded
- Use `MLModelConfiguration.computeUnits = .cpuAndGPU` for best performance
- Consider model quantization (float16) for smaller footprint

---

## Open Questions

1. **Model versioning:** How to handle model updates? Same model ID with improved weights, or new ID?

2. **Backwards compatibility:** How long to support old models? Deprecation strategy?

3. **A/B testing:** How to roll out new models gradually?

4. **Analytics:** What metrics to collect for model performance in the wild?
