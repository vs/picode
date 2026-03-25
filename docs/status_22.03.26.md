# Project Status - January 21, 2026

## Overview

The Picode monorepo contains 4 subprojects for steganography (encoding/decoding hidden messages in images). Main branch is clean with active development on picode_v2 model improvements.

---

## 1. picode-model/ (PyTorch Training Framework)

**Goal:** Train neural networks that encode/decode hidden messages in images, robust to real-world distortions (blur, noise, JPEG compression, geometric transforms).

**Progress:** ~90% complete

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| StegaStamp model | ✅ Complete | Original CVPR 2020 architecture |
| Picode model | ✅ Complete | Improved gradient flow (GroupNorm + LeakyReLU) |
| Picode v2 model | ✅ Complete | Mobile-optimized, ~310K decoder params |
| Dual-backend distortions | ✅ Complete | Native PyTorch + Kornia implementations |
| Training infrastructure | ✅ Complete | YAML config, curriculum learning, checkpointing |
| ECC (BCH, LDPC) | ✅ Complete | BCH(127,64) corrects up to 10 bit errors |
| Modal cloud training | ✅ Complete | GPU training on Modal infrastructure |
| Logging | ✅ Complete | Console + TensorBoard backends |

### Picode v2 (Recently Completed)

All 8 implementation tasks merged to main:

1. ✅ InvertedResidual block (MobileNetV2-style)
2. ✅ MobileDecoder (~310K params, under 500K target)
3. ✅ Encoder with learned message expansion + content-adaptive scaling
4. ✅ FocalFrequencyLoss for artifact reduction
5. ✅ PatchDiscriminator + WGAN-GP losses
6. ✅ Module integration with trainer
7. ✅ Integration tests
8. ✅ Training config (`configs/picode_v2.yaml`)

### Next Tasks

1. **Train picode_v2 with FFL/GAN losses** - Config ready, trainer needs alternating discriminator update loop
2. **Add ONNX export utilities** - For cross-platform deployment
3. **Add Core ML export** - Required for iOS integration
4. **Run robustness benchmarks** - Evaluate trained model against distortion sweep

---

## 2. picode-ios/ (iOS Application)

**Goal:** Mobile app for real-time steganography detection using device camera.

**Progress:** ~60% complete

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| SwiftUI UI | ✅ Complete | Modern declarative UI |
| UIKit camera integration | ✅ Complete | AVFoundation-based capture |
| MVVM architecture | ✅ Complete | Clean separation of concerns |
| Protocol-based decoder | ✅ Complete | Easy to swap implementations |
| Permission handling | ✅ Complete | Camera access |
| Core ML integration | 🚧 Blocked | Waiting for model export |
| Live scanning | 📋 Planned | Real-time detection |
| Photo library | 📋 Planned | Import from camera roll |
| Message history | 📋 Planned | Save decoded messages |

**Requirements:** iOS 16.0+, Xcode 15.0+

### Next Tasks

1. **Export trained model to Core ML** - Requires picode_v2 training completion
2. **Replace StubDecoder** - Integrate Core ML inference
3. **Implement live scanning** - Real-time camera detection
4. **Add photo library support** - Import and decode existing images

---

## 3. picode-scraper/ (Dataset Collection)

**Goal:** Collect original/processed image pairs from photography forums for training data. Images that have been uploaded, compressed, and re-downloaded provide natural distortion examples.

**Progress:** ~75% complete

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| PostgreSQL coordination | ✅ Complete | Row-level locking for distributed workers |
| Multi-worker architecture | ✅ Complete | Horizontal scaling |
| Source plugin system | ✅ Complete | Decorator-based registration |
| SIFT pair validation | ✅ Complete | Validates image pairs match |
| Image deduplication | ✅ Complete | SHA-256 + perceptual hash |
| S3 storage backend | ✅ Complete | Production storage |
| Local storage backend | ✅ Complete | Development/testing |
| Export pipeline | ✅ Complete | Train/val/test splits |
| Mock source plugin | ✅ Complete | For testing |
| DPReview plugin | ✅ Complete | Just finished |

### DPReview Scraper (Recently Completed)

All 10 implementation tasks complete:

1. ✅ `is_content_image` filter (excludes avatars, UI elements)
2. ✅ `extract_post_images` function
3. ✅ HTML test fixtures
4. ✅ `parse_search_results` parser
5. ✅ `parse_thread_page` parser
6. ✅ `get_pagination_urls` for multi-page threads
7. ✅ `DPReviewSource` class
8. ✅ Package registration
9. ✅ Integration tests
10. ✅ Full verification (~35 new tests)

### Next Tasks

1. **Deploy DPReview scraper to production** - Start collecting data
2. **Add additional forum sources** - Flickr, 500px, photography forums
3. **Build image verification UI** - Quality control for collected pairs
4. **Advanced filtering options** - Resolution, format, metadata filters

---

## 4. docs/ (Documentation)

**Goal:** Comprehensive technical documentation and design plans.

**Progress:** ~85% complete

### Technical Documentation (11 files)

- `distortions_implementation.md` - Dual-backend comparison
- `model_implementation.md` - Architecture details
- `gradient_flow_analysis.md` - Gradient behavior analysis
- `mobile_deployment.md` - iOS deployment considerations
- `model_improvements.md` - Enhancement proposals
- `detection_improvements.md` - Detection optimization
- `data_scraper.md` - Scraper architecture
- `stegastamp-decoder-architecture-analysis.md` - Decoder analysis
- `variable_message_length.md` - Variable-length encoding design
- `multimodel_support.md` - Multi-model exploration
- `variable_message_length_review.md` - Design review

### Design Documents (6 files)

- `2026-03-04-fast-detector-phase1.md` - Fast detector design
- `2026-03-04-picode-scraper-design.md` - Scraper architecture
- `2026-03-05-phase6-dpreview-scraper-design.md` - DPReview design
- `2026-03-05-phase6-dpreview-scraper.md` - DPReview implementation (DONE)
- `2026-03-22-picode-v2-design.md` - Picode v2 design
- `2026-03-22-picode-v2-implementation.md` - Picode v2 implementation (DONE)

### Next Tasks

1. **Add Core ML export guide** - Document model conversion process
2. **Document picode_v2 training results** - Metrics, robustness benchmarks
3. **Add iOS deployment playbook** - App Store submission guide

---

## Feature Branches

| Branch | Location | Purpose | Status |
|--------|----------|---------|--------|
| `feature/fast-detector` | `.worktrees/fast-detector` | Quick encoding detection | Integration tests done |
| `feature/picode-v2-improvements` | `.worktrees/picode-v2` | Training refinements | FFL support added |
| `blind-detection` | `.worktrees/blind-detection` | Experimental | Synced to main |

---

## Priority Recommendations

### High Priority

1. **Train picode_v2 model end-to-end** - Complete the training loop with FFL/GAN losses
2. **Export to Core ML** - Unblocks iOS development
3. **Deploy DPReview scraper** - Start collecting training data

### Medium Priority

1. **Complete fast-detector branch** - Merge detection network
2. **Add more forum source plugins** - Expand dataset collection
3. **iOS live scanning feature** - Real-time camera detection

### Low Priority

1. **Photo library support in iOS** - Import existing images
2. **Image verification UI** - Manual quality control
3. **TensorRT optimization** - Server-side inference speedup

---

## Recent Commits (Main Branch)

```
c77c2ca feat(picode_v2): add training config
45f6633 test(picode_v2): add integration tests
40e2408 feat(picode_v2): integrate with training infrastructure
c019e8c feat(picode_v2): add PatchDiscriminator and GAN losses
18ba118 feat(picode_v2): add FocalFrequencyLoss
254188c feat(picode_v2): add Encoder with artifact reduction
f97a2fd feat(picode_v2): add MobileDecoder
e4da6f7 feat(picode_v2): add InvertedResidual block
309a201 docs: add picode_v2 implementation plan
597cd12 fix(design): use Upsample+Conv instead of ConvTranspose2d
```

---

## Code Quality

- **Type hints:** Strict mypy compliance (Python 3.10+ syntax)
- **Style:** Ruff enforced (100 char line length)
- **Commits:** Conventional format (feat:, fix:, test:, docs:, refactor:, chore:)
- **Tests:** ~400+ tests in picode-model, ~150 tests in picode-scraper
- **Documentation:** Google-style docstrings
