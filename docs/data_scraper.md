# Real-World Capture Dataset: Collection & Training Strategy

> **Note:** All file paths in this document are relative to `picode-model/` unless otherwise specified.

## Executive Summary

This document outlines a strategy for building a **paired dataset of (original image, real-world capture)** where captures are photographs of images displayed on screens or printed on paper. Unlike synthetic distortions, real captures exhibit complex, correlated artifacts that are difficult to simulate. This dataset enables:

1. **Encoder/decoder improvement** - Train models that survive actual camera capture, not just synthetic distortions
2. **Fast detection training** - Train FastDetector on realistic examples with natural perspective, lighting, and context

**Key insight:** Current training uses synthetic distortions (`picode/distortions/`) which simulate individual effects in isolation. Real captures combine multiple effects (moiré + perspective + uneven lighting + focus blur) in ways that synthetic pipelines cannot replicate.

---

## 1. Dataset Structure

### 1.1 Directory Layout

```
dataset/
├── pairs/
│   ├── 00001/
│   │   ├── original.png        # Source image (clean)
│   │   ├── capture.jpg         # Real-world photograph
│   │   ├── corners.json        # Detected quadrilateral corners
│   │   └── metadata.json       # Capture conditions
│   ├── 00002/
│   ...
├── by_type/
│   ├── screen/                 # Symlinks to screen captures
│   └── print/                  # Symlinks to print captures
├── splits/
│   ├── train.txt               # Training pair IDs
│   ├── val.txt                 # Validation pair IDs
│   └── test.txt                # Test pair IDs
└── index.csv                   # Global metadata index
```

### 1.2 Metadata Schema

```json
{
  "pair_id": "00001",
  "source": "flickr",
  "source_url": "https://...",
  "original_url": "https://...",
  "capture_type": "screen",
  "capture_device": "unknown",
  "display_device": "monitor",
  "resolution_original": [1920, 1080],
  "resolution_capture": [4032, 3024],
  "estimated_distance": null,
  "lighting_condition": "indoor",
  "quality_score": 0.85,
  "corner_confidence": 0.92,
  "collection_date": "2026-03-04",
  "license": "cc-by-2.0"
}
```

### 1.3 What Real Captures Provide

| Artifact Type | Screen Capture | Print Capture | Synthetic Available? |
|---------------|----------------|---------------|---------------------|
| **Moiré patterns** | Strong (pixel grid interference) | Weak | Partial |
| **Halftone/dithering** | None | Strong (CMYK dots) | No |
| **Perspective distortion** | Yes | Yes | Yes |
| **Lighting gradients** | Reflections, glare | Shadows, uneven illumination | Partial |
| **Color shift** | Gamut, white balance | Ink/paper color response | Partial |
| **Focus blur** | Depth of field | Depth of field | Yes |
| **Motion blur** | Camera shake | Camera shake | Yes |
| **Refresh artifacts** | Banding, flicker (video) | None | No |
| **Paper texture** | None | Visible in macro | No |
| **Bezel/edge context** | Screen frame visible | Paper edges, background | No |

---

## 2. Collection Strategies

### 2.1 Primary Methods

#### Method 1: Reverse Image Search Pipeline

Scrape images containing visible screens or prints, then find the original:

```
1. Search for "photo of monitor", "screen in photo", "photographed print"
2. Download candidate images
3. Detect screen/print region using object detection or edge detection
4. Extract and rectify the displayed content
5. Run reverse image search (Google, TinEye, Yandex)
6. Filter matches by perceptual hash similarity (pHash > 0.9)
7. Download matched original
8. Verify pair quality
```

**Pros:** Large-scale, diverse real-world conditions
**Cons:** Noisy matches, licensing complexity, compute-intensive

#### Method 2: Social Media Comparison Posts

Target communities where creators share original vs. captured comparisons:

- Photography forums: "monitor vs print" calibration discussions
- Art communities: "how my digital art looks printed"
- Behind-the-scenes posts: work displayed at exhibitions
- Unboxing posts: print products showing the delivered item

**Search queries:**
- `"original vs print" photo`
- `"on my monitor" vs "printed"`
- `"photo of my photo"`
- `"exhibition shot" artwork`

**Pros:** High-quality pairs, often both images in same post
**Cons:** Smaller scale, manual curation needed

### 2.2 Additional Sourcing Ideas

#### Idea 3: Museum & Gallery Exhibition Photos

People photograph artwork at museums and galleries. The originals are often available in:
- Museum digital collections (Met, MoMA, Rijksmuseum open access)
- Artist portfolios and websites
- Wikimedia Commons

**Pipeline:**
```
1. Scrape Flickr/Instagram for museum tags + artwork visible
2. Use artwork recognition or reverse search to identify piece
3. Find high-res original in museum API
```

#### Idea 4: Conference & Presentation Slides

Tech conferences generate thousands of photos showing slides on screens:

- Twitter/X posts from conferences (search: #conference2026 + photo)
- Official conference photo galleries
- Match to slides on SlideShare, SpeakerDeck, GitHub repos

**Advantage:** Slides often have simple, high-contrast content good for initial training.

#### Idea 5: Product Listings with Screen Displays

E-commerce photos where sellers show devices displaying images:

- Phone/tablet listings showing sample wallpapers
- TV listings showing demo content
- Digital frame listings with sample photos

**Sources:** eBay, Amazon marketplace, Craigslist

#### Idea 6: Movie/TV Behind-the-Scenes

On-set photos showing monitors with footage:

- Match to released frames from final content
- Film databases (IMDB, TMDB) for frame matching

**Challenge:** Frame-accurate matching required.

#### Idea 7: Digital Signage & Billboards

Photos of public displays, advertisements:

- Transit ads photographed by commuters
- Store window displays
- Stadium screens during events

**Original sources:** Brand asset libraries, ad archives, agency portfolios

#### Idea 8: Photo Frame Product Reviews

Reviews of digital photo frames include:
- Manufacturer sample images (often findable)
- Reviewer's own photos (if they share both)

**Sources:** YouTube reviews (frame grabs), Amazon reviews with photos

#### Idea 9: Art Print Marketplaces

Sites like Society6, Redbubble, Etsy:
- Product mockups show the original art
- Customer review photos show received prints
- Both derived from same source

**Advantage:** Clear provenance, same original guaranteed.

#### Idea 10: Academic & Research Datasets

Existing datasets may contain usable pairs:

| Dataset | Original Use | Pair Type |
|---------|-------------|-----------|
| **DocUNet** | Document dewarping | Document photos |
| **SIQAD** | Screen image quality | Screen captures |
| **CURE-TSR** | Traffic sign recognition | Real-world sign photos |
| **DIML** | Display image matching | Screen photo pairs |
| **Copydays** | Copy detection | Various transformations |

### 2.3 Controlled Lab Collection (Supplementary)

For gaps in scraped data, systematic capture:

```
Equipment:
- 2-3 smartphone models (iPhone, Pixel, Samsung)
- 1-2 monitors (IPS, OLED)
- Printer (inkjet, laser)
- Tripod with angle adjustment
- Controlled lighting rig

Capture matrix:
- 5 angles: 0°, 15°, 30°, 45°, 60°
- 3 distances: 0.3m, 0.5m, 1.0m
- 4 lighting: bright, dim, mixed, backlit
- 2 focus: sharp, slight blur
```

---

## 3. Training Usage: Encoder/Decoder

### 3.1 Learned Capture Simulation

Train a neural network to replicate real capture transformations:

```python
class CaptureSimulator(nn.Module):
    """Learns real-world capture transformation from paired data."""

    def __init__(self, capture_type: str = "both"):
        super().__init__()
        self.capture_type = capture_type

        # Encoder-decoder architecture
        self.encoder = nn.Sequential(
            # Downsampling path
            ConvBlock(3, 64, stride=2),   # -> H/2
            ConvBlock(64, 128, stride=2), # -> H/4
            ConvBlock(128, 256, stride=2),# -> H/8
        )

        # Capture-specific effects
        self.moire_gen = MoireGenerator()      # Screen-specific
        self.halftone_gen = HalftoneGenerator() # Print-specific

        # Shared decoder
        self.decoder = nn.Sequential(
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, image: Tensor, capture_type: str = "screen") -> Tensor:
        features = self.encoder(image)

        if capture_type == "screen":
            features = self.moire_gen(features)
        else:
            features = self.halftone_gen(features)

        return torch.sigmoid(self.decoder(features))
```

**Training objective:**
```python
def train_capture_simulator(model, original, capture):
    simulated = model(original)

    loss = (
        F.l1_loss(simulated, capture) * 1.0 +
        lpips_loss(simulated, capture) * 0.5 +
        adversarial_loss(simulated, capture) * 0.1  # GAN for realism
    )
    return loss
```

**Usage in encoder training:**
```python
# Replace synthetic distortions with learned simulator
encoded = encoder(image, message)
captured = capture_simulator(encoded)  # Differentiable!
decoded = decoder(captured)
message_loss = F.binary_cross_entropy_with_logits(decoded, message)
```

### 3.2 Cycle-Consistency Training

Learn bidirectional mapping between clean and captured domains:

```
Forward:  Original → Captured
Backward: Captured → Original (restoration)
```

This provides:
1. Unlimited synthetic captures from any image
2. Understanding of what information survives round-trip
3. Regularization for the capture simulator

```python
class CaptureRestorer(nn.Module):
    """Restores clean image from capture (inverse of CaptureSimulator)."""
    # Similar architecture, trained jointly
    pass

def cycle_consistency_loss(original, simulator, restorer):
    captured = simulator(original)
    restored = restorer(captured)
    return F.l1_loss(restored, original)
```

### 3.3 Direct Fine-tuning Pipeline

For highest quality, create watermarked captures directly:

```
1. Encode: original + message → encoded_image
2. Display: Show encoded_image on screen/print it
3. Capture: Photograph with various devices/conditions
4. Pair: (capture, message) for decoder fine-tuning
```

**Decoder fine-tuning:**
```python
def finetune_decoder(decoder, real_captures, messages):
    """Fine-tune on real captured watermarks."""
    for capture, message in zip(real_captures, messages):
        # May need to rectify first
        rectified = rectify_to_400x400(capture, corners)
        logits = decoder(rectified)
        loss = F.binary_cross_entropy_with_logits(logits, message)
        loss.backward()
```

### 3.4 Domain Randomization Enhancement

Use dataset statistics to improve synthetic augmentations:

```python
class RealismMatchedAugmentation:
    """Augmentations calibrated to match real capture statistics."""

    def __init__(self, dataset_stats: dict):
        # Extract distributions from real pairs
        self.blur_sigma_dist = dataset_stats['blur_sigma']
        self.noise_std_dist = dataset_stats['noise_std']
        self.perspective_dist = dataset_stats['perspective_strength']
        self.color_shift_dist = dataset_stats['color_shift']

    def __call__(self, image: Tensor) -> Tensor:
        # Sample from real distributions, not uniform
        sigma = np.random.choice(self.blur_sigma_dist)
        noise = np.random.choice(self.noise_std_dist)
        # Apply calibrated augmentations
        ...
```

---

## 4. Training Usage: Fast Detection

### 4.1 Real Corner Ground Truth

Extract corner labels from paired data using feature matching:

```python
def extract_corners_from_pair(original: np.ndarray, capture: np.ndarray) -> Quadrilateral:
    """Find where original appears in capture."""

    # Detect keypoints
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(original, None)
    kp2, desc2 = sift.detectAndCompute(capture, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Find homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Transform corners
    h, w = original.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)

    return Quadrilateral.from_numpy(transformed.reshape(4, 2))
```

**Advantage over synthetic:** Real perspective distributions, natural camera angles, realistic corner positions.

### 4.2 Hard Negative Mining

Create challenging negatives using captured non-watermarked images:

```python
class DetectionDatasetWithRealNegatives(Dataset):
    """Detection dataset mixing synthetic positives with real negatives."""

    def __init__(
        self,
        watermark_encoder: nn.Module,
        real_capture_pairs: list,  # (original, capture) pairs
        positive_ratio: float = 0.5,
    ):
        self.encoder = watermark_encoder
        self.real_pairs = real_capture_pairs

    def _generate_positive(self, idx: int) -> dict:
        """Watermarked image + real capture simulation."""
        original, capture = self.real_pairs[idx]

        # Encode watermark into original
        message = torch.randint(0, 2, (1, 100)).float()
        watermarked = self.encoder(original, message)

        # Apply learned capture simulation (trained on this dataset)
        simulated_capture = self.capture_simulator(watermarked)
        corners = extract_corners_from_pair(original, capture)

        return {
            "image": simulated_capture,
            "is_watermark": 1.0,
            "corners": corners.to_tensor(),
        }

    def _generate_hard_negative(self, idx: int) -> dict:
        """Real capture of NON-watermarked image."""
        original, capture = self.real_pairs[idx]

        # Use real capture directly (no watermark)
        # Detector must learn: "photographed screen" ≠ "watermarked"
        return {
            "image": capture,
            "is_watermark": 0.0,
            "corners": torch.zeros(8),
        }
```

**Why this matters:** Prevents detector from learning shortcuts like "moiré pattern = watermark" or "perspective distortion = watermark."

### 4.3 Domain Adaptation

Bridge the gap between synthetic training and real deployment:

```python
class DomainAdaptiveDetector(nn.Module):
    """FastDetector with domain adaptation for real captures."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.domain_classifier = GradientReversalLayer()  # Adversarial

    def forward(self, x: Tensor, domain_adapt: bool = False):
        features = self.backbone(x)

        detection_output = self.detection_head(features)

        if domain_adapt:
            # Adversarial: make synthetic features match real
            reversed_features = self.domain_classifier(features)
            domain_pred = self.domain_head(reversed_features)
            return detection_output, domain_pred

        return detection_output
```

**Training procedure:**
```
Phase 1: Train on synthetic data (unlimited)
Phase 2: Fine-tune with domain adaptation
         - Detection loss on synthetic (with labels)
         - Domain confusion loss on real (no labels needed)
Phase 3: Fine-tune on real labeled data (limited)
```

### 4.4 Augmentation Policy Learning

Learn augmentation parameters that make synthetic data match real:

```python
def learn_augmentation_policy(synthetic_data, real_captures):
    """AutoAugment-style policy learning for capture simulation."""

    # Extract statistics from real captures
    real_stats = {
        'blur_distribution': measure_blur(real_captures),
        'noise_distribution': measure_noise(real_captures),
        'color_distribution': measure_color_shift(real_captures),
        'perspective_distribution': measure_perspective(real_captures),
    }

    # Optimize augmentation parameters to match
    policy = AugmentationPolicy()
    optimizer = optim.Adam(policy.parameters())

    for batch in synthetic_data:
        augmented = policy(batch)

        # Minimize distribution distance
        loss = (
            kl_divergence(measure_blur(augmented), real_stats['blur']) +
            kl_divergence(measure_noise(augmented), real_stats['noise']) +
            # ... other statistics
        )
        loss.backward()
        optimizer.step()

    return policy
```

---

## 5. Data Processing Pipeline

### 5.1 Pair Validation

Not all scraped pairs are usable. Validation pipeline:

```python
class PairValidator:
    """Validates and scores scraped image pairs."""

    def validate(self, original: np.ndarray, capture: np.ndarray) -> dict:
        # 1. Size check
        if min(original.shape[:2]) < 256:
            return {"valid": False, "reason": "original too small"}

        # 2. Find homography
        corners, confidence = self.find_corners(original, capture)
        if confidence < 0.7:
            return {"valid": False, "reason": "poor feature matching"}

        # 3. Check coverage (original should fill significant portion)
        coverage = self.compute_coverage(corners, capture.shape)
        if coverage < 0.1:
            return {"valid": False, "reason": "original too small in capture"}

        # 4. Perceptual similarity check
        rectified = self.rectify(capture, corners)
        similarity = self.compute_similarity(original, rectified)
        if similarity < 0.6:
            return {"valid": False, "reason": "poor content match"}

        return {
            "valid": True,
            "corners": corners,
            "corner_confidence": confidence,
            "coverage": coverage,
            "similarity": similarity,
            "quality_score": (confidence + coverage + similarity) / 3,
        }
```

### 5.2 Corner Refinement

Feature matching gives approximate corners. Refinement improves accuracy:

```python
def refine_corners(capture: np.ndarray, initial_corners: np.ndarray) -> np.ndarray:
    """Refine corner positions using edge detection."""

    gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    refined = []
    for corner in initial_corners:
        # Search in local window for strongest edge intersection
        window = 20
        x, y = int(corner[0]), int(corner[1])
        local = edges[y-window:y+window, x-window:x+window]

        # Find corner in edge image
        corners_harris = cv2.cornerHarris(local.astype(np.float32), 2, 3, 0.04)
        max_loc = np.unravel_index(corners_harris.argmax(), corners_harris.shape)

        refined.append([x - window + max_loc[1], y - window + max_loc[0]])

    return np.array(refined, dtype=np.float32)
```

### 5.3 Capture Type Classification

Automatically classify screen vs. print:

```python
class CaptureTypeClassifier:
    """Classify capture as screen or print based on artifacts."""

    def classify(self, capture: np.ndarray) -> str:
        features = {
            'moire_strength': self.detect_moire(capture),
            'halftone_strength': self.detect_halftone(capture),
            'edge_sharpness': self.measure_edges(capture),
            'color_gamut': self.analyze_gamut(capture),
        }

        # Screen: strong moiré, sharp edges, wide gamut
        # Print: halftone pattern, softer edges, limited gamut

        screen_score = features['moire_strength'] + features['edge_sharpness']
        print_score = features['halftone_strength'] + (1 - features['color_gamut'])

        return "screen" if screen_score > print_score else "print"
```

---

## 6. Implementation Roadmap

### Phase 1: Data Collection Infrastructure

1. **Scraper framework** - Modular scrapers for different sources
2. **Reverse image search integration** - API wrappers for Google, TinEye
3. **Pair validator** - Feature matching, quality scoring
4. **Storage pipeline** - Download, organize, deduplicate
5. **Metadata tracking** - Source URLs, licenses, quality scores

### Phase 2: Initial Dataset Build

6. **Run scrapers** - Target 10K pairs initially
7. **Manual validation** - Review random sample, tune thresholds
8. **Corner ground truth** - Extract and refine corners
9. **Type classification** - Label screen vs. print
10. **Train/val/test split** - Stratified by source, type, quality

### Phase 3: Model Training Integration

11. **CaptureSimulator** - Train on paired dataset
12. **Augmentation calibration** - Match synthetic to real statistics
13. **Decoder fine-tuning** - Test on real captures
14. **Detection training** - Hard negative mining with real pairs

### Phase 4: Controlled Lab Supplement

15. **Capture rig setup** - Devices, lighting, automation
16. **Watermarked image captures** - For decoder ground truth
17. **Systematic variation** - Angle, distance, lighting matrix
18. **Integration** - Merge with scraped data

### Phase 5: Continuous Collection

19. **Automated pipelines** - Scheduled scraping runs
20. **Quality monitoring** - Track dataset statistics over time
21. **Active learning** - Prioritize captures where model fails
22. **Community contribution** - App for user-submitted pairs

---

## 7. Expected Outcomes

### 7.1 Encoder/Decoder Improvements

| Metric | Current (Synthetic Only) | Target (With Real Data) |
|--------|-------------------------|------------------------|
| Bit accuracy (real screen capture) | ~70-80% | >90% |
| Bit accuracy (real print capture) | ~60-75% | >85% |
| Bit accuracy (synthetic distortions) | ~95% | >95% (maintain) |

### 7.2 FastDetector Improvements

| Metric | Synthetic Training Only | With Real Capture Data |
|--------|------------------------|----------------------|
| Precision on real captures | ~80% | >95% |
| Recall on real captures | ~70% | >90% |
| False positive rate (real negatives) | ~15% | <5% |
| Corner IoU on real captures | ~0.7 | >0.85 |

### 7.3 Dataset Scale Targets

| Phase | Timeline | Target Pairs | Sources |
|-------|----------|--------------|---------|
| Initial | Phase 1-2 | 10,000 | Web scraping |
| Expanded | Phase 3-4 | 50,000 | Scraping + lab |
| Production | Phase 5 | 100,000+ | Continuous + community |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Low match rate in reverse search** | Slow collection | Multiple search engines, relaxed thresholds initially |
| **Poor corner accuracy** | Bad training signal | Refinement pipeline, manual validation sample |
| **Copyright issues** | Legal risk | Track licenses, prefer CC/open content, fair use for research |
| **Domain shift** | Real data doesn't help | Start with domain adaptation, gradual mixing |
| **Capture simulator overfits** | Poor generalization | Diverse training data, regularization, held-out test set |

---

## 9. References

### Related Project Documents
- [Detection Improvements](./detection_improvements.md) - FastDetector architecture and training
- [Model Improvements](./model_improvements.md) - Encoder/decoder quality improvements
- [Mobile Deployment](./mobile_deployment.md) - Constraints for mobile models

### External Resources
- [DIML Dataset](https://github.com/AIM3-RUC/DIML) - Display image matching dataset
- [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) - Document unwarping dataset
- [TinEye API](https://tineye.com/api) - Reverse image search
- [Google Vision API](https://cloud.google.com/vision) - Image matching and OCR

---

*Document created: 2026-03-04*
*Status: Design proposal pending implementation*
