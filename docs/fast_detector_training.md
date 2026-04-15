# FastDetector Training Guide

This guide explains how to train the FastDetector model for mobile watermark detection. FastDetector is a lightweight MobileNetV3-based model that detects watermarks in real-time on iOS/Android devices.

## Prerequisites

Before training FastDetector, you need:

1. **Trained encoder checkpoint** - FastDetector learns to detect artifacts from YOUR encoder
2. **Training images** - Same images used for encoder training work well
3. **Negative images** (optional but recommended) - Non-watermarked images for hard negative mining

### Using Your picode_v2 Checkpoint

If you trained picode_v2 today, locate your encoder checkpoint:

```bash
# Check your checkpoints directory
ls checkpoints/
# or
ls runs/*/checkpoints/
```

Your encoder checkpoint will be named something like `encoder.pt`, `best_encoder.pt`, or embedded in a combined checkpoint.

## Quick Start

### 1. Prepare Directory Structure

```bash
cd picode-model

# Create detection training directories
mkdir -p data/detection/positive  # Your training images (same as encoder training)
mkdir -p data/detection/negative  # Hard negatives (optional)
```

### 2. Link or Copy Training Images

```bash
# Option A: Symlink your existing training images
ln -s ../train data/detection/positive

# Option B: Copy images
cp -r data/train/* data/detection/positive/
```

### 3. Train FastDetector

```python
# train_fast_detector.py
import torch
from torch.utils.data import DataLoader, random_split

from picode.detection import FastDetectorModel
from picode.detection.training import DetectionDataset, DetectionLoss, DetectionTrainer
from picode.models.picode_v2 import Encoder  # or your model type


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load your trained encoder
    encoder = Encoder(num_bits=100)
    checkpoint = torch.load("checkpoints/your_encoder.pt", map_location=device)

    # Handle different checkpoint formats
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "model_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["model_state_dict"])
    else:
        encoder.load_state_dict(checkpoint)

    encoder.eval()
    encoder.to(device)
    print("Loaded encoder checkpoint")

    # 2. Create dataset
    dataset = DetectionDataset(
        image_dir="data/detection/positive",
        encoder=encoder,
        num_bits=100,
        positive_ratio=0.5,  # 50% watermarked, 50% clean
        input_size=320,
        perspective_strength=(0.0, 0.15),
    )
    print(f"Dataset size: {len(dataset)} images")

    # 3. Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 4. Create model and loss
    model = FastDetectorModel(input_size=320, pretrained_backbone=True)
    loss_fn = DetectionLoss(cls_weight=1.0, corner_weight=5.0, conf_weight=0.5)

    # 5. Create trainer and train
    trainer = DetectionTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-4,
        weight_decay=1e-4,
        device=device,
    )

    print("Starting training...")
    history = trainer.fit(
        num_epochs=50,
        save_dir="checkpoints/detection",
        save_every=10,
    )

    # 6. Save final model
    trainer.save_checkpoint("checkpoints/fast_detector_final.pt")
    print("Training complete!")

    # Print final metrics
    print(f"Final train loss: {history[-1]['train_loss']:.4f}")
    print(f"Final val loss: {history[-1]['val_loss']:.4f}")


if __name__ == "__main__":
    main()
```

Run training:

```bash
cd picode-model
source ../venv/bin/activate
python train_fast_detector.py
```

## Configuration Options

### Using YAML Config

You can also use the provided config file:

```bash
# Edit configs/detection_training.yaml to set your encoder checkpoint path
vim configs/detection_training.yaml
```

Key settings in `detection_training.yaml`:

```yaml
data:
  positive_dir: ./data/detection/positive
  negative_dir: ./data/detection/negative  # Optional
  encoder_checkpoint: ./checkpoints/encoder.pt  # YOUR checkpoint
  positive_ratio: 0.5
  batch_size: 16

training:
  num_epochs: 50
  lr: 0.0001

loss:
  cls_weight: 1.0      # Classification loss weight
  corner_weight: 5.0   # Corner regression weight (higher = better localization)
  conf_weight: 0.5     # Corner confidence weight
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 50 | Training epochs (50-100 recommended) |
| `lr` | 1e-4 | Learning rate |
| `batch_size` | 16 | Batch size (reduce if OOM) |
| `positive_ratio` | 0.5 | Ratio of watermarked samples |
| `perspective_strength` | (0, 0.15) | Perspective distortion range |
| `pretrained_backbone` | true | Use ImageNet pretrained MobileNetV3 |

## Hard Negative Mining (Recommended)

For better detection accuracy, add hard negative images that could fool a naive detector:

```bash
# Download or collect hard negatives
mkdir -p data/detection/negative

# Good sources for hard negatives:
# - JPEG-compressed images (has block artifacts)
# - Resized/upscaled images (has interpolation artifacts)
# - Screenshots (has sharp edges)
# - Instagram-filtered photos
# - Scanned documents
```

Then modify your dataset to use both:

```python
from picode.detection.training import HardNegativeDataset

dataset = HardNegativeDataset(
    positive_dir="data/detection/positive",
    negative_dirs=[
        "data/detection/negative/jpeg_artifacts",
        "data/detection/negative/screenshots",
        "data/detection/negative/filtered",
    ],
    encoder=encoder,
    negative_ratio_per_dir={
        "jpeg_artifacts": 0.15,
        "screenshots": 0.10,
        "filtered": 0.10,
    },
)
```

## Monitoring Training

### TensorBoard

```bash
# In another terminal
tensorboard --logdir runs/detection
```

### Expected Metrics

| Epoch | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 1-5 | 1.5-2.0 | 1.5-2.0 | Warmup phase |
| 10-20 | 0.5-1.0 | 0.5-1.0 | Rapid improvement |
| 30-50 | 0.2-0.5 | 0.3-0.6 | Fine-tuning |

### Signs of Good Training

- Classification accuracy > 95%
- Corner IoU > 0.85 on validation set
- Val loss tracks train loss (no overfitting)

### Signs of Problems

- Val loss increasing while train loss decreases → overfitting, add more data or augmentation
- Loss stuck at ~0.69 → model not learning, check encoder checkpoint
- NaN losses → reduce learning rate

## Evaluation

After training, evaluate on held-out test set:

```python
from picode.detection.training import DetectionEvaluator

evaluator = DetectionEvaluator(
    model=model,
    encoder=encoder,
    test_dir="data/test",
    device=device,
)

metrics = evaluator.evaluate()
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"Mean IoU: {metrics['mean_iou']:.3f}")

# Robustness sweep
robustness = evaluator.evaluate_robustness(
    jpeg_qualities=[30, 50, 70, 90],
    blur_sigmas=[0.5, 1.0, 2.0],
    noise_stds=[0.01, 0.02, 0.05],
)
```

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Precision | > 95% | Avoid false positives |
| Recall | > 90% | Catch most watermarks |
| Mean IoU | > 0.85 | Accurate localization |
| Recall @ JPEG Q40 | > 80% | Robust to compression |

## Export to Core ML (iOS)

Once training is complete, export for iOS:

```bash
# Using CLI
detect-export coreml checkpoints/fast_detector_final.pt -o FastDetector.mlpackage

# With quantization (smaller model)
detect-export coreml checkpoints/fast_detector_final.pt -o FastDetector.mlpackage --quantize fp16
```

Or via Python:

```python
from picode.detection import FastDetectorModel
from picode.detection.export import convert_to_coreml, CoreMLExportConfig

# Load trained model
model = FastDetectorModel(input_size=320)
checkpoint = torch.load("checkpoints/fast_detector_final.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Export to Core ML
config = CoreMLExportConfig(
    quantization="fp16",  # or "int8" for smallest size
    minimum_deployment_target="iOS15",
)

output_path = convert_to_coreml(
    model=model,
    output_path="FastDetector.mlpackage",
    config=config,
)
print(f"Exported to {output_path}")
```

### Model Sizes

| Quantization | Size | Accuracy Impact |
|--------------|------|-----------------|
| float32 | ~5 MB | Baseline |
| float16 | ~2.5 MB | Minimal |
| int8 | ~1.5 MB | Slight degradation |

## Add to iOS App

1. **Drag `FastDetector.mlpackage` into Xcode**
   - Add to picode-ios target
   - Ensure "Copy items if needed" is checked

2. **Xcode compiles to `.mlmodelc`**
   - Automatic during build

3. **FastDetector.swift loads it automatically**
   ```swift
   // FastDetector.swift already handles this:
   if let detectorURL = Bundle.main.url(forResource: "FastDetector", withExtension: "mlmodelc") {
       // Model loaded!
   }
   ```

4. **Build and run on device**

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  1. Train Encoder/Decoder (picode_v2)                       │
│     └── checkpoints/encoder.pt                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Prepare Detection Data                                  │
│     ├── data/detection/positive/ (training images)          │
│     └── data/detection/negative/ (hard negatives)           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Train FastDetector                                      │
│     └── python train_fast_detector.py                       │
│     └── checkpoints/fast_detector_final.pt                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Export to Core ML                                       │
│     └── detect-export coreml fast_detector.pt -o Model.mlp  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Add to Xcode & Run on iPhone                            │
│     └── Drag FastDetector.mlpackage into project            │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "No images found in directory"

```bash
# Check your image directory has .jpg or .png files
ls data/detection/positive/*.jpg | head
```

### "CUDA out of memory"

Reduce batch size:

```python
train_loader = DataLoader(train_dataset, batch_size=8, ...)  # Reduce from 16
```

### "Model not learning (loss ~0.69)"

- Verify encoder checkpoint loads correctly
- Check encoder produces visible watermarks
- Try lower learning rate (1e-5)

### "Core ML export fails"

```bash
# Install coremltools
pip install coremltools>=7.0

# Check model runs on CPU first
python -c "
import torch
from picode.detection import FastDetectorModel
m = FastDetectorModel()
x = torch.randn(1, 3, 320, 320)
print(m(x))
"
```

## Next Steps

After FastDetector training:

1. **Train decoder separately** if not already done
2. **Export decoder to Core ML** for complete pipeline
3. **Test end-to-end** on iOS device
4. **Iterate** based on real-world performance

## References

- [Detection Improvements Design Doc](./detection_improvements.md)
- [Mobile Deployment Guide](./mobile_deployment.md)
- [picode_v2 Model Architecture](./model_improvements.md)
