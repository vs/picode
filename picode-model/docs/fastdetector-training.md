# FastDetector Training Guide

This guide covers training the FastDetector model for real-time watermark detection on iOS.

## Overview

FastDetector is a lightweight MobileNetV3-based model that:
- Classifies images as watermarked or clean
- Predicts corner coordinates for perspective correction
- Runs at 30+ FPS on iOS Neural Engine

## Pre-Generated Dataset Workflow (Recommended)

Pre-generating the dataset locally saves significant Modal GPU costs since the encoder (which generates watermarked images) is the bottleneck.

### Step 1: Generate Dataset Locally

Run on your laptop (MPS) or VPS (CPU):

```bash
cd picode-model

# Generate 50k samples (~1.5 hours on MPS, 2-3 hours on CPU)
python scripts/prepare_detection_dataset.py \
    --encoder-checkpoint checkpoints/best.pt \
    --image-dir data/coco/train2017 \
    --output-dir data/detection_dataset \
    --num-samples 50000

# Options:
#   --positive-ratio 0.5    # Ratio of watermarked vs clean images
#   --input-size 320        # Detector input size
#   --batch-size 100        # Samples per shard file
#   --seed 42               # Random seed for reproducibility
```

**Output:**
- `data/detection_dataset/shard_00000.pt`, `shard_00001.pt`, ...
- `data/detection_dataset/metadata.json`

### Step 2: Upload to Modal Volume

```bash
modal volume put picode-data data/detection_dataset /detection_dataset
```

### Step 3: Train on Modal

```bash
# Test run (10 epochs)
modal run scripts/modal_train_detector.py::train_detector_pregenerated --epochs 10

# Full training (50 epochs)
modal run scripts/modal_train_detector.py::train_detector_pregenerated \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001
```

### Step 4: Download Trained Model

```bash
modal volume get picode-checkpoints detection/ ./checkpoints/detection
```

## Cost Comparison

| Method | Data Generation | Training (50 epochs) | Total |
|--------|-----------------|---------------------|-------|
| Pre-generated | Free (local) | ~$0.30-0.50 | ~$0.40 |
| On-the-fly | ~$1.20 (T4 GPU) | ~$0.30-0.50 | ~$1.70 |

## Alternative: On-the-Fly Generation

If you prefer to generate data during training (slower, more expensive):

```bash
# Uses encoder on GPU during training
modal run scripts/modal_train_detector.py::train_detector \
    --epochs 50 \
    --encoder-checkpoint stegastamp_original/best.pt
```

## Export to CoreML (iOS)

After training, export for iOS deployment:

```bash
python scripts/export_detector.py \
    --checkpoint checkpoints/detection/best.pt \
    --output checkpoints/detection/FastDetector.mlpackage \
    --fp16 \
    --validate
```

## Evaluation

Evaluate trained detector:

```bash
# Local evaluation
python scripts/evaluate_detector.py \
    --detector-checkpoint checkpoints/detection/best.pt \
    --encoder-checkpoint checkpoints/best.pt \
    --image-dir data/coco/val2017 \
    --num-samples 500

# Modal evaluation
modal run scripts/modal_train_detector.py::evaluate_detector \
    --detector-checkpoint detection/best.pt \
    --num-samples 500
```

## Files

- `scripts/prepare_detection_dataset.py` - Generate pre-encoded dataset locally
- `scripts/modal_train_detector.py` - Modal training (both methods)
- `scripts/train_detector.py` - Local training script
- `scripts/export_detector.py` - CoreML export
- `scripts/evaluate_detector.py` - Evaluation script
- `picode/detection/training/` - Training infrastructure
- `picode/detection/fast_detector.py` - FastDetector model
