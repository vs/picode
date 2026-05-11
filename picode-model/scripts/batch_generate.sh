#!/bin/bash
# Batch generation script - runs in small chunks to avoid OOM

cd ~/workspace/code/picode-model
source .venv/bin/activate

BATCH_NUM=$1
SAMPLES=2000
OUTPUT_DIR="data/detection_batch_${BATCH_NUM}"

echo "=== Batch $BATCH_NUM: generating $SAMPLES samples ==="
python scripts/prepare_detection_dataset.py \
    --encoder-checkpoint checkpoints/best.pt \
    --image-dir data/coco/train2017 \
    --output-dir "$OUTPUT_DIR" \
    --num-samples $SAMPLES \
    --positive-ratio 0.5 \
    --batch-size 50

echo "=== Batch $BATCH_NUM complete ==="
