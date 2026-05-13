#!/bin/bash
# Generate batch, upload to Modal, delete locally

cd ~/workspace/code/picode-model
source .venv/bin/activate

BATCH_NUM=$1
OUTPUT_DIR="data/detection_batch_${BATCH_NUM}"

echo "=== Batch $BATCH_NUM: generating ==="

# Generate
./scripts/batch_generate.sh $BATCH_NUM

if [ $? -ne 0 ]; then
    echo "ERROR: Batch $BATCH_NUM generation failed"
    exit 1
fi

echo "=== Batch $BATCH_NUM: uploading to Modal ==="

# Upload to Modal
modal volume put picode-data "$OUTPUT_DIR" /detection_dataset/

if [ $? -ne 0 ]; then
    echo "ERROR: Batch $BATCH_NUM upload failed"
    exit 1
fi

echo "=== Batch $BATCH_NUM: cleaning up local ==="

# Delete local
rm -rf "$OUTPUT_DIR"

echo "=== Batch $BATCH_NUM: complete (uploaded & deleted) ==="
