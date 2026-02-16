#!/bin/bash
# Download COCO train2017 dataset
# Source: https://cocodataset.org/
# 118K diverse images, commonly used for vision tasks

set -e

DEST_DIR="${1:-data/coco}"
COCO_URL="http://images.cocodataset.org/zips/train2017.zip"

echo "COCO train2017 Dataset Downloader"
echo "=================================="
echo "Destination: $DEST_DIR"
echo ""

# Create destination directory
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

# Check if already downloaded
if [ -d "train2017" ] && [ "$(ls -1 train2017/*.jpg 2>/dev/null | wc -l)" -gt 100000 ]; then
    echo "Dataset already exists with $(ls -1 train2017/*.jpg | wc -l) images"
    echo "To re-download, remove $DEST_DIR/train2017 first"
    exit 0
fi

# Download
echo "Downloading COCO train2017 (~18GB)..."
echo "This may take a while..."
echo ""
if command -v curl &> /dev/null; then
    curl -L -O --progress-bar "$COCO_URL"
elif command -v wget &> /dev/null; then
    wget --progress=bar:force "$COCO_URL"
else
    echo "Error: curl or wget required"
    exit 1
fi

# Extract
echo ""
echo "Extracting (this may take a few minutes)..."
unzip -q train2017.zip

# Cleanup
rm train2017.zip

# Count images
NUM_IMAGES=$(ls -1 train2017/*.jpg | wc -l)
echo ""
echo "Done! Downloaded $NUM_IMAGES images to $DEST_DIR/train2017/"
echo ""
echo "To train with this dataset:"
echo "  picode-train configs/stegastamp_baseline.yaml data.path=$DEST_DIR/train2017"
