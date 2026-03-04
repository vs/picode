#!/bin/bash
# Download MIRFLICKR-25K dataset
# Source: https://press.liacs.nl/mirflickr/
# Used in the original StegaStamp paper for steganography training

set -e

DEST_DIR="${1:-data/mirflickr}"
MIRFLICKR_URL="https://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip"

echo "MIRFLICKR-25K Dataset Downloader"
echo "================================"
echo "Destination: $DEST_DIR"
echo ""

# Create destination directory
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

# Check if already downloaded
if [ -d "mirflickr" ] && [ "$(ls -1 mirflickr/*.jpg 2>/dev/null | wc -l)" -gt 20000 ]; then
    echo "Dataset already exists with $(ls -1 mirflickr/*.jpg | wc -l) images"
    echo "To re-download, remove $DEST_DIR/mirflickr first"
    exit 0
fi

# Download
echo "Downloading MIRFLICKR-25K (~2.9GB)..."
if command -v curl &> /dev/null; then
    curl -L -O --progress-bar "$MIRFLICKR_URL"
elif command -v wget &> /dev/null; then
    wget --progress=bar:force "$MIRFLICKR_URL"
else
    echo "Error: curl or wget required"
    exit 1
fi

# Extract
echo ""
echo "Extracting..."
unzip -q mirflickr25k.zip

# Cleanup
rm mirflickr25k.zip

# Count images
NUM_IMAGES=$(ls -1 mirflickr/*.jpg | wc -l)
echo ""
echo "Done! Downloaded $NUM_IMAGES images to $DEST_DIR/mirflickr/"
echo ""
echo "To train with this dataset:"
echo "  picode-train configs/stegastamp_baseline.yaml data.path=$DEST_DIR/mirflickr"
