#!/bin/bash
# Kaggle setup and utility commands for Picode model training.
#
# Mirrors the pattern of modal_setup.sh but for Kaggle Kernels API.
# The repo is private, so we upload the source as a Kaggle dataset
# rather than git cloning at runtime.
#
# Supports multiple models via --model flag (default: picodelite).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
KAGGLE_DIR="$SCRIPT_DIR/kaggle"

# Default model
MODEL_NAME="picodelite"

# Parse --model flag from any position
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model=*)
            MODEL_NAME="${1#*=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

# Derived names based on model
KERNEL_SLUG="${MODEL_NAME}-training"
CKPT_DATASET="${MODEL_NAME}-checkpoints"

# --- Helpers ---

get_username() {
    if [ -f ~/.kaggle/kaggle.json ]; then
        python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])" 2>/dev/null
    fi
}

check_kaggle() {
    if ! command -v kaggle >/dev/null 2>&1; then
        echo "Error: kaggle CLI not installed. Run: pip install kaggle"
        exit 1
    fi
}

usage() {
    cat << 'EOF'
Usage: ./scripts/kaggle_setup.sh [--model NAME] <command> [args]

Models:
    picodelite      PicodeLite model (default)
    picodelite_256bit  PicodeLite 256-bit model
    picodeframe     PicodeFrame model

Commands:
    setup           Verify kaggle CLI installed, show username
    upload-code     Package picode-model/ as Kaggle dataset <user>/picode-source
    push            Push training kernel to Kaggle
    status          Check kernel execution status
    output [dir]    Download checkpoints.zip from kernel output
    resume          Download prev output, upload as checkpoint dataset, push new run

Examples:
    ./scripts/kaggle_setup.sh setup
    ./scripts/kaggle_setup.sh upload-code
    ./scripts/kaggle_setup.sh --model picodeframe push
    ./scripts/kaggle_setup.sh --model picodeframe status
    ./scripts/kaggle_setup.sh --model picodeframe output ./kaggle_ckpts
    ./scripts/kaggle_setup.sh --model picodeframe resume
EOF
}

# --- Commands ---

cmd_setup() {
    echo "Checking kaggle CLI..."
    check_kaggle
    echo "kaggle CLI: $(which kaggle)"

    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo ""
        echo "Error: Cannot detect Kaggle username."
        echo "Make sure ~/.kaggle/kaggle.json exists with your API credentials."
        echo "  1. Go to https://www.kaggle.com/settings > API > Create New Token"
        echo "  2. Save kaggle.json to ~/.kaggle/kaggle.json"
        echo "  3. chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi

    echo "Kaggle username: $USERNAME"
    echo ""
    echo "Setup complete! Next steps:"
    echo "  1. Upload source code:  ./scripts/kaggle_setup.sh upload-code"
    echo "  2. Push training kernel: ./scripts/kaggle_setup.sh --model $MODEL_NAME push"
}

cmd_upload_code() {
    check_kaggle
    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo "Error: Cannot detect Kaggle username. Run setup first."
        exit 1
    fi

    DATASET_SLUG="picode-source"
    STAGING_DIR=$(mktemp -d)
    trap "rm -rf $STAGING_DIR" EXIT

    echo "Packaging picode-model/ as dataset $USERNAME/$DATASET_SLUG..."

    # Create dataset metadata
    cat > "$STAGING_DIR/dataset-metadata.json" << METAEOF
{
  "title": "picode-source",
  "id": "$USERNAME/$DATASET_SLUG",
  "licenses": [{"name": "Apache 2.0"}]
}
METAEOF

    # Copy source code (exclude data, checkpoints, caches)
    mkdir -p "$STAGING_DIR/picode-model"
    rsync -a --exclude='data/' --exclude='checkpoints/' --exclude='kaggle_ckpts/' \
        --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
        --exclude='venv/' --exclude='.venv/' --exclude='runs/' \
        --exclude='*.egg-info' --exclude='*.pt' --exclude='notebooks/' \
        "$PROJECT_DIR/" "$STAGING_DIR/picode-model/"

    echo "Staging directory contents:"
    du -sh "$STAGING_DIR/picode-model/"

    echo ""
    echo "Uploading dataset (this may take a minute)..."
    # Use --dir-mode to handle new vs existing dataset
    if kaggle datasets status "$USERNAME/$DATASET_SLUG" >/dev/null 2>&1; then
        echo "Dataset exists, creating new version..."
        kaggle datasets version -p "$STAGING_DIR" -m "Updated source code" --dir-mode zip
    else
        echo "Creating new dataset..."
        kaggle datasets create -p "$STAGING_DIR" --dir-mode zip
    fi

    # Wait for dataset to be ready before returning
    echo ""
    echo "Waiting for dataset to be ready..."
    for i in $(seq 1 30); do
        STATUS=$(kaggle datasets status "$USERNAME/$DATASET_SLUG" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "ready" ]; then
            echo "Dataset is ready!"
            break
        fi
        echo "  Status: $STATUS (attempt $i/30, waiting 10s...)"
        sleep 10
    done

    echo "Done! Dataset: https://www.kaggle.com/datasets/$USERNAME/$DATASET_SLUG"
}

cmd_push() {
    check_kaggle
    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo "Error: Cannot detect Kaggle username. Run setup first."
        exit 1
    fi

    echo "Model: $MODEL_NAME (kernel: $KERNEL_SLUG)"

    # Create a working copy of kernel metadata with username substituted
    PUSH_DIR=$(mktemp -d)
    trap "rm -rf $PUSH_DIR" EXIT

    # Generate kernel metadata for this model
    cat > "$PUSH_DIR/kernel-metadata.json" << METAEOF
{
  "id": "$USERNAME/$KERNEL_SLUG",
  "title": "$KERNEL_SLUG",
  "code_file": "kaggle_train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "$USERNAME/picode-source",
    "awsaf49/coco-2017-dataset"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
METAEOF

    # If checkpoint dataset exists for this model, add it to sources
    if kaggle datasets status "$USERNAME/$CKPT_DATASET" >/dev/null 2>&1; then
        echo "Found checkpoint dataset ($CKPT_DATASET), adding to kernel sources..."
        python3 -c "
import json
with open('$PUSH_DIR/kernel-metadata.json') as f:
    meta = json.load(f)
src = '$USERNAME/$CKPT_DATASET'
if src not in meta['dataset_sources']:
    meta['dataset_sources'].append(src)
with open('$PUSH_DIR/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
    fi

    # Copy training script and inject MODEL_NAME at the top
    {
        echo "import os; os.environ['MODEL_NAME'] = '$MODEL_NAME'  # injected by kaggle_setup.sh"
        cat "$KAGGLE_DIR/kaggle_train.py"
    } > "$PUSH_DIR/kaggle_train.py"

    echo "Pushing kernel $USERNAME/$KERNEL_SLUG..."
    echo "Metadata:"
    cat "$PUSH_DIR/kernel-metadata.json"
    echo ""

    kaggle kernels push -p "$PUSH_DIR"

    echo ""
    echo "Kernel pushed! Check status with: ./scripts/kaggle_setup.sh --model $MODEL_NAME status"
    echo "View at: https://www.kaggle.com/code/$USERNAME/$KERNEL_SLUG"
}

cmd_status() {
    check_kaggle
    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo "Error: Cannot detect Kaggle username. Run setup first."
        exit 1
    fi

    echo "Checking kernel status for $KERNEL_SLUG..."
    kaggle kernels status "$USERNAME/$KERNEL_SLUG"
}

cmd_output() {
    check_kaggle
    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo "Error: Cannot detect Kaggle username. Run setup first."
        exit 1
    fi

    LOCAL_DIR="${2:-./kaggle_checkpoints}"
    mkdir -p "$LOCAL_DIR"

    echo "Downloading kernel output from $KERNEL_SLUG to $LOCAL_DIR..."
    kaggle kernels output "$USERNAME/$KERNEL_SLUG" -p "$LOCAL_DIR"

    # Check if checkpoints.zip was downloaded
    if [ -f "$LOCAL_DIR/checkpoints.zip" ]; then
        echo "Extracting checkpoints.zip..."
        unzip -o "$LOCAL_DIR/checkpoints.zip" -d "$LOCAL_DIR"
        echo ""
        echo "Checkpoints extracted to $LOCAL_DIR/"
        ls -lh "$LOCAL_DIR/checkpoints/" 2>/dev/null || true
    else
        echo "Warning: checkpoints.zip not found in output"
        echo "Contents of $LOCAL_DIR:"
        ls -la "$LOCAL_DIR/"
    fi
}

cmd_resume() {
    check_kaggle
    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo "Error: Cannot detect Kaggle username. Run setup first."
        exit 1
    fi

    STAGING_DIR=$(mktemp -d)
    trap "rm -rf $STAGING_DIR" EXIT

    # Step 1: Download kernel output
    echo "=== Step 1: Downloading previous kernel output ==="
    DOWNLOAD_DIR=$(mktemp -d)
    kaggle kernels output "$USERNAME/$KERNEL_SLUG" -p "$DOWNLOAD_DIR"

    if [ ! -f "$DOWNLOAD_DIR/checkpoints.zip" ]; then
        echo "Error: No checkpoints.zip in kernel output. Has the kernel finished running?"
        rm -rf "$DOWNLOAD_DIR"
        exit 1
    fi

    # Extract checkpoints
    echo "Extracting checkpoints..."
    unzip -o "$DOWNLOAD_DIR/checkpoints.zip" -d "$STAGING_DIR"

    # Step 2: Upload as checkpoint dataset
    echo ""
    echo "=== Step 2: Uploading checkpoints as dataset ($CKPT_DATASET) ==="

    cat > "$STAGING_DIR/dataset-metadata.json" << METAEOF
{
  "title": "$CKPT_DATASET",
  "id": "$USERNAME/$CKPT_DATASET",
  "licenses": [{"name": "Apache 2.0"}]
}
METAEOF

    if kaggle datasets status "$USERNAME/$CKPT_DATASET" >/dev/null 2>&1; then
        echo "Checkpoint dataset exists, creating new version..."
        kaggle datasets version -p "$STAGING_DIR" -m "Updated checkpoints" --dir-mode zip
    else
        echo "Creating checkpoint dataset..."
        kaggle datasets create -p "$STAGING_DIR" --dir-mode zip
    fi

    rm -rf "$DOWNLOAD_DIR"

    # Step 3: Wait for dataset to be ready
    echo ""
    echo "Waiting for dataset to process (this may take a few minutes)..."
    sleep 30

    # Step 4: Push kernel with checkpoint dataset
    echo ""
    echo "=== Step 3: Pushing kernel with resume ==="
    cmd_push
}

# --- Main ---

case "${1:-}" in
    setup)       cmd_setup ;;
    upload-code) cmd_upload_code ;;
    push)        cmd_push ;;
    status)      cmd_status ;;
    output)      cmd_output "$@" ;;
    resume)      cmd_resume ;;
    *)           usage; exit 1 ;;
esac
