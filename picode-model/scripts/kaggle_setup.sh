#!/bin/bash
# Kaggle setup and utility commands for PicodeLite training.
#
# Mirrors the pattern of modal_setup.sh but for Kaggle Kernels API.
# The repo is private, so we upload the source as a Kaggle dataset
# rather than git cloning at runtime.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
KAGGLE_DIR="$SCRIPT_DIR/kaggle"
KERNEL_SLUG="picodelite-training"

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
Usage: ./scripts/kaggle_setup.sh <command> [args]

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
    ./scripts/kaggle_setup.sh push
    ./scripts/kaggle_setup.sh status
    ./scripts/kaggle_setup.sh output ./kaggle_ckpts
    ./scripts/kaggle_setup.sh resume
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
    echo "  2. Push training kernel: ./scripts/kaggle_setup.sh push"
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
    rsync -a --exclude='data/' --exclude='checkpoints/' --exclude='*.pyc' \
        --exclude='__pycache__' --exclude='.git' --exclude='venv/' \
        --exclude='.venv/' --exclude='runs/' --exclude='*.egg-info' \
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

    echo ""
    echo "Done! Dataset: https://www.kaggle.com/datasets/$USERNAME/$DATASET_SLUG"
}

cmd_push() {
    check_kaggle
    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo "Error: Cannot detect Kaggle username. Run setup first."
        exit 1
    fi

    # Create a working copy of kernel metadata with username substituted
    PUSH_DIR=$(mktemp -d)
    trap "rm -rf $PUSH_DIR" EXIT

    # Substitute username in metadata
    sed "s/INSERT_YOUR_USERNAME/$USERNAME/g" "$KAGGLE_DIR/kernel-metadata.json" > "$PUSH_DIR/kernel-metadata.json"

    # If picodelite-checkpoints dataset exists, add it to sources
    if kaggle datasets status "$USERNAME/picodelite-checkpoints" >/dev/null 2>&1; then
        echo "Found checkpoint dataset, adding to kernel sources..."
        python3 -c "
import json
with open('$PUSH_DIR/kernel-metadata.json') as f:
    meta = json.load(f)
src = '$USERNAME/picodelite-checkpoints'
if src not in meta['dataset_sources']:
    meta['dataset_sources'].append(src)
with open('$PUSH_DIR/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
    fi

    # Copy training script
    cp "$KAGGLE_DIR/kaggle_train.py" "$PUSH_DIR/kaggle_train.py"

    echo "Pushing kernel $USERNAME/$KERNEL_SLUG..."
    echo "Metadata:"
    cat "$PUSH_DIR/kernel-metadata.json"
    echo ""

    kaggle kernels push -p "$PUSH_DIR"

    echo ""
    echo "Kernel pushed! Check status with: ./scripts/kaggle_setup.sh status"
    echo "View at: https://www.kaggle.com/code/$USERNAME/$KERNEL_SLUG"
}

cmd_status() {
    check_kaggle
    USERNAME=$(get_username)
    if [ -z "$USERNAME" ]; then
        echo "Error: Cannot detect Kaggle username. Run setup first."
        exit 1
    fi

    echo "Checking kernel status..."
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

    echo "Downloading kernel output to $LOCAL_DIR..."
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

    CKPT_DATASET="picodelite-checkpoints"
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
    echo "=== Step 2: Uploading checkpoints as dataset ==="

    cat > "$STAGING_DIR/dataset-metadata.json" << METAEOF
{
  "title": "picodelite-checkpoints",
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
