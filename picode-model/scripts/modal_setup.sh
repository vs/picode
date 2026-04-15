#!/bin/bash
# Modal setup and utility commands for Picode training

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Cleanup function for upload-data command
cleanup_tarball() {
    if [ -n "$tarball" ] && [ -f "$tarball" ]; then
        rm -f "$tarball"
    fi
}

usage() {
    cat << EOF
Usage: ./scripts/modal_setup.sh <command> [args]

Commands:
    setup           Install Modal and authenticate
    upload-data     Upload training data to Modal volume
    train           Start training on Modal
    resume          Resume training from latest checkpoint
    list            List checkpoints in Modal volume
    download        Download checkpoints from Modal
    logs            View training logs
    shell           Open interactive shell on Modal GPU

Examples:
    ./scripts/modal_setup.sh setup
    ./scripts/modal_setup.sh upload-data ./data/train
    ./scripts/modal_setup.sh train
    ./scripts/modal_setup.sh train --override training.lr=0.0002
    ./scripts/modal_setup.sh resume
    ./scripts/modal_setup.sh download ./checkpoints_remote
EOF
}

# Check for required tools
check_modal() {
    if ! command -v modal >/dev/null 2>&1; then
        echo "Error: modal is not installed. Run './scripts/modal_setup.sh setup' first."
        exit 1
    fi
}

case "$1" in
    setup)
        echo "Installing Modal..."
        pip install modal
        echo ""
        echo "Authenticating with Modal..."
        if ! modal token new; then
            echo "Error: Failed to authenticate with Modal"
            exit 1
        fi
        echo ""
        echo "Creating volumes..."
        modal volume create picode-data 2>/dev/null || echo "Volume picode-data already exists"
        modal volume create picode-checkpoints 2>/dev/null || echo "Volume picode-checkpoints already exists"
        echo ""
        echo "Setup complete! Next steps:"
        echo "  1. Upload training data: ./scripts/modal_setup.sh upload-data ./data/train"
        echo "  2. Start training: ./scripts/modal_setup.sh train"
        ;;

    upload-data)
        check_modal
        if [ -z "$2" ]; then
            echo "Error: Please provide local data path"
            echo "Usage: ./scripts/modal_setup.sh upload-data <local-path>"
            exit 1
        fi

        local_path="$2"

        # Validate path exists
        if [ ! -d "$local_path" ]; then
            echo "Error: Directory does not exist: $local_path"
            exit 1
        fi

        dir_name=$(basename "$local_path")
        tarball="/tmp/picode_upload_${dir_name}.tar"

        # Set up cleanup trap
        trap cleanup_tarball EXIT

        echo "Creating tarball from $local_path..."
        if ! tar -cf "$tarball" -C "$(dirname "$local_path")" "$dir_name"; then
            echo "Error: Failed to create tarball"
            exit 1
        fi
        tarball_size=$(du -h "$tarball" | cut -f1)
        echo "Tarball created: $tarball ($tarball_size)"

        echo "Uploading tarball to Modal volume..."
        if ! modal volume put picode-data "$tarball" /; then
            echo "Error: Failed to upload tarball to Modal"
            exit 1
        fi

        echo "Extracting on Modal..."
        if ! modal run "$SCRIPT_DIR/modal_extract.py" --tarball "/picode_upload_${dir_name}.tar" --dest /; then
            echo "Error: Failed to extract tarball on Modal"
            exit 1
        fi

        echo "Cleaning up remote tarball..."
        modal volume rm picode-data "/picode_upload_${dir_name}.tar" || true

        # Local cleanup handled by trap
        echo "Upload complete!"
        echo "Verify with: modal volume ls picode-data"
        ;;

    train)
        check_modal
        shift
        echo "Starting training on Modal..."
        modal run "$SCRIPT_DIR/modal_train.py::train" --config "$PROJECT_DIR/configs/modal_training.yaml" "$@"
        ;;

    resume)
        check_modal
        shift
        echo "Resuming training from latest checkpoint..."
        modal run "$SCRIPT_DIR/modal_train.py::train" --config "$PROJECT_DIR/configs/modal_training.yaml" --resume "$@"
        ;;

    list)
        check_modal
        echo "Listing checkpoints..."
        modal volume ls picode-checkpoints --recursive
        ;;

    download)
        check_modal
        if [ -z "$2" ]; then
            local_path="./checkpoints_modal"
        else
            local_path="$2"
        fi
        echo "Downloading checkpoints to $local_path..."
        mkdir -p "$local_path"
        modal volume get picode-checkpoints / "$local_path"
        echo "Download complete!"
        ;;

    logs)
        check_modal
        echo "Viewing recent Modal logs..."
        modal app logs picode-training
        ;;

    shell)
        check_modal
        echo "Opening interactive shell on Modal GPU..."
        modal shell "$SCRIPT_DIR/modal_train.py"
        ;;

    *)
        usage
        exit 1
        ;;
esac
