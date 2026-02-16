#!/bin/bash
# Modal setup and utility commands for Picode training

set -e

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

case "$1" in
    setup)
        echo "Installing Modal..."
        pip install modal
        echo ""
        echo "Authenticating with Modal..."
        modal token new
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
        if [ -z "$2" ]; then
            echo "Error: Please provide local data path"
            echo "Usage: ./scripts/modal_setup.sh upload-data <local-path>"
            exit 1
        fi
        echo "Uploading $2 to Modal volume..."
        modal volume put picode-data "$2" /
        echo "Upload complete!"
        echo "Verify with: modal volume ls picode-data"
        ;;

    train)
        shift
        echo "Starting training on Modal..."
        modal run modal_train.py::train --config configs/modal_training.yaml "$@"
        ;;

    resume)
        shift
        echo "Resuming training from latest checkpoint..."
        modal run modal_train.py::train --config configs/modal_training.yaml --resume "$@"
        ;;

    list)
        echo "Listing checkpoints..."
        modal volume ls picode-checkpoints --recursive
        ;;

    download)
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
        echo "Viewing recent Modal logs..."
        modal app logs picode-training
        ;;

    shell)
        echo "Opening interactive shell on Modal GPU..."
        modal shell modal_train.py
        ;;

    *)
        usage
        exit 1
        ;;
esac
