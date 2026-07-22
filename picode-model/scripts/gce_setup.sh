#!/bin/bash
# GCE setup and utility commands for Picode training.
#
# Mirrors the pattern of modal_setup.sh but for Google Compute Engine.
# Uses a persistent VM with T4 GPU, tmux for training sessions,
# and gcloud compute scp for file transfer.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# VM configuration
VM_NAME="picode-training"
ZONE="us-central1-a"
MACHINE_TYPE="g2-standard-8"
GPU_TYPE="nvidia-l4"
GPU_COUNT=1
BOOT_DISK_SIZE="400GB"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

# tmux session name for training
TMUX_SESSION="training"

# --- Helpers ---

usage() {
    cat << 'EOF'
Usage: ./scripts/gce_setup.sh <command> [args]

Commands:
    create          Create GCE VM with T4 GPU
    setup           SSH into VM and install Python deps + picode package
    upload-data     Upload local training images via SCP
    upload-code     Upload picode-model source to VM
    setup-data      Download COCO train2017 on the VM (~18GB)
    train           Start training in a tmux session (detached)
    train --resume  Resume training from latest checkpoint
    train-detector  Train FastDetector using encoder checkpoint
    upload-ckpt     Upload a local checkpoint to the VM
    logs            Tail the training console output
    attach          Attach to the tmux training session
    list            List checkpoints on the VM
    download        Download checkpoints from VM via SCP
    ssh             Interactive SSH to the VM
    stop            Stop the VM (preserves disk, stops billing)
    start           Start a stopped VM
    status          Show VM status
    delete          Delete the VM entirely

Examples:
    ./scripts/gce_setup.sh create
    ./scripts/gce_setup.sh setup
    ./scripts/gce_setup.sh upload-code
    ./scripts/gce_setup.sh setup-data
    ./scripts/gce_setup.sh train
    ./scripts/gce_setup.sh logs
    ./scripts/gce_setup.sh stop
    ./scripts/gce_setup.sh start
    ./scripts/gce_setup.sh train --resume
    ./scripts/gce_setup.sh download ./checkpoints_gce
EOF
}

check_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        echo "Error: gcloud CLI not installed."
        echo "Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

vm_ssh() {
    # Run a command on the VM via SSH
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="$1"
}

vm_ssh_tty() {
    # Run a command on the VM via SSH with TTY (for interactive use)
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- -t "$1"
}

# --- Commands ---

cmd_create() {
    check_gcloud
    echo "Creating VM: $VM_NAME"
    echo "  Zone:         $ZONE"
    echo "  Machine:      $MACHINE_TYPE"
    echo "  GPU:          $GPU_TYPE x$GPU_COUNT"
    echo "  Boot disk:    $BOOT_DISK_SIZE"
    echo "  OS:           $IMAGE_FAMILY ($IMAGE_PROJECT)"
    echo ""

    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type=pd-standard \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True"

    echo ""
    echo "VM created! Wait ~2 minutes for it to boot, then run:"
    echo "  ./scripts/gce_setup.sh setup"
}

cmd_setup() {
    check_gcloud
    echo "Setting up VM environment..."
    echo ""

    # Upload and run the startup script
    echo "Uploading startup script..."
    gcloud compute scp "$SCRIPT_DIR/gce_startup.sh" "$VM_NAME:~/gce_startup.sh" --zone="$ZONE"

    echo "Running startup script (this may take 5-10 minutes)..."
    vm_ssh "chmod +x ~/gce_startup.sh && ~/gce_startup.sh"

    echo ""
    echo "Verifying GPU..."
    vm_ssh "python3 -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\""

    echo ""
    echo "Setup complete! Next steps:"
    echo "  1. Upload source code:  ./scripts/gce_setup.sh upload-code"
    echo "  2. Setup training data: ./scripts/gce_setup.sh setup-data"
    echo "  3. Start training:      ./scripts/gce_setup.sh train"
}

cmd_upload_data() {
    check_gcloud
    local_path="$1"
    if [ -z "$local_path" ]; then
        echo "Error: Please provide local data path"
        echo "Usage: ./scripts/gce_setup.sh upload-data <local-path>"
        exit 1
    fi

    if [ ! -d "$local_path" ]; then
        echo "Error: Directory does not exist: $local_path"
        exit 1
    fi

    echo "Creating remote data directory..."
    vm_ssh "mkdir -p ~/data"

    echo "Uploading $local_path to VM:~/data/ ..."
    gcloud compute scp --recurse "$local_path" "$VM_NAME:~/data/" --zone="$ZONE"

    echo ""
    echo "Upload complete! Verifying..."
    vm_ssh "ls ~/data/ && echo '' && echo 'Total:' && du -sh ~/data/"
}

cmd_upload_code() {
    check_gcloud
    echo "Uploading picode-model source to VM..."

    # Create a tarball excluding data, checkpoints, caches
    tarball=$(mktemp /tmp/picode_source_XXXXXX.tar.gz)
    trap "rm -f $tarball" EXIT

    tar -czf "$tarball" \
        -C "$PROJECT_DIR/.." \
        --exclude='picode-model/data' \
        --exclude='picode-model/checkpoints' \
        --exclude='picode-model/kaggle_ckpts' \
        --exclude='picode-model/runs' \
        --exclude='picode-model/notebooks' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='.venv' \
        --exclude='*.egg-info' \
        --exclude='*.pt' \
        picode-model/

    tarball_size=$(du -h "$tarball" | cut -f1)
    echo "Source tarball: $tarball_size"

    echo "Uploading..."
    gcloud compute scp "$tarball" "$VM_NAME:/tmp/picode_source.tar.gz" --zone="$ZONE"

    echo "Extracting on VM..."
    vm_ssh "export PATH=\$HOME/.local/bin:\$PATH && cd ~ && rm -rf picode-model && tar -xzf /tmp/picode_source.tar.gz && rm /tmp/picode_source.tar.gz && cd picode-model && pip install '.[lpips,kornia]' -q"

    echo ""
    echo "Code uploaded and installed!"
    vm_ssh "ls ~/picode-model/"
}

cmd_setup_data() {
    check_gcloud
    echo "Downloading COCO train2017 on the VM..."
    echo "This downloads ~18GB directly to the VM (no local transfer needed)."
    echo ""

    vm_ssh "mkdir -p ~/data && cd ~/data && \
        if [ -d train ]; then \
            echo 'train/ already exists, skipping download.'; \
            echo \"Images: \$(ls train/ | wc -l)\"; \
        else \
            echo 'Downloading COCO train2017 (~18GB)...' && \
            wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip && \
            echo 'Extracting...' && \
            unzip -q train2017.zip && \
            mv train2017 train && \
            rm train2017.zip && \
            echo \"Done! Images: \$(ls train/ | wc -l)\"; \
        fi"
}

cmd_setup_mirflickr() {
    check_gcloud
    echo "Downloading MIR Flickr 1M on the VM (~100GB total)..."
    echo "  Train images (zips 0-8): ~/data/mirflickr_train/"
    echo "  Detect images (zip 9):  ~/data/mirflickr_detect/"
    echo ""

    vm_ssh "mkdir -p \$HOME/data/mirflickr_train \$HOME/data/mirflickr_detect && \
        BASE_URL='https://press.liacs.nl/mirflickr/mirflickr1m.v3b' && \
        for i in \$(seq 0 9); do \
            if [ -f \$HOME/data/mirflickr_train/done_\$i ] || [ -f \$HOME/data/mirflickr_detect/done_\$i ]; then \
                echo \"Zip \$i already processed, skipping.\"; \
                continue; \
            fi; \
            echo \"Downloading images\$i.zip ...\"; \
            wget -q --show-progress -O /tmp/mirflickr_\$i.zip \$BASE_URL/images\$i.zip; \
            if [ \$i -eq 9 ]; then \
                echo \"Extracting zip \$i to mirflickr_detect/ ...\"; \
                unzip -q -o -j /tmp/mirflickr_\$i.zip -d \$HOME/data/mirflickr_detect/; \
                rm /tmp/mirflickr_\$i.zip; \
                touch \$HOME/data/mirflickr_detect/done_\$i; \
                echo \"Zip \$i done. Images: \$(ls \$HOME/data/mirflickr_detect/*.jpg 2>/dev/null | wc -l)\"; \
            else \
                echo \"Extracting zip \$i to mirflickr_train/ ...\"; \
                unzip -q -o -j /tmp/mirflickr_\$i.zip -d \$HOME/data/mirflickr_train/; \
                rm /tmp/mirflickr_\$i.zip; \
                touch \$HOME/data/mirflickr_train/done_\$i; \
                echo \"Zip \$i done. Images: \$(ls \$HOME/data/mirflickr_train/*.jpg 2>/dev/null | wc -l)\"; \
            fi; \
        done; \
        echo 'MIR Flickr download complete!'; \
        echo \"Train images: \$(ls \$HOME/data/mirflickr_train/*.jpg 2>/dev/null | wc -l)\"; \
        echo \"Detect images: \$(ls \$HOME/data/mirflickr_detect/*.jpg 2>/dev/null | wc -l)\""
}

cmd_generate_all_detection_data() {
    check_gcloud
    echo "Generating detection data for all 4 models + clean..."
    echo "  Models: b30, b48, b72, b96 + clean"
    echo "  Output: ~/data/detection_shards/{b30,b48,b72,b96,clean}"
    echo ""

    gen_cmd="export PATH=\$HOME/.local/bin:\$PATH && cd ~/picode-model && \
        echo '=== Generating b30 ===' && \
        python3 scripts/generate_detection_data.py \
            --encoder \$HOME/checkpoints/picotrust_b30s16m75/best.pt \
            --model-label b30 \
            --positive-ratio 1.0 \
            --data-dir \$HOME/data/mirflickr_detect \
            --output-dir \$HOME/data/detection_shards/b30 \
            --sobel-sigma 5.0 --sobel-floor 0.75 \
            --strengths '0.010,0.012,0.014,0.016' \
            --shard-size 500 && \
        echo '=== Generating b48 ===' && \
        python3 scripts/generate_detection_data.py \
            --encoder \$HOME/checkpoints/picotrust_b48s18m80/best.pt \
            --model-label b48 \
            --positive-ratio 1.0 \
            --data-dir \$HOME/data/mirflickr_detect \
            --output-dir \$HOME/data/detection_shards/b48 \
            --sobel-sigma 5.0 --sobel-floor 0.80 \
            --strengths '0.012,0.014,0.016,0.018' \
            --shard-size 500 && \
        echo '=== Generating b72 ===' && \
        python3 scripts/generate_detection_data.py \
            --encoder \$HOME/checkpoints/picotrust_b72s20m85/best.pt \
            --model-label b72 \
            --positive-ratio 1.0 \
            --data-dir \$HOME/data/mirflickr_detect \
            --output-dir \$HOME/data/detection_shards/b72 \
            --sobel-sigma 5.0 --sobel-floor 0.85 \
            --strengths '0.014,0.016,0.018,0.020' \
            --shard-size 500 && \
        echo '=== Generating b96 ===' && \
        python3 scripts/generate_detection_data.py \
            --encoder \$HOME/checkpoints/picotrust_b96s25m90/best.pt \
            --model-label b96 \
            --positive-ratio 1.0 \
            --data-dir \$HOME/data/mirflickr_detect \
            --output-dir \$HOME/data/detection_shards/b96 \
            --sobel-sigma 5.0 --sobel-floor 0.90 \
            --strengths '0.018,0.020,0.023,0.025' \
            --shard-size 500 && \
        echo '=== Generating clean ===' && \
        python3 scripts/generate_detection_data.py \
            --encoder \$HOME/checkpoints/picotrust_b72s20m85/best.pt \
            --model-label clean \
            --positive-ratio 0.0 \
            --data-dir \$HOME/data/mirflickr_detect \
            --output-dir \$HOME/data/detection_shards/clean \
            --sobel-sigma 5.0 --sobel-floor 0.85 \
            --strengths '0.014,0.016,0.018,0.020' \
            --shard-size 500"

    echo "Launching all detection data generation in tmux session '$TMUX_SESSION'..."
    vm_ssh "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; \
        tmux new-session -d -s $TMUX_SESSION \"$gen_cmd; echo '=== Generation finished (exit code: '\$'?) ==='; read\""

    echo ""
    echo "Detection data generation launched in background tmux session."
    echo ""
    echo "Useful commands:"
    echo "  ./scripts/gce_setup.sh logs       # tail output"
    echo "  ./scripts/gce_setup.sh attach     # attach to tmux session"
}

cmd_train_classifier() {
    check_gcloud
    epochs="${1:-50}"
    batch_size="${2:-32}"

    echo "Training Strategy B model classifier..."
    echo "  Shard dirs: ~/data/detection_shards/{b30,b48,b72,b96}"
    echo "  Output:     ~/checkpoints/model_classifier"
    echo "  Epochs:     $epochs"
    echo "  Batch size: $batch_size"
    echo ""

    train_cmd="export PATH=\$HOME/.local/bin:\$PATH && cd ~/picode-model && python3 scripts/train_classifier.py \
        --shard-dirs \$HOME/data/detection_shards/b30 \$HOME/data/detection_shards/b48 \$HOME/data/detection_shards/b72 \$HOME/data/detection_shards/b96 \
        --model-labels b30 b48 b72 b96 \
        --output-dir \$HOME/checkpoints/model_classifier \
        --epochs $epochs \
        --batch-size $batch_size \
        --num-workers 0 \
        --grad-clip 1.0"

    echo "Launching classifier training in tmux session '$TMUX_SESSION'..."
    vm_ssh "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; \
        tmux new-session -d -s $TMUX_SESSION \"$train_cmd; echo '=== Training finished (exit code: '\$'?) ==='; read\""

    echo ""
    echo "Classifier training launched in background tmux session."
    echo ""
    echo "Useful commands:"
    echo "  ./scripts/gce_setup.sh logs       # tail training output"
    echo "  ./scripts/gce_setup.sh attach     # attach to tmux session"
    echo "  ./scripts/gce_setup.sh list       # list checkpoints"
    echo "  ./scripts/gce_setup.sh stop       # stop VM (pause billing)"
}

cmd_train() {
    check_gcloud
    resume=""
    config="configs/picotrust_v20.yaml"

    # Parse args: optional config file, optional --resume flag (in any order)
    for arg in "$@"; do
        if [ "$arg" = "--resume" ]; then
            resume="--resume"
        elif [[ "$arg" == *.yaml || "$arg" == *.yml ]]; then
            config="$arg"
        fi
    done

    if [ -n "$resume" ]; then
        echo "Finding latest checkpoint on VM..."
        # Find latest checkpoint file on VM
        latest=$(vm_ssh "ls -t ~/checkpoints/*/checkpoint_*.pt 2>/dev/null | head -1" || true)
        if [ -z "$latest" ]; then
            echo "Error: No checkpoint found to resume from."
            echo "Start fresh with: ./scripts/gce_setup.sh train"
            exit 1
        fi
        resume="--resume $latest"
        echo "Resuming from: $latest"
    else
        echo "Starting training..."
    fi

    echo "  Config: $config"

    # Build the training command with overrides for GCE paths
    # CLI: picode-train CONFIG [--resume PATH] [key=value overrides...]
    # Note: \$HOME is escaped so it expands on the VM, not locally
    train_cmd="export PATH=\$HOME/.local/bin:\$PATH && cd ~/picode-model && picode-train \
        $config \
        $resume \
        checkpoint.dir=\$HOME/checkpoints \
        data.path=\$HOME/data/mirflickr_train \
        logging.tensorboard_dir=\$HOME/checkpoints/runs"

    echo "Launching training in tmux session '$TMUX_SESSION'..."
    vm_ssh "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; \
        tmux new-session -d -s $TMUX_SESSION \"$train_cmd; echo '=== Training finished (exit code: '\$'?) ==='; read\""

    echo ""
    echo "Training launched in background tmux session."
    echo ""
    echo "Useful commands:"
    echo "  ./scripts/gce_setup.sh logs       # tail training output"
    echo "  ./scripts/gce_setup.sh attach     # attach to tmux session"
    echo "  ./scripts/gce_setup.sh list       # list checkpoints"
    echo "  ./scripts/gce_setup.sh stop       # stop VM (pause billing)"
}

cmd_upload_ckpt() {
    check_gcloud
    local_path="$1"
    if [ -z "$local_path" ]; then
        echo "Error: Please provide local checkpoint path"
        echo "Usage: ./scripts/gce_setup.sh upload-ckpt <path-to-checkpoint.pt>"
        exit 1
    fi

    if [ ! -f "$local_path" ]; then
        echo "Error: File does not exist: $local_path"
        exit 1
    fi

    # Preserve directory structure relative to checkpoints/
    # e.g. kaggle_ckpts/checkpoints/picotrust_v10_s010/best.pt -> ~/checkpoints/picotrust_v10_s010/best.pt
    local basename
    basename=$(basename "$(dirname "$local_path")")
    echo "Uploading checkpoint to VM:~/checkpoints/$basename/ ..."
    vm_ssh "mkdir -p ~/checkpoints/$basename"
    gcloud compute scp "$local_path" "$VM_NAME:~/checkpoints/$basename/" --zone="$ZONE"
    echo "Upload complete!"
    vm_ssh "ls -lh ~/checkpoints/$basename/"
}

cmd_generate_detection_data() {
    check_gcloud
    encoder_ckpt="${1:-picotrust_b7238s20p00d00_sobel/best.pt}"

    echo "Generating detection training data..."
    echo "  Encoder: ~/checkpoints/$encoder_ckpt"
    echo "  Output:  ~/data/detection_b72s20m85"
    echo ""

    gen_cmd="export PATH=\$HOME/.local/bin:\$PATH && cd ~/picode-model && python3 scripts/generate_detection_data.py \
        --encoder \$HOME/checkpoints/$encoder_ckpt \
        --data-dir \$HOME/data/train \
        --output-dir \$HOME/data/detection_b72s20m85 \
        --sobel-sigma 5.0 --sobel-floor 0.85 \
        --strengths '0.010,0.012,0.015,0.020' \
        --shard-size 500"

    echo "Launching data generation in tmux session '$TMUX_SESSION'..."
    vm_ssh "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; \
        tmux new-session -d -s $TMUX_SESSION \"$gen_cmd; echo '=== Generation finished (exit code: '\$'?) ==='; read\""

    echo ""
    echo "Data generation launched. Monitor with:"
    echo "  ./scripts/gce_setup.sh logs"
}

cmd_train_detector() {
    check_gcloud
    epochs="${1:-50}"
    batch_size="${2:-32}"

    echo "Training FastDetector (multi-model, pre-generated data)..."
    echo "  Data:       ~/data/detection_shards/{b30,b48,b72,b96,clean}"
    echo "  Output:     ~/checkpoints/detector_multimodel"
    echo "  Epochs:     $epochs"
    echo "  Batch size: $batch_size"
    echo ""

    train_cmd="export PATH=\$HOME/.local/bin:\$PATH && cd ~/picode-model && python3 scripts/train_detector.py \
        --pregenerated-dir \$HOME/data/detection_shards/b30 \$HOME/data/detection_shards/b48 \$HOME/data/detection_shards/b72 \$HOME/data/detection_shards/b96 \$HOME/data/detection_shards/clean \
        --output-dir \$HOME/checkpoints/detector_multimodel \
        --epochs $epochs \
        --batch-size $batch_size \
        --num-workers 0 \
        --grad-clip 1.0"

    echo "Launching detector training in tmux session '$TMUX_SESSION'..."
    vm_ssh "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; \
        tmux new-session -d -s $TMUX_SESSION \"$train_cmd; echo '=== Training finished (exit code: '\$'?) ==='; read\""

    echo ""
    echo "Detector training launched in background tmux session."
    echo ""
    echo "Useful commands:"
    echo "  ./scripts/gce_setup.sh logs       # tail training output"
    echo "  ./scripts/gce_setup.sh attach     # attach to tmux session"
    echo "  ./scripts/gce_setup.sh list       # list checkpoints"
    echo "  ./scripts/gce_setup.sh stop       # stop VM (pause billing)"
}

cmd_logs() {
    check_gcloud
    echo "Tailing training output (Ctrl+C to stop)..."
    echo ""
    # Capture tmux pane content and follow
    vm_ssh_tty "tmux capture-pane -t $TMUX_SESSION -p -S -100 2>/dev/null || echo 'No training session found.'"
}

cmd_attach() {
    check_gcloud
    echo "Attaching to tmux session (Ctrl+B, D to detach)..."
    vm_ssh_tty "tmux attach-session -t $TMUX_SESSION 2>/dev/null || echo 'No training session found.'"
}

cmd_list() {
    check_gcloud
    echo "Listing checkpoints on VM..."
    vm_ssh "if [ -d ~/checkpoints ]; then \
        find ~/checkpoints -name '*.pt' -exec ls -lh {} \; 2>/dev/null | sort; \
    else \
        echo 'No checkpoints directory found.'; \
    fi"
}

cmd_download() {
    check_gcloud
    local_path="${1:-./checkpoints_gce}"
    mkdir -p "$local_path"

    echo "Downloading checkpoints to $local_path..."
    gcloud compute scp --recurse "$VM_NAME:~/checkpoints/" "$local_path/" --zone="$ZONE"

    echo ""
    echo "Download complete!"
    ls -lh "$local_path/"
}

cmd_ssh() {
    check_gcloud
    gcloud compute ssh "$VM_NAME" --zone="$ZONE"
}

cmd_stop() {
    check_gcloud
    echo "Stopping VM $VM_NAME (disk preserved, GPU billing stops)..."
    gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
    echo "VM stopped. Restart with: ./scripts/gce_setup.sh start"
}

cmd_start() {
    check_gcloud
    echo "Starting VM $VM_NAME..."
    gcloud compute instances start "$VM_NAME" --zone="$ZONE"
    echo ""
    echo "VM started! Wait ~30 seconds for SSH, then:"
    echo "  ./scripts/gce_setup.sh train --resume"
}

cmd_status() {
    check_gcloud
    gcloud compute instances describe "$VM_NAME" --zone="$ZONE" \
        --format="table(name, status, machineType.basename(), \
        scheduling.preemptible, networkInterfaces[0].accessConfigs[0].natIP)"
}

cmd_delete() {
    check_gcloud
    echo "This will permanently delete VM $VM_NAME and its boot disk."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE"
        echo "VM deleted."
    else
        echo "Cancelled."
    fi
}

# --- Main ---

case "${1:-}" in
    create)      cmd_create ;;
    setup)       cmd_setup ;;
    upload-data) shift; cmd_upload_data "$@" ;;
    upload-code) cmd_upload_code ;;
    setup-data)  cmd_setup_data ;;
    train)       shift; cmd_train "$@" ;;
    generate-detection-data) shift; cmd_generate_detection_data "$@" ;;
    setup-mirflickr) cmd_setup_mirflickr ;;
    generate-all-detection-data) cmd_generate_all_detection_data ;;
    train-classifier) shift; cmd_train_classifier "$@" ;;
    train-detector) shift; cmd_train_detector "$@" ;;
    upload-ckpt) shift; cmd_upload_ckpt "$@" ;;
    logs)        cmd_logs ;;
    attach)      cmd_attach ;;
    list)        cmd_list ;;
    download)    shift; cmd_download "$@" ;;
    ssh)         cmd_ssh ;;
    stop)        cmd_stop ;;
    start)       cmd_start ;;
    status)      cmd_status ;;
    delete)      cmd_delete ;;
    *)           usage; exit 1 ;;
esac
