#!/bin/bash
# Run v6 de-annealing experiment: step strength from 0.012 → 0.013 → 0.014 → 0.015
# Each phase runs 30k steps, saving best checkpoint per strength level.
#
# Usage (on GCE VM):
#   cd ~/picode-model && bash scripts/run_v6_deanneal.sh
#
# Expects v5 checkpoint at ~/checkpoints/picotrust_v5/best.pt (or latest)

set -e

export PATH=$HOME/.local/bin:$PATH

# Find the v5 checkpoint to start from
V5_CKPT=$(ls -t ~/checkpoints/picotrust_v5/checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$V5_CKPT" ]; then
    V5_CKPT=~/checkpoints/picotrust_v5/best.pt
fi

if [ ! -f "$V5_CKPT" ]; then
    echo "Error: No v5 checkpoint found"
    exit 1
fi

echo "========================================="
echo "v6 De-annealing Experiment"
echo "========================================="
echo "Starting from: $V5_CKPT"
echo ""

# Phase 1: strength 0.013, steps 170k → 200k
echo "=== Phase 1/3: strength 0.013 (steps 170k-200k) ==="
picode-train configs/picotrust_v6a.yaml \
    --resume "$V5_CKPT" \
    checkpoint.dir=$HOME/checkpoints \
    data.path=$HOME/data/train \
    logging.tensorboard_dir=$HOME/checkpoints/runs

# Find latest v6a checkpoint
V6A_CKPT=$(ls -t ~/checkpoints/picotrust_v6a/checkpoint_*.pt 2>/dev/null | head -1)
echo ""
echo "Phase 1 complete. Checkpoint: $V6A_CKPT"
echo ""

# Phase 2: strength 0.014, steps 200k → 230k
echo "=== Phase 2/3: strength 0.014 (steps 200k-230k) ==="
picode-train configs/picotrust_v6b.yaml \
    --resume "$V6A_CKPT" \
    checkpoint.dir=$HOME/checkpoints \
    data.path=$HOME/data/train \
    logging.tensorboard_dir=$HOME/checkpoints/runs

# Find latest v6b checkpoint
V6B_CKPT=$(ls -t ~/checkpoints/picotrust_v6b/checkpoint_*.pt 2>/dev/null | head -1)
echo ""
echo "Phase 2 complete. Checkpoint: $V6B_CKPT"
echo ""

# Phase 3: strength 0.015, steps 230k → 260k
echo "=== Phase 3/3: strength 0.015 (steps 230k-260k) ==="
picode-train configs/picotrust_v6c.yaml \
    --resume "$V6B_CKPT" \
    checkpoint.dir=$HOME/checkpoints \
    data.path=$HOME/data/train \
    logging.tensorboard_dir=$HOME/checkpoints/runs

echo ""
echo "========================================="
echo "De-annealing experiment complete!"
echo "========================================="
echo "Checkpoints to evaluate:"
echo "  v5  (0.012): ~/checkpoints/picotrust_v5/best.pt"
echo "  v6a (0.013): ~/checkpoints/picotrust_v6a/best.pt"
echo "  v6b (0.014): ~/checkpoints/picotrust_v6b/best.pt"
echo "  v6c (0.015): ~/checkpoints/picotrust_v6c/best.pt"
