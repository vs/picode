#!/bin/bash
# One-time VM setup script for GCE Picode training.
# Uploaded and executed by `gce_setup.sh setup`.
#
# Expects a Deep Learning VM image (PyTorch + CUDA + NVIDIA drivers pre-installed).
# Installs remaining picode dependencies only.

set -e

echo "=== GCE Picode VM Setup ==="
echo ""

# --- System packages ---
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq tmux unzip 2>/dev/null

# --- Verify pre-installed GPU stack ---
echo ""
echo "NVIDIA driver status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --- PATH setup ---
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/.local/bin:$PATH

# --- Python packages ---
echo ""
echo "Upgrading pip..."
pip install -q --upgrade pip setuptools

echo "Installing picode dependencies..."

# Install numpy<2.0 and scipy first (pyldpc build deps)
pip install -q "numpy<2.0" scipy

# Install remaining picode dependencies not in the DLVM base
pip install -q \
    "torchmetrics>=1.0" \
    "click>=8.0" \
    "pillow>=10.0" \
    "pyldpc>=0.7.9" \
    "galois>=0.4.0" \
    "pyyaml>=6.0" \
    "lpips>=0.1" \
    "kornia>=0.7.0" \
    "tensorboard>=2.0" \
    "opencv-python>=4.5.0"

# --- Create directories ---
echo ""
echo "Creating directories..."
mkdir -p ~/data ~/checkpoints

# --- Verify ---
echo ""
echo "=== Setup Complete ==="
python3 -c "
import torch
print(f'Python:  {__import__(\"sys\").version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:     {torch.cuda.get_device_name(0)}')
    print(f'VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""
echo "Next: upload code with ./scripts/gce_setup.sh upload-code"
