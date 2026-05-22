#!/bin/bash
# One-time VM setup script for GCE Picode training.
# Uploaded and executed by `gce_setup.sh setup`.
#
# Installs: system deps, NVIDIA drivers, Python 3.10, PyTorch, picode deps.

set -e

echo "=== GCE Picode VM Setup ==="
echo ""

# --- System packages ---
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential tmux unzip python3-pip python3-venv gsutil 2>/dev/null

# --- NVIDIA driver ---
# The VM was created with --metadata="install-nvidia-driver=True" which uses
# the GCE startup script to install drivers. If that hasn't finished yet,
# or if using a plain Ubuntu image, install manually.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver not found. Installing..."
    sudo apt-get install -y -qq linux-headers-$(uname -r)
    # Install CUDA toolkit (includes driver)
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-4 cuda-drivers
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

echo ""
echo "NVIDIA driver status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (driver not yet ready — may need reboot)"

# --- Python packages ---
echo ""
echo "Installing Python packages..."

# Install numpy<2.0 and scipy first (pyldpc build deps)
pip install -q "numpy<2.0" scipy

# Install PyTorch with CUDA
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining picode dependencies (matching modal_train.py)
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
