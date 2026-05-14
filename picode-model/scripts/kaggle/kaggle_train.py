"""Kaggle kernel training script for Picode models.

This script runs on Kaggle with GPU enabled. It:
1. Installs missing pip dependencies (torch/torchvision/opencv are pre-installed)
2. Adds the picode source dataset to sys.path
3. Auto-detects COCO data path under /kaggle/input/
4. Optionally resumes from a checkpoint dataset
5. Runs Trainer.fit()
6. Zips checkpoints to /kaggle/working/ for download

Set MODEL_NAME env var to select model (default: picodelite).
Supported: picodelite, picodeframe
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

# --- Model selection ---
MODEL_NAME = os.environ.get("MODEL_NAME", "picodelite")
print(f"Model: {MODEL_NAME}")

# Map model names to config files and checkpoint dataset names
MODEL_CONFIGS = {
    "picodelite": "picodelite_kaggle.yaml",
    "picodeframe": "picodeframe_kaggle.yaml",
}
CKPT_DATASET_NAME = f"{MODEL_NAME}-checkpoints"

if MODEL_NAME not in MODEL_CONFIGS:
    raise ValueError(f"Unknown model: {MODEL_NAME}. Supported: {list(MODEL_CONFIGS.keys())}")

# --- Step 1: Install missing dependencies ---
print("Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "lpips", "pyldpc", "galois", "kornia",
])

# --- Step 2: Add picode source to sys.path ---
PICODE_SOURCE = "/kaggle/input/picode-source/picode-model"
if not os.path.isdir(PICODE_SOURCE):
    # Try without subfolder (depends on dataset structure)
    alt = "/kaggle/input/picode-source"
    if os.path.isdir(os.path.join(alt, "picode")):
        PICODE_SOURCE = alt
    else:
        raise FileNotFoundError(
            f"picode source not found at {PICODE_SOURCE} or {alt}. "
            "Check that the picode-source dataset was uploaded correctly."
        )

sys.path.insert(0, PICODE_SOURCE)
print(f"picode source: {PICODE_SOURCE}")

# Verify import works
from picode.training.trainer import Trainer  # noqa: E402
from picode.training.cli import parse_overrides  # noqa: E402

# --- Step 3: Auto-detect COCO data path ---
COCO_SEARCH_PATTERNS = [
    "/kaggle/input/coco-2017-dataset/coco2017/train2017",
    "/kaggle/input/coco2017/train2017",
    "/kaggle/input/coco-2017/train2017",
    "/kaggle/input/*/train2017",
]


def find_coco_path() -> str:
    """Find COCO training images under /kaggle/input/."""
    for pattern in COCO_SEARCH_PATTERNS:
        matches = glob.glob(pattern)
        for m in matches:
            if os.path.isdir(m):
                num_files = len(os.listdir(m))
                if num_files > 1000:
                    print(f"Found COCO data: {m} ({num_files} images)")
                    return m
    # List what's available for debugging
    print("Available datasets in /kaggle/input/:")
    for d in sorted(os.listdir("/kaggle/input/")):
        print(f"  {d}/")
        subdir = os.path.join("/kaggle/input", d)
        if os.path.isdir(subdir):
            for sd in sorted(os.listdir(subdir))[:5]:
                print(f"    {sd}")
    raise FileNotFoundError(
        "COCO train2017 not found. Add a COCO 2017 dataset to your kernel sources."
    )


coco_path = find_coco_path()

# --- Step 4: Find resume checkpoint ---
CHECKPOINT_SEARCH_PATTERNS = [
    f"/kaggle/input/{CKPT_DATASET_NAME}/checkpoints/{MODEL_NAME}/checkpoint_*.pt",
    f"/kaggle/input/{CKPT_DATASET_NAME}/{MODEL_NAME}/checkpoint_*.pt",
    f"/kaggle/input/{CKPT_DATASET_NAME}/checkpoint_*.pt",
    f"/kaggle/input/{CKPT_DATASET_NAME}/**/*.pt",
]

resume_checkpoint = None
for pattern in CHECKPOINT_SEARCH_PATTERNS:
    matches = sorted(glob.glob(pattern, recursive=True))
    # Filter to periodic checkpoints (not best.pt)
    periodic = [m for m in matches if "checkpoint_" in os.path.basename(m)]
    if periodic:
        resume_checkpoint = periodic[-1]
        break

if resume_checkpoint:
    print(f"Will resume from: {resume_checkpoint}")
else:
    print("No resume checkpoint found, starting fresh")

# --- Step 5: Create Trainer and run fit() ---
CONFIG_PATH = os.path.join(PICODE_SOURCE, "configs", MODEL_CONFIGS[MODEL_NAME])
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

overrides = parse_overrides([
    f"data.path={coco_path}",
    f"checkpoint.dir={CHECKPOINT_DIR}",
])

try:
    if resume_checkpoint:
        print(f"\nResuming training from {resume_checkpoint}")
        trainer = Trainer.from_checkpoint(resume_checkpoint)
        # Override checkpoint dir to write to /kaggle/working/
        trainer.checkpointer.dir = __import__("pathlib").Path(CHECKPOINT_DIR) / trainer.config.experiment_name
        trainer.checkpointer.dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"\nStarting fresh training with config: {CONFIG_PATH}")
        trainer = Trainer.from_config(CONFIG_PATH, overrides)

    print(f"Experiment: {trainer.config.experiment_name}")
    print(f"Device: {trainer.device}")
    print(f"Current step: {trainer.global_step}")
    print(f"Target steps: {trainer.config.training.num_steps}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print()

    trainer.fit()

finally:
    # --- Step 6: Zip checkpoints for download ---
    print("\nZipping checkpoints...")
    zip_path = "/kaggle/working/checkpoints.zip"
    if os.path.isdir(CHECKPOINT_DIR):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(CHECKPOINT_DIR):
                for f in files:
                    filepath = os.path.join(root, f)
                    arcname = os.path.relpath(filepath, "/kaggle/working")
                    zf.write(filepath, arcname)
        size_mb = os.path.getsize(zip_path) / 1e6
        print(f"Saved {zip_path} ({size_mb:.1f} MB)")
    else:
        print("No checkpoints directory found to zip")

print("\nDone!")
