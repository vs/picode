"""Kaggle kernel training script for Picode models.

This script runs on Kaggle with GPU enabled. It:
1. Installs missing pip dependencies (torch/torchvision/opencv are pre-installed)
2. Adds the picode source dataset to sys.path
3. Auto-detects COCO data path under /kaggle/input/
4. Optionally resumes from a checkpoint dataset
5. Runs Trainer.fit()
6. Zips checkpoints to /kaggle/working/ for download

Set MODEL_NAME env var to select model (default: picodelite).
Supported: picodelite, picodelite_256bit, picodeframe
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

# --- Model selection ---
MODEL_NAME = os.environ.get("MODEL_NAME", "picodeframe")
print(f"Model: {MODEL_NAME}")

# Map model names to config files and checkpoint dataset names
MODEL_CONFIGS = {
    "picodelite": "picodelite_kaggle.yaml",
    "picodelite_256bit": "picodelite_256bit_kaggle.yaml",
    "picodeframe": "picodeframe_kaggle.yaml",
    "picotrust_v9b": "picotrust_v9b.yaml",
    "picotrust_v10": "picotrust_v10.yaml",
    "picotrust_v10_s012": "picotrust_v10_s012.yaml",
    "picotrust_v10_s010": "picotrust_v10_s010.yaml",
    "picotrust_v22": "picotrust_v22.yaml",
    "picotrust_b7238s20p03d00": "picotrust_b7238s20p03d00.yaml",
}
# Kaggle normalizes dataset slugs: underscores become hyphens
CKPT_DATASET_NAME = f"{MODEL_NAME}-checkpoints"
CKPT_DATASET_SLUG = CKPT_DATASET_NAME.replace("_", "-")

if MODEL_NAME not in MODEL_CONFIGS:
    raise ValueError(f"Unknown model: {MODEL_NAME}. Supported: {list(MODEL_CONFIGS.keys())}")

# --- Step 1: Check GPU compatibility ---
# P100 (sm_60) is incompatible with PyTorch 2.2+ (requires sm_70+).
# No fix available on Python 3.12 (PyTorch 2.1.x only supports Python <= 3.11).
# Exit early so the kernel finishes quickly and user can re-push for a T4.
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} (sm_{cap[0]}{cap[1]})")
    if cap[0] < 7:
        print(f"\nERROR: {gpu_name} (sm_{cap[0]}{cap[1]}) is not supported by PyTorch {torch.__version__}.")
        print("PyTorch 2.2+ requires sm_70 or higher. Re-push the kernel to get a T4 GPU.")
        sys.exit(1)

print("Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "lpips", "pyldpc", "galois", "kornia",
])

# --- Step 2: Add picode source to sys.path ---
# Kaggle dataset structure varies depending on upload method (--dir-mode zip/tar/skip).
# Search common layouts to find the picode package.
PICODE_SOURCE = None
SEARCH_PATHS = [
    "/kaggle/input/picode-source/picode-model",
    "/kaggle/input/picode-source",
    # --dir-mode zip can nest inside an extra directory
    "/kaggle/input/picode-source/picode-model/picode-model",
]

# Search all datasets under /kaggle/input/ (handles slug variations)
if os.path.isdir("/kaggle/input"):
    for dataset in os.listdir("/kaggle/input"):
        ds_path = os.path.join("/kaggle/input", dataset)
        if not os.path.isdir(ds_path):
            continue
        SEARCH_PATHS.append(ds_path)
        for entry in os.listdir(ds_path):
            candidate = os.path.join(ds_path, entry)
            if os.path.isdir(candidate):
                SEARCH_PATHS.append(candidate)
                # One more level for double nesting
                for sub in os.listdir(candidate):
                    sub_path = os.path.join(candidate, sub)
                    if os.path.isdir(sub_path):
                        SEARCH_PATHS.append(sub_path)

for path in dict.fromkeys(SEARCH_PATHS):  # deduplicate, preserve order
    if os.path.isdir(path) and os.path.isdir(os.path.join(path, "picode")):
        PICODE_SOURCE = path
        break

if PICODE_SOURCE is None:
    # Print all available datasets for debugging
    print("ERROR: Could not find picode package.")
    print()
    print("Available datasets in /kaggle/input/:")
    if os.path.isdir("/kaggle/input"):
        for d in sorted(os.listdir("/kaggle/input")):
            full = os.path.join("/kaggle/input", d)
            if os.path.isdir(full):
                contents = os.listdir(full)
                print(f"  {d}/ ({len(contents)} items)")
                for item in sorted(contents)[:10]:
                    item_path = os.path.join(full, item)
                    if os.path.isdir(item_path):
                        sub_contents = os.listdir(item_path)
                        print(f"    {item}/ ({len(sub_contents)} items)")
                        for sub in sorted(sub_contents)[:5]:
                            print(f"      {sub}")
                    else:
                        size = os.path.getsize(item_path)
                        print(f"    {item} ({size / 1e6:.1f} MB)")
            else:
                print(f"  {d} (file)")
    else:
        print("  /kaggle/input does not exist!")
    print()
    raise FileNotFoundError(
        "picode source not found. See directory listing above."
    )

sys.path.insert(0, PICODE_SOURCE)
print(f"picode source: {PICODE_SOURCE}")

# Verify import works
from picode.training.trainer import Trainer  # noqa: E402
from picode.training.cli import parse_overrides  # noqa: E402

# --- Step 3: Auto-detect COCO data path ---
def find_coco_path() -> str:
    """Find COCO training images under /kaggle/input/.

    Searches recursively for a 'train2017' directory with >1000 images.
    Handles both old (/kaggle/input/<slug>/) and new
    (/kaggle/input/datasets/<user>/<slug>/) Kaggle mount layouts.
    """
    for pattern in glob.glob("/kaggle/input/**/train2017", recursive=True):
        if os.path.isdir(pattern):
            num_files = len(os.listdir(pattern))
            if num_files > 1000:
                print(f"Found COCO data: {pattern} ({num_files} images)")
                return pattern
    raise FileNotFoundError(
        "COCO train2017 not found. Add a COCO 2017 dataset to your kernel sources."
    )


coco_path = find_coco_path()

# --- Step 4: Find resume checkpoint ---
# Search for checkpoints from the dedicated checkpoint dataset only (not source dataset).
# Filter to current model name to avoid loading wrong model's checkpoints.
resume_checkpoint = None
# Try both underscore and hyphenated forms (Kaggle normalizes slugs)
for ckpt_pattern in [CKPT_DATASET_NAME, CKPT_DATASET_SLUG]:
    all_ckpts = sorted(glob.glob(
        f"/kaggle/input/**/{ckpt_pattern}/**/checkpoint_*.pt", recursive=True
    ))
    if not all_ckpts:
        # Also try flat layout (checkpoint directly in dataset root)
        all_ckpts = sorted(glob.glob(
            f"/kaggle/input/**/{ckpt_pattern}/checkpoint_*.pt", recursive=True
        ))
    if all_ckpts:
        break
if not all_ckpts:
    # Fallback: search by model name under any dataset
    all_ckpts = sorted(glob.glob(
        f"/kaggle/input/**/{MODEL_NAME}/checkpoint_*.pt", recursive=True
    ))
    # Filter out anything from picode-source (avoid stale checkpoints in source upload)
    all_ckpts = [c for c in all_ckpts if "picode-source" not in c]
if all_ckpts:
    resume_checkpoint = all_ckpts[-1]

if resume_checkpoint:
    print(f"Will resume from: {resume_checkpoint}")
else:
    print("No resume checkpoint found, starting fresh")

# --- Step 5: Create Trainer and run fit() ---
# Search for config file — may be at PICODE_SOURCE/configs/ or PICODE_SOURCE/../configs/
CONFIG_NAME = MODEL_CONFIGS[MODEL_NAME]
CONFIG_PATH = None
for candidate in [
    os.path.join(PICODE_SOURCE, "configs", CONFIG_NAME),
    os.path.join(os.path.dirname(PICODE_SOURCE), "configs", CONFIG_NAME),
    os.path.join(PICODE_SOURCE, "picode-model", "configs", CONFIG_NAME),
]:
    if os.path.isfile(candidate):
        CONFIG_PATH = candidate
        break

# Fallback: recursive search
if CONFIG_PATH is None:
    matches = glob.glob(f"/kaggle/input/**/{CONFIG_NAME}", recursive=True)
    if matches:
        CONFIG_PATH = matches[0]

if CONFIG_PATH is None:
    raise FileNotFoundError(
        f"Config {CONFIG_NAME} not found. Searched under {PICODE_SOURCE}/configs/ "
        f"and /kaggle/input/**/{CONFIG_NAME}"
    )

print(f"Config: {CONFIG_PATH}")
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

overrides = parse_overrides([
    f"data.path={coco_path}",
    f"checkpoint.dir={CHECKPOINT_DIR}",
])

try:
    if resume_checkpoint:
        print(f"\nResuming training from {resume_checkpoint}")
        trainer = Trainer.from_checkpoint(
            resume_checkpoint, config_path=CONFIG_PATH, overrides=overrides,
        )
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
    # Log critical config values for verification
    print(f"message_loss_type: {trainer.config.loss.message_loss_type}")
    if trainer.config.frame is not None:
        print(f"frame_l2_scale: {trainer.config.frame.frame_l2_scale}")
        print(f"frame_lpips_scale: {trainer.config.frame.frame_lpips_scale}")
        print(f"frame_color_scale: {trainer.config.frame.frame_color_scale}")
        print(f"residual_max_amplitude: {trainer.config.frame.residual_max_amplitude}")
    # Verify encoder has the amplitude cap
    if hasattr(trainer.encoder, 'max_residual_amplitude'):
        print(f"encoder.max_residual_amplitude: {trainer.encoder.max_residual_amplitude}")
    else:
        print("WARNING: encoder has no max_residual_amplitude attribute (old code?)")
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
