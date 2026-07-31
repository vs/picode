"""Modal deployment for Picode training.

Usage:
    # First time setup
    pip install modal
    modal token new

    # Upload training data (one time)
    modal run modal_train.py::upload_data --local-path ./data/train

    # Run training
    modal run modal_train.py::train --config configs/stegastamp_baseline.yaml

    # Run training with overrides
    modal run modal_train.py::train --config configs/stegastamp_baseline.yaml --override training.lr=0.0002

    # Resume from checkpoint
    modal run modal_train.py::train --config configs/stegastamp_baseline.yaml --resume

    # Download checkpoints locally
    modal run modal_train.py::download_checkpoints --local-path ./checkpoints_remote

    # Interactive shell for debugging
    modal shell modal_train.py
"""

import modal

# Create Modal app
app = modal.App("picode-training")

# Persistent volumes for data and checkpoints
data_volume = modal.Volume.from_name("picode-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("picode-checkpoints", create_if_missing=True)

# Container image with all dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2.0", "scipy")  # pyldpc build deps
    .pip_install(
        "torch>=2.0",
        "torchvision>=0.15",
        "torchmetrics>=1.0",
        "click>=8.0",
        "pillow>=10.0",
        "pyldpc>=0.7.9",
        "galois>=0.4.0",
        "pyyaml>=6.0",
        "lpips>=0.1",
        "kornia>=0.7.0",
        "tensorboard>=2.0",
        "opencv-python-headless>=4.0",
    )
    .add_local_dir(".", remote_path="/root", copy=True, ignore=["data/", "checkpoints*/", "kaggle_*/", "*.pyc", "__pycache__", ".git", "venv/", ".venv/", "runs/", "*.pt", "notebooks/", "picode-model/"])
    .run_commands("cd /root && pip install -e .")
)

# Volume mount paths
DATA_PATH = "/data"
CHECKPOINT_PATH = "/checkpoints"


@app.function(
    image=image,
    gpu="A10G",  # A10G 24GB ($1.10/hr). Options: T4, A10G, A100, H100
    timeout=3600 * 24,  # 24 hour max — 140k steps with compositing plus the
    # print-to-photo chain runs 8-10h, close enough to a 12h ceiling to risk
    # a mid-run kill. Only actual runtime is billed, so a higher cap is free.
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
)
def train(
    config: str = "configs/stegastamp_baseline.yaml",
    override: str = "",
    resume: bool = False,
    resume_from: str = "",
):
    """Run training on Modal GPU.

    Args:
        config: Path to config file (relative to repo root)
        override: Comma-separated key=value overrides (e.g., "training.lr=0.0002,data.batch_size=8")
        resume: Whether to resume from latest checkpoint
        resume_from: Specific checkpoint filename to resume from (e.g., "checkpoint_00020000.pt")
    """
    import os
    import sys

    # Add repo to path (mounted at /root)
    sys.path.insert(0, "/root")
    os.chdir("/root")

    # Import after path setup
    from picode.training.cli import parse_overrides
    from picode.training.trainer import Trainer

    # Build overrides dict - override checkpoint/logging paths (data path from config)
    override_list = [o.strip() for o in override.split(",") if o.strip()]
    override_list.extend([
        f"checkpoint.dir={CHECKPOINT_PATH}",
        f"logging.tensorboard_dir={CHECKPOINT_PATH}/runs",
    ])
    override_dict = parse_overrides(override_list)

    print(f"Starting training with config: {config}")
    print(f"Overrides: {override_dict}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Find checkpoint if resuming
    if resume_from:
        # Resume from specific checkpoint with NEW config
        import glob
        matches = glob.glob(f"{CHECKPOINT_PATH}/*/{resume_from}")
        if matches:
            checkpoint_path = matches[0]
            print(f"Resuming from specific checkpoint: {checkpoint_path}")
            print(f"Using config: {config}")
            trainer = Trainer.from_checkpoint(
                checkpoint_path, config_path=config, overrides=override_dict,
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
    elif resume:
        import glob
        checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/*/checkpoint_*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Resuming from: {latest}")
            trainer = Trainer.from_checkpoint(latest, config_path=config, overrides=override_dict)
        else:
            print("No checkpoint found, starting fresh")
            trainer = Trainer.from_config(config, override_dict)
    else:
        trainer = Trainer.from_config(config, override_dict)

    print(f"\nExperiment: {trainer.config.experiment_name}")
    print(f"Device: {trainer.device}")
    print(f"Steps: {trainer.config.training.num_steps}")
    print(f"LR: {trainer.config.training.lr}")
    print()

    # Monkey-patch _train_step to detect bootstrap collapse early.
    #
    # The check must land AFTER the no-image-loss phase. During that phase the
    # encoder trains with no L2 restraint, so a low decoder_prob_std is expected
    # rather than diagnostic — Trainer.fit's own collapse detector refuses to
    # judge before max(warmup_steps, no_im_loss_steps) for exactly this reason.
    # A hardcoded step 2000 killed healthy PicoTrust runs, whose standard recipe
    # uses no_im_loss_steps=10000. Mirror the trainer's gating, keeping the
    # original step 2000 as a floor so configs with a short warmup (e.g.
    # no_im_loss_steps=1000) still fail fast.
    _training = trainer.config.training
    _bootstrap_check_step = max(
        2000,
        max(_training.warmup_steps, _training.no_im_loss_steps) + 1000,
    )
    _original_train_step = trainer._train_step
    _bootstrap_min_prob_std = 0.02
    print(f"Bootstrap collapse check at step {_bootstrap_check_step} "
          f"(no_im_loss_steps={_training.no_im_loss_steps})")

    def _patched_train_step(images):
        metrics = _original_train_step(images)
        step = trainer.global_step
        if step == _bootstrap_check_step:
            prob_std = metrics.get("decoder_prob_std", 0.0)
            if prob_std < _bootstrap_min_prob_std:
                raise RuntimeError(
                    f"BOOTSTRAP COLLAPSED at step {step}: "
                    f"prob_std={prob_std:.4f} < {_bootstrap_min_prob_std}"
                )
            print(f"\nBootstrap OK at step {step}: prob_std={prob_std:.4f}")
        return metrics

    trainer._train_step = _patched_train_step

    # Train
    trainer.fit()

    # Commit volume changes
    checkpoint_volume.commit()

    print("\nTraining complete!")
    print(f"Checkpoints saved to Modal volume: picode-checkpoints")


@app.function(image=image, volumes={DATA_PATH: data_volume})
def upload_data(local_path: str):
    """Upload training data to Modal volume.

    Run from local machine:
        modal run modal_train.py::upload_data --local-path ./data/train
    """
    # This function is a placeholder - actual upload happens via modal volume put
    print(f"Data volume mounted at {DATA_PATH}")
    print("Use 'modal volume put picode-data <local-path> <remote-path>' to upload")


@app.function(image=image, volumes={CHECKPOINT_PATH: checkpoint_volume})
def download_checkpoints(local_path: str = "./checkpoints_remote"):
    """Download checkpoints from Modal volume.

    Run from local machine:
        modal run modal_train.py::download_checkpoints --local-path ./checkpoints_remote
    """
    print(f"Checkpoint volume mounted at {CHECKPOINT_PATH}")
    print("Use 'modal volume get picode-checkpoints <remote-path> <local-path>' to download")


@app.function(image=image, volumes={CHECKPOINT_PATH: checkpoint_volume})
def list_checkpoints():
    """List all checkpoints in the Modal volume."""
    import os

    print(f"Checkpoints in {CHECKPOINT_PATH}:")
    for root, dirs, files in os.walk(CHECKPOINT_PATH):
        for f in files:
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {path} ({size_mb:.1f} MB)")


@app.function(
    image=image,
    gpu="T4",
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
)
def evaluate(checkpoint_path: str, robustness: bool = False):
    """Run evaluation on a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint in volume (e.g., /checkpoints/exp/best.pt)
        robustness: Whether to run robustness sweep
    """
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    import torch
    from picode.training.trainer import Trainer

    print(f"Loading checkpoint: {checkpoint_path}")
    trainer = Trainer.from_checkpoint(checkpoint_path)

    metrics = trainer.evaluate()
    print("\nEvaluation Results:")
    print(f"  Bit Accuracy:     {metrics.bit_accuracy:.4f}")
    print(f"  Message Accuracy: {metrics.message_accuracy:.4f}")
    print(f"  PSNR:             {metrics.psnr:.2f} dB")
    print(f"  SSIM:             {metrics.ssim:.4f}")
    if metrics.lpips is not None:
        print(f"  LPIPS:            {metrics.lpips:.4f}")

    if robustness:
        print("\nRobustness Sweep:")
        images = next(iter(trainer.dataloader)).to(trainer.device)
        messages = torch.randint(
            0, 2, (images.shape[0], trainer.config.training.num_bits),
            device=trainer.device
        ).float()
        results = trainer.robustness_sweep(images, messages)

        current_dist = None
        for r in results:
            if r.distortion != current_dist:
                current_dist = r.distortion
                print(f"\n  {r.distortion}:")
            print(f"    strength={r.strength:<6} bit_acc={r.bit_accuracy:.4f}")


@app.local_entrypoint()
def main():
    """Default entrypoint - shows help."""
    print(__doc__)
