"""Modal deployment for FastDetector training.

RECOMMENDED: Pre-generate dataset locally, then train on Modal (faster & cheaper):

    # 1. Generate dataset locally (CPU/MPS - free compute)
    python scripts/prepare_detection_dataset.py \
        --encoder-checkpoint checkpoints/best.pt \
        --image-dir data/coco/train2017 \
        --output-dir data/detection_dataset \
        --num-samples 50000

    # 2. Upload to Modal volume
    modal volume put picode-data data/detection_dataset /detection_dataset

    # 3. Train on Modal with pre-generated data (fast!)
    modal run scripts/modal_train_detector.py::train_detector_pregenerated --epochs 50

ALTERNATIVE: Generate data on-the-fly (slower, uses GPU for encoder):

    # Run detector training with on-the-fly data generation
    modal run scripts/modal_train_detector.py::train_detector --epochs 10

Download trained detector:

    modal volume get picode-checkpoints detection/ ./checkpoints/detection
"""

import modal

# Create Modal app
app = modal.App("picode-detector-training")

# Persistent volumes for data and checkpoints
data_volume = modal.Volume.from_name("picode-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("picode-checkpoints", create_if_missing=True)

# Container image with all dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "libgl1-mesa-glx", "libglib2.0-0")  # OpenCV deps
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
    )
    .add_local_dir(
        ".",
        remote_path="/root",
        copy=True,
        ignore=["data/", "checkpoints/", "*.pyc", "__pycache__", ".git", "venv/", ".venv/"],
    )
    .run_commands("cd /root && pip install -e .")
)

# Volume mount paths
DATA_PATH = "/data"
CHECKPOINT_PATH = "/checkpoints"


@app.function(
    image=image,
    gpu="T4",  # T4 is cheapest ($0.59/hr). Options: T4, A10G, A100, H100
    timeout=3600 * 6,  # 6 hour max
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
)
def train_detector(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.0001,
    encoder_checkpoint: str = "stegastamp_original/best.pt",
):
    """Train FastDetector on Modal GPU.

    Args:
        epochs: Number of training epochs (default 10 for testing)
        batch_size: Batch size (default 16)
        lr: Learning rate (default 0.0001)
        encoder_checkpoint: Path to encoder checkpoint in volume (relative to /checkpoints)
    """
    import os
    import sys

    # Add repo to path (mounted at /root)
    sys.path.insert(0, "/root")
    os.chdir("/root")

    import torch
    from torch.utils.data import DataLoader, random_split

    from picode.detection.fast_detector import FastDetectorModel
    from picode.detection.training import (
        DetectionAugmentation,
        DetectionDataset,
        DetectionLoss,
        DetectionTrainer,
    )
    from picode.models.factory import create_encoder
    from picode.training.config import ModelConfig

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load encoder on GPU for faster data generation
    # Use num_workers=0 in DataLoader to keep everything in main process
    encoder_path = f"{CHECKPOINT_PATH}/{encoder_checkpoint}"
    print(f"Loading encoder from {encoder_path}...")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")
    encoder_size = model_cfg.get("encoder_size", 400)
    training_cfg = config.get("training", {})
    num_bits = training_cfg.get("num_bits", 100)

    # Compute strength for picotrust models
    residual_strength = training_cfg.get("residual_strength", 0)
    strength = None
    if residual_strength > 0:
        step = ckpt.get("step", 0)
        anneal_target = training_cfg.get("residual_strength_anneal_target", residual_strength)
        anneal_start = training_cfg.get("residual_strength_anneal_start", 0)
        anneal_steps = training_cfg.get("residual_strength_anneal_steps", 1)
        if step >= anneal_start and anneal_steps > 0:
            t = min((step - anneal_start) / anneal_steps, 1.0)
            strength = residual_strength + t * (anneal_target - residual_strength)
        else:
            strength = residual_strength

    loss_cfg = config.get("loss", {})
    use_mask = loss_cfg.get("mask_reg") is not None
    mc = ModelConfig(
        type=model_type,
        encoder_size=encoder_size,
        decoder_size=model_cfg.get("decoder_size", 400),
    )
    encoder = create_encoder(mc, num_bits=num_bits, strength=strength, use_mask=use_mask)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device)
    encoder.eval()
    print(f"Encoder loaded: type={model_type}, num_bits={num_bits}, size={encoder_size}")

    # Create datasets
    data_dir = f"{DATA_PATH}/train2017"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Creating datasets from {data_dir}...")

    augmentation = DetectionAugmentation(
        photometric_p=0.5,
        geometric_p=0.5,
    )

    full_dataset = DetectionDataset(
        image_dir=data_dir,
        encoder=encoder,
        num_bits=num_bits,
        positive_ratio=0.5,
        input_size=320,
        encoder_input_size=encoder_size,
        perspective_strength=(0.05, 0.20),
        transform=augmentation,
    )

    # Split into train/val (90/10)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    # Use num_workers=0 to keep encoder on GPU in main process (faster than CPU workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep in main process for GPU encoder
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Keep in main process for GPU encoder
    )

    # Create model
    print("Creating FastDetectorModel...")
    model = FastDetectorModel(input_size=320, pretrained=True)

    # Create loss function
    loss_fn = DetectionLoss(
        cls_weight=1.0,
        corner_weight=5.0,
        conf_weight=0.5,
    )

    # Create trainer
    output_dir = f"{CHECKPOINT_PATH}/detection"
    trainer = DetectionTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device=str(device),
    )

    # Train
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Checkpoints will be saved to {output_dir}")

    history = trainer.fit(
        num_epochs=epochs,
        save_dir=output_dir,
        save_every=5,
    )

    # Save final checkpoint
    from pathlib import Path

    final_path = Path(output_dir) / "final.pt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete. Final checkpoint saved to {final_path}")

    # Print final metrics
    if history:
        final_metrics = history[-1]
        print(f"Final train loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Final val loss: {final_metrics.get('val_loss', 'N/A'):.4f}")

    # Commit volume changes
    checkpoint_volume.commit()

    print(f"\nCheckpoints saved to Modal volume: picode-checkpoints/detection/")


@app.function(
    image=image,
    gpu="T4",
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
)
def evaluate_detector(
    detector_checkpoint: str = "detection/best.pt",
    encoder_checkpoint: str = "stegastamp_original/best.pt",
    num_samples: int = 500,
):
    """Evaluate trained FastDetector.

    Args:
        detector_checkpoint: Path to detector checkpoint (relative to /checkpoints)
        encoder_checkpoint: Path to encoder checkpoint (relative to /checkpoints)
        num_samples: Number of samples to evaluate
    """
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    import torch
    from torch.utils.data import DataLoader

    from picode.detection.fast_detector import FastDetector
    from picode.detection.training import DetectionDataset, DetectionEvaluator
    from picode.models.factory import create_encoder
    from picode.training.config import ModelConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load detector
    detector_path = f"{CHECKPOINT_PATH}/{detector_checkpoint}"
    print(f"Loading detector: {detector_path}")
    detector = FastDetector.from_checkpoint(detector_path, device=str(device))

    # Load encoder
    encoder_path = f"{CHECKPOINT_PATH}/{encoder_checkpoint}"
    print(f"Loading encoder: {encoder_path}")
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")
    encoder_size = model_cfg.get("encoder_size", 400)
    training_cfg = config.get("training", {})
    num_bits = training_cfg.get("num_bits", 100)

    # Compute strength for picotrust models
    residual_strength = training_cfg.get("residual_strength", 0)
    strength = None
    if residual_strength > 0:
        step = ckpt.get("step", 0)
        anneal_target = training_cfg.get("residual_strength_anneal_target", residual_strength)
        anneal_start = training_cfg.get("residual_strength_anneal_start", 0)
        anneal_steps = training_cfg.get("residual_strength_anneal_steps", 1)
        if step >= anneal_start and anneal_steps > 0:
            t = min((step - anneal_start) / anneal_steps, 1.0)
            strength = residual_strength + t * (anneal_target - residual_strength)
        else:
            strength = residual_strength

    loss_cfg = config.get("loss", {})
    use_mask = loss_cfg.get("mask_reg") is not None
    mc = ModelConfig(
        type=model_type,
        encoder_size=encoder_size,
        decoder_size=model_cfg.get("decoder_size", 400),
    )
    encoder = create_encoder(mc, num_bits=num_bits, strength=strength, use_mask=use_mask)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device).eval()
    print(f"Encoder loaded: type={model_type}, num_bits={num_bits}, size={encoder_size}")

    # Create evaluation dataset
    data_dir = f"{DATA_PATH}/train2017"
    print(f"Creating evaluation dataset from {data_dir}...")
    dataset = DetectionDataset(
        image_dir=data_dir,
        encoder=encoder,
        num_bits=num_bits,
        positive_ratio=0.5,
        encoder_input_size=encoder_size,
    )

    # Limit samples
    if num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
    )

    print(f"Evaluating on {len(dataset)} samples...")

    # Evaluate
    evaluator = DetectionEvaluator(detector.model, device=str(device))
    metrics = evaluator.evaluate_dataset(loader)

    print("\n=== Evaluation Results ===")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1 Score:  {metrics.f1:.4f}")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Mean IoU:  {metrics.mean_iou:.4f}")


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 6,  # 6 hour max
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
)
def train_detector_pregenerated(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.0001,
    dataset_path: str = "detection_dataset",
):
    """Train FastDetector using pre-generated dataset (much faster).

    The dataset should be created using scripts/prepare_detection_dataset.py
    and uploaded to the Modal volume.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size (can be larger since no encoder overhead)
        lr: Learning rate
        dataset_path: Path to pre-generated dataset in data volume
    """
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    import torch
    from torch.utils.data import DataLoader, random_split

    from picode.detection.fast_detector import FastDetectorModel
    from picode.detection.training import (
        DetectionAugmentation,
        DetectionLoss,
        DetectionTrainer,
        PregeneratedDetectionDataset,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load pre-generated dataset
    data_dir = f"{DATA_PATH}/{dataset_path}"
    print(f"Loading pre-generated dataset from {data_dir}...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Pre-generated dataset not found: {data_dir}\n"
            "Run scripts/prepare_detection_dataset.py locally, then:\n"
            "  modal volume put picode-data <output_dir> /detection_dataset"
        )

    augmentation = DetectionAugmentation(
        photometric_p=0.5,
        geometric_p=0.5,
    )

    full_dataset = PregeneratedDetectionDataset(
        data_dir=data_dir,
        transform=augmentation,
    )

    # Split into train/val (90/10)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders - can use multiple workers since no encoder!
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print("Creating FastDetectorModel...")
    model = FastDetectorModel(input_size=320, pretrained=True)

    # Create loss function
    loss_fn = DetectionLoss(
        cls_weight=1.0,
        corner_weight=5.0,
        conf_weight=0.5,
    )

    # Create trainer
    output_dir = f"{CHECKPOINT_PATH}/detection"
    trainer = DetectionTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device=str(device),
    )

    # Train
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Checkpoints will be saved to {output_dir}")

    history = trainer.fit(
        num_epochs=epochs,
        save_dir=output_dir,
        save_every=5,
    )

    # Save final checkpoint
    from pathlib import Path

    final_path = Path(output_dir) / "final.pt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete. Final checkpoint saved to {final_path}")

    # Print final metrics
    if history:
        final_metrics = history[-1]
        print(f"Final train loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Final val loss: {final_metrics.get('val_loss', 'N/A'):.4f}")

    # Commit volume changes
    checkpoint_volume.commit()

    print(f"\nCheckpoints saved to Modal volume: picode-checkpoints/detection/")


@app.local_entrypoint()
def main():
    """Default entrypoint - shows help."""
    print(__doc__)
