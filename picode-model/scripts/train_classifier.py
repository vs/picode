#!/usr/bin/env python
"""Train ModelClassifier for Strategy B routing.

Trains a MobileNetV3-Small classifier to predict which encoder model
produced a watermarked image (e.g. b30, b48, b72, b96).

Usage:
    python scripts/train_classifier.py \\
        --shard-dirs data/b30_shards data/b48_shards data/b72_shards data/b96_shards \\
        --model-labels b30 b48 b72 b96 \\
        --epochs 50 \\
        --output-dir checkpoints/model_classifier
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from picode.detection.model_classifier import ModelClassifierModel
from picode.detection.training.pregenerated_dataset import PregeneratedDetectionDataset

# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class LabeledDatasetWrapper(Dataset):
    """Wraps a PregeneratedDetectionDataset with a fixed integer class label.

    Each sample dict from the underlying dataset is augmented with a
    ``"class_label"`` key containing ``torch.tensor(label, dtype=torch.long)``.

    Args:
        dataset: Underlying PregeneratedDetectionDataset.
        label: Integer class index for all samples in this dataset.
    """

    def __init__(self, dataset: Dataset, label: int) -> None:
        self.dataset = dataset
        self.label = label

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.dataset[idx]
        sample["class_label"] = torch.tensor(self.label, dtype=torch.long)
        return sample


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ModelClassifier for Strategy B routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shard-dirs",
        nargs="+",
        required=True,
        help="One shard directory per model (positives only, one per model label).",
    )
    parser.add_argument(
        "--model-labels",
        nargs="+",
        required=True,
        help="Model label for each shard directory (e.g. b30 b48 b72 b96).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/model_classifier",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 = main process).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save a periodic checkpoint every N epochs.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping max norm (None = disabled).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------


def run_epoch(
    model: ModelClassifierModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float | None = None,
    train: bool = True,
) -> tuple[float, float]:
    """Run one full epoch (train or eval).

    Args:
        model: The classifier model.
        loader: DataLoader providing batches.
        criterion: CrossEntropyLoss.
        optimizer: Adam optimizer (None for eval).
        device: Torch device.
        grad_clip: Optional gradient clipping norm.
        train: If True, update weights; otherwise run in no-grad eval mode.

    Returns:
        Tuple of (mean_loss, accuracy) for the epoch.
    """
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            images: Tensor = batch["image"].to(device)
            labels: Tensor = batch["class_label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += bs

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if len(args.shard_dirs) != len(args.model_labels):
        raise ValueError(
            f"--shard-dirs ({len(args.shard_dirs)}) and --model-labels "
            f"({len(args.model_labels)}) must have the same length."
        )

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load datasets
    sub_datasets: list[Dataset] = []
    for shard_dir, label_str in zip(args.shard_dirs, args.model_labels):
        label_idx = args.model_labels.index(label_str)
        print(f"Loading shard dir '{shard_dir}' as label '{label_str}' (idx={label_idx})...")
        base_ds = PregeneratedDetectionDataset(data_dir=shard_dir)
        wrapped_ds = LabeledDatasetWrapper(base_ds, label=label_idx)
        print(f"  {len(wrapped_ds)} samples")
        sub_datasets.append(wrapped_ds)

    full_dataset: Dataset
    if len(sub_datasets) == 1:
        full_dataset = sub_datasets[0]
    else:
        full_dataset = ConcatDataset(sub_datasets)
    print(f"Total samples: {len(full_dataset)}")  # type: ignore[arg-type]

    # Train / val split
    n_total = len(full_dataset)  # type: ignore[arg-type]
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Model, loss, optimizer
    num_classes = len(args.model_labels)
    print(f"Creating ModelClassifierModel: {num_classes} classes {args.model_labels}")
    model = ModelClassifierModel(
        num_classes=num_classes,
        class_names=args.model_labels,
        input_size=320,
        pretrained=True,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = -1

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=args.grad_clip, train=True,
        )

        # Validate
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device,
            train=False,
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_path = output_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": args.model_labels,
                    "input_size": 320,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  -> New best val_acc={val_acc:.4f}, saved to {best_path}")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": args.model_labels,
                    "input_size": 320,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  -> Periodic checkpoint saved to {ckpt_path}")

    # Final checkpoint
    final_path = output_dir / "final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "class_names": args.model_labels,
            "input_size": 320,
            "val_acc": best_val_acc,
            "epoch": args.epochs,
        },
        final_path,
    )

    print("\nTraining complete.")
    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
