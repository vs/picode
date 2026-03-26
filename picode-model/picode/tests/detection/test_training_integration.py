# picode/tests/detection/test_training_integration.py
"""Integration tests for detection training pipeline."""

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset

from picode.detection import FastDetectorModel
from picode.detection.training import (
    DetectionAugmentation,
    DetectionDataset,
    DetectionEvaluator,
    DetectionLoss,
    DetectionTrainer,
    HardNegativeDataset,
)


class MockEncoder:
    """Mock encoder for testing."""

    def __call__(self, image: Tensor, message: Tensor) -> Tensor:
        # Return image with small perturbation
        return image + torch.randn_like(image) * 0.01


class TestTrainingPipelineIntegration:
    @pytest.fixture
    def positive_dataset(self, tmp_path) -> DetectionDataset:
        """Create positive dataset with mock encoder."""
        from PIL import Image
        import numpy as np

        img_dir = tmp_path / "positives"
        img_dir.mkdir()
        for i in range(6):
            img = Image.fromarray(
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            )
            img.save(img_dir / f"pos_{i}.jpg")

        return DetectionDataset(
            image_dir=str(img_dir),
            encoder=MockEncoder(),
            num_bits=100,
            positive_ratio=1.0,
            input_size=320,
        )

    @pytest.fixture
    def negative_dataset(self, tmp_path) -> HardNegativeDataset:
        """Create negative dataset."""
        from PIL import Image
        import numpy as np

        img_dir = tmp_path / "negatives"
        img_dir.mkdir()
        for i in range(6):
            img = Image.fromarray(
                np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            )
            img.save(img_dir / f"neg_{i}.jpg")

        return HardNegativeDataset(
            image_dir=str(img_dir),
            input_size=320,
            transform_p=0.8,
        )

    def test_combined_dataset(
        self,
        positive_dataset: DetectionDataset,
        negative_dataset: HardNegativeDataset,
    ) -> None:
        """Test combining positive and negative datasets."""
        combined = ConcatDataset([positive_dataset, negative_dataset])

        assert len(combined) == len(positive_dataset) + len(negative_dataset)

        # Check both types of samples are accessible
        pos_sample = combined[0]
        neg_sample = combined[len(positive_dataset)]

        assert pos_sample["is_watermark"] == 1.0
        assert neg_sample["is_watermark"] == 0.0

    def test_augmentation_in_pipeline(
        self, positive_dataset: DetectionDataset
    ) -> None:
        """Test augmentation applied during training."""
        augment = DetectionAugmentation(photometric_p=1.0, geometric_p=1.0)

        sample = positive_dataset[0]
        augmented = augment(sample)

        assert augmented["image"].shape == sample["image"].shape
        assert augmented["corners"].shape == sample["corners"].shape

    def test_dataloader_batching(
        self,
        positive_dataset: DetectionDataset,
        negative_dataset: HardNegativeDataset,
    ) -> None:
        """Test DataLoader batching works correctly."""
        combined = ConcatDataset([positive_dataset, negative_dataset])
        loader = DataLoader(combined, batch_size=4, shuffle=True)

        batch = next(iter(loader))

        assert batch["image"].shape == (4, 3, 320, 320)
        assert batch["is_watermark"].shape == (4,)
        assert batch["corners"].shape == (4, 8)

    def test_full_training_loop(
        self,
        positive_dataset: DetectionDataset,
        negative_dataset: HardNegativeDataset,
    ) -> None:
        """Test full training loop runs without errors."""
        # Create model
        model = FastDetectorModel(input_size=320, pretrained=False)

        # Create dataloaders
        combined = ConcatDataset([positive_dataset, negative_dataset])
        train_loader = DataLoader(combined, batch_size=2, shuffle=True)
        val_loader = DataLoader(combined, batch_size=2, shuffle=False)

        # Create trainer
        trainer = DetectionTrainer(
            model=model,
            loss_fn=DetectionLoss(),
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-4,
            device="cpu",
        )

        # Train for 1 epoch
        history = trainer.fit(num_epochs=1)

        assert len(history) == 1
        assert history[0]["train_loss"] >= 0
        assert history[0]["val_loss"] >= 0

    def test_evaluation_after_training(
        self,
        positive_dataset: DetectionDataset,
        negative_dataset: HardNegativeDataset,
    ) -> None:
        """Test evaluation works after training."""
        model = FastDetectorModel(input_size=320, pretrained=False)

        combined = ConcatDataset([positive_dataset, negative_dataset])
        loader = DataLoader(combined, batch_size=2, shuffle=False)

        evaluator = DetectionEvaluator(model=model, threshold=0.5, device="cpu")
        metrics = evaluator.evaluate_dataset(loader)

        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_checkpoint_save_load_resume(
        self,
        positive_dataset: DetectionDataset,
        negative_dataset: HardNegativeDataset,
        tmp_path,
    ) -> None:
        """Test checkpoint save/load/resume training."""
        model = FastDetectorModel(input_size=320, pretrained=False)
        combined = ConcatDataset([positive_dataset, negative_dataset])
        train_loader = DataLoader(combined, batch_size=2, shuffle=True)

        trainer = DetectionTrainer(
            model=model,
            loss_fn=DetectionLoss(),
            train_loader=train_loader,
            lr=1e-4,
            device="cpu",
        )

        # Train 1 epoch and save
        trainer.fit(num_epochs=1)
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)

        # Create new trainer and load
        model2 = FastDetectorModel(input_size=320, pretrained=False)
        trainer2 = DetectionTrainer(
            model=model2,
            loss_fn=DetectionLoss(),
            train_loader=train_loader,
            lr=1e-4,
            device="cpu",
        )
        trainer2.load_checkpoint(checkpoint_path)

        # Verify epoch was restored
        assert trainer2.epoch == 1

        # Continue training
        history = trainer2.fit(num_epochs=1)
        assert len(history) == 1
