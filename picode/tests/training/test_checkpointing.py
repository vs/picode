"""Tests for checkpointing."""

from pathlib import Path

import torch
import torch.nn as nn

from picode.training.checkpointing import Checkpointer
from picode.training.config import CheckpointConfig, Config, DataConfig


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.linear(x)
        return result


class TestCheckpointer:
    def test_should_save(self) -> None:
        config = CheckpointConfig(save_every_steps=100)
        ckpt = Checkpointer(config, "test_exp")
        assert not ckpt.should_save(0)
        assert not ckpt.should_save(50)
        assert ckpt.should_save(100)
        assert ckpt.should_save(200)

    def test_is_best_lower_is_better(self) -> None:
        config = CheckpointConfig()
        ckpt = Checkpointer(config, "test", metric_name="loss", lower_is_better=True)
        assert ckpt.is_best({"loss": 0.5})  # First is always best
        ckpt.best_metric = 0.5
        assert ckpt.is_best({"loss": 0.4})  # Lower is better
        assert not ckpt.is_best({"loss": 0.6})

    def test_save_and_load(self, tmp_path: Path) -> None:
        config = CheckpointConfig(dir=str(tmp_path), save_every_steps=10)
        ckpt = Checkpointer(config, "test_exp")

        encoder = SimpleModel()
        decoder = SimpleModel()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
        cfg = Config(experiment_name="test", data=DataConfig(source="folder", path="/data"))

        ckpt.save(
            step=10,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=None,
            config=cfg,
            metrics={"loss": 0.5},
        )

        # Verify checkpoint file exists
        assert (tmp_path / "test_exp" / "checkpoint_00000010.pt").exists()

        # Load it back
        state = ckpt.load()
        assert state is not None
        assert state.step == 10

    def test_keeps_last_n(self, tmp_path: Path) -> None:
        config = CheckpointConfig(dir=str(tmp_path), save_every_steps=10, keep_last=2)
        ckpt = Checkpointer(config, "test_exp")

        encoder = SimpleModel()
        decoder = SimpleModel()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
        cfg = Config(experiment_name="test", data=DataConfig(source="folder", path="/data"))

        for step in [10, 20, 30, 40]:
            ckpt.save(step, encoder, decoder, optimizer, None, cfg, {"loss": 0.5})

        # Should only have last 2
        checkpoints = list((tmp_path / "test_exp").glob("checkpoint_*.pt"))
        assert len(checkpoints) == 2

    def test_saves_best(self, tmp_path: Path) -> None:
        config = CheckpointConfig(dir=str(tmp_path), save_every_steps=10)
        ckpt = Checkpointer(config, "test_exp", metric_name="loss")

        encoder = SimpleModel()
        decoder = SimpleModel()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
        cfg = Config(experiment_name="test", data=DataConfig(source="folder", path="/data"))

        ckpt.save(10, encoder, decoder, optimizer, None, cfg, {"loss": 0.5})
        ckpt.save(20, encoder, decoder, optimizer, None, cfg, {"loss": 0.3})  # New best
        ckpt.save(30, encoder, decoder, optimizer, None, cfg, {"loss": 0.4})  # Not best

        best_state = ckpt.load_best()
        assert best_state is not None
        assert best_state.step == 20
