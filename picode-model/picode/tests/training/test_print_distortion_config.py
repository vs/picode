"""Print-to-photo distortions must parse into DistortionRamp, not raw dicts."""

import torch

from picode.training.config import DistortionRamp, load_config
from picode.training.distortion_strategy import create_distortion_strategy

PRINT_KEYS = [
    "resolution_loss",
    "shot_noise",
    "barrel_distortion",
    "vignetting",
    "chromatic_aberration",
]

YAML = """
experiment_name: t
data:
  source: folder
  path: ./data
distortion:
  strategy: curriculum
  resolution_loss:
    strength: 0.35
    ramp_steps: 100
  shot_noise:
    strength: 0.04
    ramp_steps: 100
  barrel_distortion:
    strength: 0.08
    ramp_steps: 100
  vignetting:
    strength: 0.25
    ramp_steps: 100
  chromatic_aberration:
    strength: 0.002
    ramp_steps: 100
"""


def _cfg(tmp_path):
    p = tmp_path / "print.yaml"
    p.write_text(YAML)
    return load_config(p)


def test_print_distortions_parse_as_dataclasses(tmp_path):
    """Each print-to-photo key becomes a DistortionRamp."""
    dist = _cfg(tmp_path).distortion
    for key in PRINT_KEYS:
        value = getattr(dist, key)
        assert isinstance(value, DistortionRamp), f"{key} parsed as {type(value).__name__}"


def test_print_distortions_keep_their_values(tmp_path):
    dist = _cfg(tmp_path).distortion
    assert dist.resolution_loss.strength == 0.35
    assert dist.resolution_loss.ramp_steps == 100
    assert dist.chromatic_aberration.strength == 0.002


def test_strategy_runs_with_print_distortions(tmp_path):
    """The curriculum must apply them without an AttributeError."""
    cfg = _cfg(tmp_path)
    strategy = create_distortion_strategy(cfg.distortion)
    out = strategy(torch.rand(1, 3, 64, 64), 100)  # past the ramp
    assert out.shape == (1, 3, 64, 64)
    assert torch.isfinite(out).all()


def test_print_distortions_default_to_none():
    """Configs that do not mention them are unaffected."""
    from picode.training.config import DistortionConfig

    dist = DistortionConfig()
    for key in PRINT_KEYS:
        assert getattr(dist, key) is None
