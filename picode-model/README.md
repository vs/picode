# picode-model

> Part of the [Picode monorepo](../README.md).

The PyTorch package behind Picode. It contains the models, the differentiable distortions, error
correction, detection, the training loop and the command-line tools.

## Install

```bash
python -m venv ../venv && source ../venv/bin/activate
pip install -e ".[dev]"            # add ,lpips for perceptual loss; ,export-ios for Core ML export
```

## Command-line tools

| Command | Purpose |
|---------|---------|
| `picode` | Encode a text message into an image / decode it (`picode -c CKPT encode …` / `decode …`) |
| `picode-train` | Train from a YAML config, with `section.key=value` overrides |
| `distort` | Apply any distortion (or all of them) to an image |
| `detect` | Find encoded regions in images or video |
| `detect-export` | Export the fast detector to Core ML / TFLite |

## Package layout

```
picode/
├── models/        # picotrust (production), stegastamp (baseline), picotier and experimental variants
├── distortions/   # native/ (pure PyTorch) and kornia/ backends
├── ecc/           # ldpc/ (soft-decision BP) and bch/
├── detection/     # detectors, rectifier, training, export
├── training/      # config, trainer, data, distortion strategies, evaluation, checkpointing
└── tests/
configs/           # training configs (see ../docs/picotrust.md for what each produced)
scripts/           # evaluation, sample/figure generation, Modal / GCE / Kaggle / Lightning runners
```

## Tests and linting

```bash
pytest picode/tests/ -q
ruff check picode/
mypy picode/                       # strict; the whole package type-checks
```

See the [root README](../README.md) for the quick start and the [docs](../docs/) for results.
