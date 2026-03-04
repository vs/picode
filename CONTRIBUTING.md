# Contributing

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dev dependencies:

```bash
cd picode-model
pip install -e ".[dev]"
```

## Making Changes

1. Create a feature branch from `main`
2. Write tests first (TDD)
3. Implement the minimal code to pass tests
4. Run the full test suite:

```bash
cd picode-model
pytest picode/tests/ -v
```

5. Run linting and type checking:

```bash
cd picode-model
ruff check picode/
mypy picode/
```

6. Commit with a descriptive message

## Project Structure

This is a monorepo with multiple sub-projects. When contributing to the PyTorch model code, work within `picode-model/`:

```
/
├── docs/                    # Shared documentation
├── picode-ios/              # iOS application
├── picode-model/            # PyTorch training framework (main dev work)
│   ├── picode/              # Python package
│   │   ├── distortions/     # Image distortions with multiple backends
│   │   │   ├── base.py      # Distortion ABC
│   │   │   ├── native/      # Pure PyTorch implementations
│   │   │   └── kornia/      # Kornia-based implementations
│   │   ├── ecc/             # Error correction codes
│   │   │   ├── base.py      # ECC ABC
│   │   │   ├── bch/         # BCH implementation (galois library)
│   │   │   └── ldpc/        # LDPC implementation (pyldpc library)
│   │   ├── models/          # Encoder/decoder models
│   │   │   ├── base.py      # Encoder/Decoder ABC
│   │   │   └── stegastamp/  # StegaStamp implementation
│   │   ├── training/        # Training infrastructure
│   │   │   ├── config.py    # YAML config loading
│   │   │   ├── trainer.py   # Trainer with loss ramping
│   │   │   └── ...
│   │   └── tests/           # Test suite
│   ├── configs/             # Training configuration files
│   ├── scripts/             # Utility scripts
│   └── pyproject.toml       # Package configuration
└── venv/                    # Shared Python virtual environment
```

## Adding a New Implementation

All paths below are relative to `picode-model/`.

### New Distortion Backend

1. Create a new directory: `picode/distortions/mybackend/`
2. Implement distortions inheriting from `picode.distortions.base.Distortion`
3. Export in `__init__.py`
4. Add tests in `picode/tests/distortions/`

### New Model Architecture

1. Create a new directory: `picode/models/mymodel/`
2. Implement encoder/decoder inheriting from `picode.models.base.Encoder/Decoder`
3. Export in `__init__.py`
4. Add tests in `picode/tests/models/`

### New ECC Implementation

1. Create a new directory: `picode/ecc/myecc/`
2. Implement ECC inheriting from `picode.ecc.base.ECC`
3. Export in `__init__.py`
4. Add tests in `picode/tests/ecc/`

### New Training Component

1. Add module in `picode/training/`
2. Follow existing patterns (dataclasses for config, protocols for interfaces)
3. Export in `picode/training/__init__.py`
4. Add tests in `picode/tests/training/`

Key training components:
- **Distortion strategies**: Inherit from base strategy, implement `get_distortion(step)` method
- **Loggers**: Implement the `Logger` protocol from `picode.training.logging.base`
- **Config sections**: Use dataclasses with `from_dict` class methods

## Code Review

- All changes require review before merging
- Address all review comments
- Keep PRs focused and small

## Commit Message Style

Format:
```
<type>: <short description>

<optional body explaining why, not what>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `refactor`: Code changes that don't add features or fix bugs
- `chore`: Build, config, or tooling changes

Examples:
```
feat: add perspective warp distortion

Implements 4-point perspective transformation.
Supports configurable intensity and corner displacement.
```

```
feat(models): add LPIPS perceptual loss option

Adds optional perceptual loss using LPIPS for better
image quality during training.
```

```
feat(training): add curriculum distortion strategy

Implements gradual distortion strength ramping during training,
matching StegaStamp's curriculum learning approach.
```

```
fix: clamp output values to [0, 1] range

Some distortions could produce values outside valid range,
causing issues in downstream processing.
```

```
test: add gradient flow tests for encoder
```

Keep messages concise. First line under 72 characters.

## Testing Guidelines

- Test files mirror source files (e.g., `blur.py` -> `test_blur.py`)
- Use fixtures from `conftest.py` for common test data
- Always test:
  - Output shape preservation
  - Output range validity [0, 1]
  - Gradient flow (for differentiability)

Example test:

```python
def test_encoder_gradient_flow(sample_image, sample_message):
    from picode.models.stegastamp import Encoder

    encoder = Encoder(num_bits=100)
    sample_image.requires_grad = True

    output = encoder(sample_image, sample_message)
    output.sum().backward()

    assert sample_image.grad is not None
    assert not torch.isnan(sample_image.grad).any()
```

### Dual-Backend Distortion Tests

Distortion tests should be parametrized to test both backends:

```python
import pytest

@pytest.mark.parametrize("backend", ["native", "kornia"])
def test_gaussian_blur_shape(sample_image, backend):
    if backend == "native":
        from picode.distortions.native import GaussianBlur
    else:
        from picode.distortions.kornia import GaussianBlur

    blur = GaussianBlur(intensity=0.5)
    output = blur(sample_image)
    assert output.shape == sample_image.shape
```

### Training Tests

Training tests should use small models and few steps:

```python
def test_trainer_step(tmp_path):
    from picode.training import Trainer, Config

    config = Config(
        training=TrainingConfig(num_steps=10, num_bits=10),
        # ... minimal config
    )
    trainer = Trainer(config)

    # Run a few steps
    for _ in range(3):
        metrics = trainer.step()
        assert "loss" in metrics
```
