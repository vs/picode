# Contributing

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dev dependencies:

```bash
pip install -e ".[dev]"
```

## Making Changes

1. Create a feature branch from `main`
2. Write tests first (TDD)
3. Implement the minimal code to pass tests
4. Run the full test suite:

```bash
pytest picode/tests/ -v
```

5. Run linting and type checking:

```bash
ruff check picode/
mypy picode/
```

6. Commit with a descriptive message

## Project Structure

When contributing, note that this is a single unified package with swappable implementations:

```
picode/
├── distortions/           # Image distortions with multiple backends
│   ├── base.py            # Distortion ABC
│   ├── native/            # Pure PyTorch implementations
│   └── kornia/            # Kornia-based implementations
├── ecc/                   # Error correction codes
│   ├── base.py            # ECC ABC
│   ├── bch/               # BCH implementation
│   └── ldpc/              # LDPC implementation
├── models/                # Encoder/decoder models
│   ├── base.py            # Encoder/Decoder ABC
│   └── stegastamp/        # StegaStamp implementation
└── tests/                 # Test suite
    ├── distortions/
    ├── ecc/
    └── models/
```

## Adding a New Implementation

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
fix: clamp output values to [0, 1] range

Some distortions could produce values outside valid range,
causing issues in downstream processing.
```

```
test: add gradient flow tests for encoder
```

Keep messages concise. First line under 72 characters.

## Testing Guidelines

- Test files mirror source files (e.g., `blur.py` → `test_blur.py`)
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
