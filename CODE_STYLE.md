# Code Style Guidelines

## Formatting

- Use `ruff` for linting and formatting
- Line length: 100 characters
- Use double quotes for strings

```bash
ruff check picode/
ruff format picode/
```

## Type Hints

- All public functions must have type hints
- Use `Tensor` from `torch` for tensor types
- Use `| None` syntax (Python 3.10+) instead of `Optional`
- Run mypy for type checking:

```bash
mypy picode/
```

## Docstrings

- Use Google-style docstrings
- Document all public classes and functions
- Include Args, Returns, and Raises sections

```python
def encode(self, image: Tensor, message: Tensor) -> Tensor:
    """Encode a message into an image.

    Args:
        image: Input image tensor of shape (B, 3, H, W) in [0, 1] range.
        message: Binary message tensor of shape (B, num_bits).

    Returns:
        Encoded image tensor of shape (B, 3, H, W) in [0, 1] range.

    Raises:
        ValueError: If image is not in valid range.
    """
```

## Module Organization

### Distortions (`picode/distortions/`)

- Base class in `base.py`
- One distortion type per file in `native/` (blur.py, noise.py, etc.)
- All distortions inherit from `Distortion` base class
- Export public API in `__init__.py`

### Models (`picode/models/`)

- Base classes in `base.py`
- Each architecture in its own subdirectory (e.g., `stegastamp/`)
- Encoder and decoder in separate files
- Loss functions in `loss.py`
- Training utilities in `train.py`

### ECC (`picode/ecc/`)

- Base class in `base.py`
- Each implementation in its own subdirectory (e.g., `bch/`, `ldpc/`)

## Testing

- One test file per module (test_blur.py, test_encoder.py, etc.)
- Test function naming: `test_<function>_<scenario>`
- Use pytest fixtures for shared setup (see `conftest.py`)
- Standard assertions: shape preservation, range validity [0,1], gradient flow

```bash
pytest picode/tests/ -v                           # All tests
pytest picode/tests/distortions/ -v               # Distortion tests
pytest picode/tests/models/ -v                    # Model tests
pytest -k "test_forward" -v                       # Pattern matching
```

## Naming Conventions

- Classes: PascalCase (e.g., `PerspectiveWarp`, `Encoder`, `StegaStampTrainer`)
- Functions/methods: snake_case (e.g., `sample_parameters`, `train_step`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_INTENSITY`)
- Private methods: Leading underscore (e.g., `_compute_homography`)

## Tensor Conventions

- Format: NCHW (batch, channels, height, width)
- Range: [0, 1] for images
- Common test sizes: 400×400 for models, 64×64 for unit tests

## Import Patterns

```python
# Distortions - import from specific backend
from picode.distortions.native import GaussianBlur, Compose
from picode.distortions.base import Distortion

# Models - import from specific architecture
from picode.models.stegastamp import Encoder, Decoder, train_step
from picode.models.base import Encoder as BaseEncoder

# ECC - import from specific implementation
from picode.ecc.base import ECC
```
