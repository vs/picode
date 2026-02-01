# Contributing

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dev dependencies for both packages:

```bash
cd util && pip install -e ".[dev]"
cd ../model && pip install -e ".[dev]"
```

## Making Changes

1. Create a feature branch from `main`
2. Write tests first (TDD)
3. Implement the minimal code to pass tests
4. Run the full test suite:

```bash
pytest util/tests/ -v          # Distortions tests
pytest model/tests/ -v         # Model tests
```

5. Run linting and type checking:

```bash
ruff check util/distortions/ model/stegastamp/
mypy util/distortions/ model/stegastamp/
```

6. Commit with a descriptive message

## Project Structure

When contributing, note that this is a monorepo with two independent packages:

- **util/distortions/**: Image distortion library
- **model/stegastamp/**: Encoder, decoder, and training utilities

Each package has its own `pyproject.toml` and test suite.

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
feat(model): add LPIPS perceptual loss option

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
    encoder = Encoder(num_bits=100)
    sample_image.requires_grad = True

    output = encoder(sample_image, sample_message)
    output.sum().backward()

    assert sample_image.grad is not None
    assert not torch.isnan(sample_image.grad).any()
```
