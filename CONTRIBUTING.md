# Contributing to Picode

Thanks for your interest. This is a research codebase, so small, focused pull requests with a
clear motivation are the easiest to review.

## Setup

```bash
git clone https://github.com/vs/picode.git && cd picode
python -m venv venv && source venv/bin/activate
pip install -e "./picode-model[dev]"
pip install -e "./picode-scraper[dev]"
```

For the iOS app you also need Xcode 15+ and XcodeGen (see [picode-ios/README.md](picode-ios/README.md)).

## Before you open a pull request

| Project | Checks |
|---------|--------|
| picode-model | `pytest picode/tests/ -q` and `ruff check picode/` |
| picode-scraper | `pytest tests/ -q`, `ruff check picode_scraper/` and `mypy picode_scraper/` |
| picode-ios | `xcodebuild test -scheme picode-ios -destination 'platform=iOS Simulator,name=iPhone 17 Pro'` |

- Add or update tests with every behavior change. Distortion tests run against both backends
  (`@pytest.mark.parametrize("backend", ["native", "kornia"])`).
- Tensors are NCHW, float, in `[0, 1]`. Unit tests use 64×64 images, and model tests use real
  model sizes.
- Python code uses type hints (`X | None`), Google-style docstrings and 100-character lines
  (enforced by ruff).
- Don't commit datasets, checkpoints (`*.pt`), exported models or generated images. They are
  git-ignored, so keep them in `data/`, `checkpoints/` or `exports/`. Figures for the docs go in
  `docs/assets/` and must be generated from freely licensed sources.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `build:`, `chore:`, optionally with a scope,
e.g. `fix(scraper): honour PICODE_CONFIG in the CLI`.

## Training runs

Training configs live in `picode-model/configs/`. When you propose a new model, include the
config, the evaluation command you ran and its results (PSNR, bit accuracy, robustness sweep),
and follow the naming scheme `b{bits}s{strength}m{mask_floor}` described in
[docs/picotrust.md](docs/picotrust.md).

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
