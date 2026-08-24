# Contributing to Picode

Thanks for your interest. This is a research codebase, so small, focused pull requests with a
clear motivation are the easiest to review.

## Setup

```bash
git clone https://github.com/vs/picode.git && cd picode
python -m venv venv && source venv/bin/activate
pip install -e "./picode-model[dev]"
pip install -e "./picode-scraper[dev]"
./scripts/setup-hooks.sh          # enables the leak guard git hooks
```

For the iOS app you also need Xcode 15+ and XcodeGen (see [picode-ios/README.md](picode-ios/README.md)).

## Before you open a pull request

| Project | Checks |
|---------|--------|
| picode-model | `pytest picode/tests/ -q`, `ruff check picode/` and `mypy picode/` |
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

## Leak guard

`scripts/guard/` keeps secrets and personal data out of the repository. `setup-hooks.sh` points
`core.hooksPath` at `.githooks/`, which runs the guard in three places:

| Hook | What it checks |
|------|----------------|
| `pre-commit` | Staged files: token formats (AWS, GitHub, Hugging Face, Modal, OpenAI, Google, Slack, private keys, JWTs), credential assignments, database URLs with real passwords, email addresses, personal home paths, and forbidden files (`.env*`, keys, tool/session directories, model weights, data exports outside `tests/fixtures/`, images outside `docs/`, files over 1 MB) |
| `commit-msg` | The same content rules, applied to the message |
| `pre-push` | Every outgoing commit, plus a refusal to force-push or delete `main` |

The hooks fail closed: if Python is missing, the commit or push is refused. CI
(`.github/workflows/leak-guard.yml`) runs the same generic rules on every push and pull
request, over the new commits and the whole tree.

- **False positive?** Add `guard:allow` to that line, for example in a trailing comment, and
  say why in the PR.
- **Private denylist (maintainer only).** `python3 scripts/guard/refresh_denylist.py` reads
  local credentials and account identifiers and writes them to
  `~/.config/picode-guard/denylist.txt` (mode 600, outside the repo). When that file exists,
  the hooks also block those exact values. Entries already present in tracked files are
  skipped and listed for manual review.
- **Override.** `PICODE_GUARD_OVERRIDE=1` bypasses the guard. Only the repository owner uses
  it, and the Claude Code hooks block it.
- **Tests:** `python3 -m unittest scripts/guard/test_rules.py`. Fake secrets in the tests are
  assembled at run time, so the test file passes its own guard.

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
