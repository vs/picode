# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them privately through
GitHub's [private vulnerability reporting](https://github.com/vs/picode/security/advisories/new)
(*Security → Report a vulnerability*). You should get a response within a week. Please give us
a reasonable window to ship a fix before you disclose publicly.

## Supported versions

Only the latest commit on `main` gets fixes.

## Scope and assumptions

Picode is research software. It is not a hardened service. Keep these assumptions in mind when
you assess a finding:

- **The payload is not a security feature.** Encoded IDs are neither encrypted nor
  authenticated. Anyone with a decoder can read them, and a determined adversary can remove or
  forge them. Reports showing that a mark can be stripped or spoofed are expected behavior,
  though research results are welcome as regular issues.
- **Checkpoints are pickles.** `torch.load(..., weights_only=False)` executes code from the
  file. Load only checkpoints you trust, such as the ones published at
  huggingface.co/vadishev/picotrust.
- **The scraper has no authentication layer.** It is a CLI plus workers that talk to
  PostgreSQL and optional S3. Run it on a private network. The bundled `docker-compose.yaml`
  uses development credentials (`picode`/`picode`) and publishes Postgres on `localhost:5432`.
  Change both before you deploy anywhere shared, and pass real credentials through environment
  variables (see [.env.example](.env.example)), never through committed config files.
- **The iOS app** processes camera frames on-device and has no network access.

In scope: code execution, credential leakage, injection (for example SQL in the scraper),
path traversal in the CLIs, and anything that makes the documented deployment unsafe.
