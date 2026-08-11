#!/usr/bin/env python3
"""Build the private leak-guard denylist from local data sources.

Reads credentials and account identifiers from the owner's machine (Modal, Kaggle,
Hugging Face, gcloud, git config, local .env files) and writes them to
~/.config/picode-guard/denylist.txt with mode 0600. The file never enters the repo.

Entries already present in the tracked working tree are skipped (they are public), but
every skipped entry is listed so a human can confirm it is not an existing leak.

Usage: python3 scripts/guard/refresh_denylist.py [--extra FILE] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules  # noqa: E402

HOME = Path.home()
OUT = Path(os.environ.get("PICODE_GUARD_DENYLIST", "~/.config/picode-guard/denylist.txt")).expanduser()
PUBLIC_EMAIL = "semyon.vadishev@gmail.com"
MIN_LEN = 4
IGNORE_VALUES = {"true", "false", "none", "null", "default", "us-west1", "us-central1", "us-east1",
                 "localhost", "picode"}


def collect() -> list[rules.DenyEntry]:
    out: list[rules.DenyEntry] = []

    def add(kind: str, value: object, label: str) -> None:
        v = str(value).strip()
        if v.lower() == PUBLIC_EMAIL:
            return
        if len(v) >= MIN_LEN and v.lower() not in IGNORE_VALUES and not re.fullmatch(r"[a-z]+-[a-z]+\d-[a-z]", v):
            out.append(rules.DenyEntry(kind, v, label))

    modal = HOME / ".modal.toml"
    if modal.is_file():
        for profile, cfg in tomllib.loads(modal.read_text()).items():
            add("word-ci", profile, "modal-workspace")
            for k, v in cfg.items():
                if "token" in k or "secret" in k:
                    add("literal", v, f"modal-{k}")
    kaggle = HOME / ".kaggle" / "kaggle.json"
    if kaggle.is_file():
        data = json.loads(kaggle.read_text())
        add("word-ci", data.get("username", ""), "kaggle-username")
        add("literal", data.get("key", ""), "kaggle-key")
    for f, label in [(HOME / ".kaggle" / "access_token", "kaggle-access-token"),
                     (HOME / ".cache" / "huggingface" / "token", "hf-token")]:
        if f.is_file():
            add("literal", f.read_text().strip(), label)
    gcloud = HOME / ".config" / "gcloud" / "configurations"
    for cfg in sorted(gcloud.glob("config_*")) if gcloud.is_dir() else []:
        for line in cfg.read_text().splitlines():
            m = re.match(r"\s*(account|project)\s*=\s*(\S+)", line)
            if m:
                add("word-ci", m[2], f"gcloud-{m[1]}")
    for key in ("user.email", "github.user", "sendemail.smtpuser"):
        r = subprocess.run(["git", "config", "--global", key], capture_output=True, text=True)
        if r.stdout.strip() and r.stdout.strip() != PUBLIC_EMAIL:
            add("word-ci", r.stdout.strip(), f"git-{key}")
    # every value from local .env files next to (or inside) the repository
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                               text=True).stdout.strip() or ".")
    for env in list(root.glob("**/.env")) + list(root.glob("**/.env.*")):
        if env.name == ".env.example" or "venv" in env.parts or "node_modules" in env.parts:
            continue
        for line in env.read_text(errors="replace").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]?([^'\"#\s]+)", line)
            if m and len(m[2]) >= 8:
                add("literal", m[2], f"env:{m[1]}")
    # the home directory itself
    add("literal", str(HOME) + "/", "home-path")
    return out


def tracked_text() -> str:
    files = subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.split(b"\0")
    chunks = []
    for f in files:
        p = Path(f.decode()) if f else None
        if p and p.is_file() and p.stat().st_size < 2_000_000:
            data = p.read_bytes()
            if not rules.is_binary(data):
                chunks.append(data.decode("utf-8", "replace"))
    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=Path, help="extra kind<TAB>value<TAB>label lines to include")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = collect()
    if args.extra:
        entries += rules.parse_denylist(args.extra.read_text())
    seen: set[tuple[str, str]] = set()
    unique = [e for e in entries if (e.kind, e.value) not in seen and not seen.add((e.kind, e.value))]

    corpus = tracked_text()
    kept, skipped = [], []
    for e in unique:
        (skipped if e.compile().search(corpus) else kept).append(e)

    print(f"denylist: {len(kept)} entries kept, {len(skipped)} skipped as already public")
    for e in skipped:
        shown = e.value if e.kind in ("word-ci", "ticker", "ticker-ci") and "token" not in e.label else rules.redact(e.value)
        print(f"  REVIEW skipped [{e.label}] {shown} — present in tracked files; confirm this is not a leak")
    if args.dry_run:
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT.parent, 0o700)
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("# Picode leak-guard denylist — private, generated by refresh_denylist.py\n")
        for e in kept:
            fh.write(f"{e.kind}\t{e.value}\t{e.label}\n")
    os.chmod(OUT, 0o600)
    print(f"wrote {OUT} (mode 600)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
