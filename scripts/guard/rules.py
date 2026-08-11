"""Pure leak-detection rules for the Picode leak guard.

No I/O here: callers pass paths, text and an optional private denylist, and get
findings back. See scan.py for the git / Claude Code entry points.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

ALLOW_MARKER = "guard:allow"
MAX_FILE_BYTES = 1_000_000
MAX_ASSET_BYTES = 5_000_000


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int
    excerpt: str  # already redacted

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.rule}] {loc}: {self.excerpt}"


@dataclass(frozen=True)
class DenyEntry:
    kind: str  # literal | word-ci | ticker | ticker-ci | regex
    value: str
    label: str = ""

    def compile(self) -> re.Pattern[str]:
        if self.kind == "regex":
            return re.compile(self.value)
        if self.kind == "literal":
            return re.compile(re.escape(self.value))
        if self.kind == "word-ci":
            return re.compile(rf"(?<![\w.-]){re.escape(self.value)}(?![\w-])", re.I)
        if self.kind == "ticker":
            return re.compile(rf"(?<![A-Za-z0-9]){re.escape(self.value)}(?![A-Za-z0-9])")
        if self.kind == "ticker-ci":
            return re.compile(rf"(?<![A-Za-z0-9]){re.escape(self.value)}(?![A-Za-z0-9])", re.I)
        raise ValueError(f"unknown denylist kind: {self.kind}")


def parse_denylist(text: str) -> list[DenyEntry]:
    """Parse ``kind<TAB>value[<TAB>label]`` lines; ``#`` starts a comment."""
    entries = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            raise ValueError(f"bad denylist line (need kind<TAB>value): {raw[:20]}...")
        entries.append(DenyEntry(parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return entries


def redact(value: str) -> str:
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:2]}…{value[-2:]} ({len(value)} chars)"


# --- content rules ------------------------------------------------------------------------

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("modal-token", re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe-key", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{16,}\b")),
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "credential-assignment",
        re.compile(
            r"""(?ix)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|passw(?:or)?d)
            \s*[:=]\s*["']?(?P<val>[A-Za-z0-9_\-/+=]{16,})"""
        ),
    ),
]

# Placeholder values that are fine in credential assignments / DB URLs.
PLACEHOLDER_VALUES = re.compile(
    r"(?i)^(?:x+|\*+|change-?me|changeme|placeholder|example|dummy|fake|test|pass(?:word)?|"
    r"secret|picode|user|your[-_a-z]*|<[^>]+>|\$\{?[A-Z_]+\}?|secure_password|redacted)$"
)

DB_URL = re.compile(r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|amqp)(?:\+\w+)?://"
                    r"(?P<user>[^\s:/@]+):(?P<pw>[^\s@/]+)@(?P<host>[^\s/:]+)")
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
EMAIL_ALLOW = re.compile(
    r"(?i)^(?:semyon\.vadishev@gmail\.com|noreply@anthropic\.com|[^@]+@users\.noreply\.github\.com"
    r"|[^@]+@example\.(?:com|org|net)|[^@]+@[^@]*\.example\.(?:com|org|net)|you@your-?domain\.\w+)$"
)
PERSONAL_PATH = re.compile(r"(?:/Users|/home)/(?P<user>[A-Za-z0-9._-]+)|C:\\Users\\(?P<win>[^\\\s]+)")
PATH_USER_ALLOW = {"dev", "user", "you", "username", "runner", "me", "shared", "ubuntu", "jovyan"}


def _allowed_line(line: str) -> bool:
    return ALLOW_MARKER in line


def scan_text(
    path: str,
    text: str,
    deny: list[DenyEntry] | None = None,
    compiled: list[tuple[DenyEntry, re.Pattern[str]]] | None = None,
) -> list[Finding]:
    """Scan text content (file body or commit message) and return findings."""
    findings: list[Finding] = []
    deny_c = compiled if compiled is not None else [(d, d.compile()) for d in (deny or [])]
    for lineno, line in enumerate(text.splitlines(), 1):
        if _allowed_line(line):
            continue
        for rule, rx in SECRET_PATTERNS:
            for m in rx.finditer(line):
                val = m.groupdict().get("val") or m.group(0)
                if rule == "credential-assignment" and PLACEHOLDER_VALUES.match(val):
                    continue
                findings.append(Finding(rule, path, lineno, redact(val)))
        for m in DB_URL.finditer(line):
            if not PLACEHOLDER_VALUES.match(m["pw"]):
                findings.append(Finding("db-url-password", path, lineno, f"…:{redact(m['pw'])}@{m['host']}"))
        for m in EMAIL.finditer(line):
            if not EMAIL_ALLOW.match(m.group(0)):
                findings.append(Finding("email", path, lineno, redact(m.group(0))))
        for m in PERSONAL_PATH.finditer(line):
            user = m["user"] or m["win"]
            if user.lower() not in PATH_USER_ALLOW:
                findings.append(Finding("personal-path", path, lineno, m.group(0).replace(user, redact(user))))
        for entry, rx in deny_c:
            if rx.search(line):
                label = entry.label or entry.kind
                findings.append(Finding(f"denylist:{label}", path, lineno, "private value (redacted)"))
    return findings


# --- file rules ---------------------------------------------------------------------------

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".bmp", ".tif", ".tiff"}
DATA_EXT = {".csv", ".tsv", ".db", ".sqlite", ".sqlite3", ".parquet", ".jsonl", ".ndjson", ".xlsx",
            ".dump", ".sql.gz"}
MODEL_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".tflite", ".mlmodel", ".h5", ".npz"}
KEY_EXT = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".mobileprovision", ".p8"}
SECRET_FILES = {"kaggle.json", ".modal.toml", "credentials.json", "service-account.json", ".netrc",
                ".pgpass", "id_rsa", "id_ed25519", "id_ecdsa", ".npmrc", ".pypirc"}
SESSION_DIRS = (".claude/", ".superpowers/", ".worktrees/", ".idea/", ".vscode/", "wandb/",
                "checkpoints/", "kaggle_ckpts/", "runs/", "data/")
IMAGE_DIRS = ("docs/assets/", "docs/", "picode-ios/picode-ios/App/Assets.xcassets/")
FIXTURE_DIRS = ("tests/fixtures/", "picode-scraper/tests/fixtures/", "picode-model/picode/tests/fixtures/")


def scan_path(path: str, size: int | None = None) -> list[Finding]:
    """Rules that depend only on the file path (and size)."""
    p = PurePosixPath(path)
    name = p.name
    suffix = "".join(p.suffixes[-2:]) if name.endswith(".sql.gz") else p.suffix.lower()
    out: list[Finding] = []

    def add(rule: str, why: str) -> None:
        out.append(Finding(rule, path, 0, why))

    if (name == ".env" or name.startswith(".env.")) and name != ".env.example":
        add("forbidden-file", "environment file; use .env.example with placeholders")
    if suffix in KEY_EXT or name in SECRET_FILES:
        add("forbidden-file", "key or credential file")
    for d in SESSION_DIRS:
        if f"/{d}" in f"/{path}" and not path.startswith(("picode-model/picode/", "picode-scraper/picode_scraper/")):
            add("forbidden-file", f"local tool/session/data directory ({d})")
            break
    if name == "CLAUDE.md" or name == ".coverage":
        add("forbidden-file", "local assistant notes / coverage data")
    if suffix in MODEL_EXT or name.endswith((".mlpackage", ".mlmodelc")) or ".mlpackage/" in path:
        add("forbidden-file", "model weights belong on Hugging Face, not in git")
    if suffix in DATA_EXT and not path.startswith(FIXTURE_DIRS) and "/tests/fixtures/" not in path:
        add("data-export", "data export outside a tests/fixtures directory")
    if suffix in IMAGE_EXT and not (path.startswith(IMAGE_DIRS) or "/tests/fixtures/" in path):
        add("image-location", "images belong in docs/assets/ (generated from licensed sources)")
    if size is not None:
        limit = MAX_ASSET_BYTES if path.startswith("docs/assets/") else MAX_FILE_BYTES
        if size > limit:
            add("large-file", f"{size / 1e6:.1f} MB > {limit / 1e6:.0f} MB limit")
    return out


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:8000]


# --- Bash command rules (Claude Code PreToolUse) --------------------------------------------

_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\s*(?:\n|$)", re.S)
_QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")

BASH_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("no-verify", re.compile(r"(?<![\w-])--no-verify\b")),
    ("commit -n", re.compile(r"\bgit\b[^;&|]*\bcommit\b[^;&|]*\s-[a-zA-Z]*n[a-zA-Z]*\b")),
    ("force push", re.compile(r"\bgit\b[^;&|]*\bpush\b[^;&|]*(?:\s--force(?:-with-lease)?\b|\s-[a-zA-Z]*f\b|\s\+\S+)")),
    ("hooksPath change", re.compile(r"\bcore\.hooksPath\b|\bgit\b[^;&|]*-c\s*core\.hookspath", re.I)),
    ("filter-repo", re.compile(r"\bgit[- ]filter-repo\b|\bgit_filter_repo\b|\bfilter-branch\b")),
    ("guard override", re.compile(r"\bPICODE_GUARD_OVERRIDE\b")),
    ("hook bypass", re.compile(r"\bHUSKY=0\b|\bSKIP=|\bGIT_HOOKS_DISABLED\b")),
]


def strip_quoted(command: str) -> str:
    """Drop heredoc bodies and quoted strings so their contents cannot trigger rules."""
    command = _HEREDOC.sub(" ", command)
    return _QUOTED.sub("''", command)


def scan_bash(command: str) -> list[str]:
    """Return the names of blocked patterns found in a shell command."""
    visible = strip_quoted(command)
    hits = [name for name, rx in BASH_RULES if rx.search(visible)]
    # Setting the override env var inside quotes (e.g. env "PICODE_GUARD_OVERRIDE=1") is still an override.
    if "guard override" not in hits and re.search(r"PICODE_GUARD_OVERRIDE\s*=", command):
        hits.append("guard override")
    return hits
