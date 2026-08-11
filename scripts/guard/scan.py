#!/usr/bin/env python3
"""Picode leak guard: blocks secrets and personal data from entering the repository.

Modes:
  --staged            scan the staged diff (pre-commit)
  --commit-msg FILE   scan a commit message (commit-msg)
  --pre-push          scan every outgoing commit; refuse non-fast-forward pushes to main
  --range A..B        scan commits in a range (CI)
  --tree [REV]        scan every file in a tree (default HEAD)
  --claude-hook       Claude Code PreToolUse hook for Write/Edit/MultiEdit/NotebookEdit
  --claude-bash       Claude Code PreToolUse hook for Bash

The private denylist lives outside the repo (see refresh_denylist.py) and is used when
present; CI runs without it. Mark a line with ``guard:allow`` to accept a false positive.
Only the repository owner may set PICODE_GUARD_OVERRIDE=1 to bypass the guard.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules  # noqa: E402

DENYLIST = Path(os.environ.get("PICODE_GUARD_DENYLIST", "~/.config/picode-guard/denylist.txt")).expanduser()
ZERO = "0" * 40
PROTECTED = {"refs/heads/main"}


def git(*args: str, input: bytes | None = None, check: bool = True) -> bytes:
    return subprocess.run(["git", *args], input=input, capture_output=True, check=check).stdout


def load_denylist() -> list[tuple[rules.DenyEntry, object]]:
    if not DENYLIST.is_file():
        return []
    mode = DENYLIST.stat().st_mode & 0o777
    if mode & 0o077:
        print(f"leak-guard: warning: {DENYLIST} is mode {mode:o}; run chmod 600", file=sys.stderr)
    entries = rules.parse_denylist(DENYLIST.read_text(encoding="utf-8"))
    return [(e, e.compile()) for e in entries]


def scan_blob(path: str, data: bytes, deny, size: int | None = None) -> list[rules.Finding]:
    findings = rules.scan_path(path, len(data) if size is None else size)
    if not rules.is_binary(data):
        findings += rules.scan_text(path, data.decode("utf-8", "replace"), compiled=deny)
    return findings


def cat_blobs(shas: list[str]) -> dict[str, bytes]:
    """Read many blobs with one git cat-file --batch (stdin written from a thread)."""
    import threading

    out: dict[str, bytes] = {}
    if not shas:
        return out
    proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def writer() -> None:
        assert proc.stdin
        for s in shas:
            proc.stdin.write(f"{s}\n".encode())
        proc.stdin.close()

    threading.Thread(target=writer, daemon=True).start()
    assert proc.stdout
    for s in shas:
        header = proc.stdout.readline().split()
        if header[1] == b"missing":
            continue
        size = int(header[2])
        out[s] = proc.stdout.read(size)
        proc.stdout.read(1)
    proc.wait()
    return out


def changed_blobs(commit: str) -> list[tuple[str, str]]:
    """(path, blob sha) for files added or modified by a commit (first parent / root)."""
    raw = git("diff-tree", "-r", "--no-commit-id", "--root", "-z", "--diff-filter=AMCRT",
              "--no-renames", "-m", "--first-parent", commit)
    parts = raw.split(b"\0")
    res = []
    i = 0
    while i < len(parts) - 1:
        meta = parts[i].decode()
        if not meta.startswith(":"):
            i += 1
            continue
        path = parts[i + 1].decode("utf-8", "replace")
        res.append((path, meta.split()[3]))
        i += 2
    return res


def scan_commits(commits: list[str], deny) -> list[rules.Finding]:
    findings: list[rules.Finding] = []
    todo: dict[str, set[str]] = {}
    for c in commits:
        msg = git("log", "-1", "--format=%an <%ae>%n%cn <%ce>%n%B", c).decode("utf-8", "replace")
        findings += [rules.Finding(f.rule, f"commit {c[:10]} message", f.line, f.excerpt)
                     for f in rules.scan_text("msg", msg, compiled=deny)]
        for path, blob in changed_blobs(c):
            todo.setdefault(blob, set()).add(path)
    blobs = cat_blobs(list(todo))
    for blob, data in blobs.items():
        for path in sorted(todo[blob]):
            findings += scan_blob(path, data, deny)
    return findings


def report(findings: list[rules.Finding], what: str) -> int:
    uniq = sorted(set(findings), key=lambda f: (f.path, f.line, f.rule))
    if not uniq:
        return 0
    if os.environ.get("PICODE_GUARD_OVERRIDE") == "1":
        print(f"leak-guard: OVERRIDDEN ({len(uniq)} findings in {what})", file=sys.stderr)
        return 0
    print(f"leak-guard: blocked {what} — {len(uniq)} finding(s):", file=sys.stderr)
    for f in uniq[:200]:
        print(f"  {f}", file=sys.stderr)
    if len(uniq) > 200:
        print(f"  … and {len(uniq) - 200} more", file=sys.stderr)
    print("Fix the content, or add 'guard:allow' to a line that is a verified false positive.",
          file=sys.stderr)
    return 1


# --- modes -------------------------------------------------------------------------------

def mode_staged(deny) -> int:
    raw = git("diff", "--cached", "--name-only", "-z", "--diff-filter=AMCRT")
    paths = [p.decode() for p in raw.split(b"\0") if p]
    findings: list[rules.Finding] = []
    entries = {}
    for p in paths:
        line = git("ls-files", "-s", "--", p).decode().split()
        if line:
            entries[line[1]] = p
    blobs = cat_blobs(list(entries))
    for sha, data in blobs.items():
        findings += scan_blob(entries[sha], data, deny)
    return report(findings, "commit")


def mode_commit_msg(path: str, deny) -> int:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return report(rules.scan_text("commit message", text, compiled=deny), "commit message")


def mode_pre_push(deny) -> int:
    findings: list[rules.Finding] = []
    for line in sys.stdin.read().splitlines():
        local_ref, local_sha, remote_ref, remote_sha = line.split()
        if local_sha == ZERO:
            if remote_ref in PROTECTED:
                print(f"leak-guard: refusing to delete {remote_ref}", file=sys.stderr)
                return 1
            continue
        if remote_sha != ZERO and remote_ref in PROTECTED:
            exists = subprocess.run(["git", "cat-file", "-e", f"{remote_sha}^{{commit}}"]).returncode == 0
            ff = exists and subprocess.run(
                ["git", "merge-base", "--is-ancestor", remote_sha, local_sha]).returncode == 0
            if not ff:
                print(f"leak-guard: refusing non-fast-forward push to {remote_ref}", file=sys.stderr)
                return 1
        rng = [local_sha, "--not", "--remotes"] if remote_sha == ZERO else [f"{remote_sha}..{local_sha}"]
        commits = git("rev-list", *rng).decode().split()
        findings += scan_commits(commits, deny)
    return report(findings, "push")


def mode_range(spec: str, deny) -> int:
    commits = git("rev-list", spec).decode().split()
    print(f"leak-guard: scanning {len(commits)} commit(s) in {spec}", file=sys.stderr)
    return report(scan_commits(commits, deny), f"range {spec}")


def mode_tree(rev: str, deny) -> int:
    raw = git("ls-tree", "-r", "-z", "--long", rev)
    todo: dict[str, list[tuple[str, int]]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        _mode, typ, sha, size = meta.split()
        if typ == b"blob":
            todo.setdefault(sha.decode(), []).append((path.decode(), int(size)))
    findings: list[rules.Finding] = []
    for sha, data in cat_blobs(list(todo)).items():
        for path, size in todo[sha]:
            findings += scan_blob(path, data, deny, size)
    return report(findings, f"tree {rev}")


def _repo_root(cwd: str) -> Path | None:
    r = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return Path(r.stdout.strip()) if r.returncode == 0 else None


def mode_claude_hook(deny) -> int:
    """PreToolUse for Write/Edit/MultiEdit/NotebookEdit. Exit 2 blocks the tool call."""
    event = json.load(sys.stdin)
    tool = event.get("tool_name", "")
    inp = event.get("tool_input", {}) or {}
    file_path = inp.get("file_path") or inp.get("notebook_path")
    if not file_path:
        return 0
    root = _repo_root(event.get("cwd") or os.getcwd())
    target = Path(file_path).resolve()
    if root is None or root.resolve() not in target.parents:
        return 0  # outside the repository
    rel = target.relative_to(root.resolve()).as_posix()
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--no-index", rel]).returncode == 0
    if ignored:
        return 0  # git-ignored paths never reach a commit
    if tool == "Write":
        texts = [inp.get("content", "")]
    elif tool == "Edit":
        texts = [inp.get("new_string", "")]
    elif tool == "MultiEdit":
        texts = [e.get("new_string", "") for e in inp.get("edits", [])]
    elif tool == "NotebookEdit":
        texts = [inp.get("new_source", "")]
    else:
        return 0
    findings = rules.scan_path(rel)
    for t in texts:
        findings += rules.scan_text(rel, t or "", compiled=deny)
    if report(findings, f"{tool} to {rel}"):
        return 2
    return 0


def mode_claude_bash() -> int:
    event = json.load(sys.stdin)
    command = (event.get("tool_input") or {}).get("command", "")
    hits = rules.scan_bash(command)
    if hits:
        print(f"leak-guard: blocked Bash command ({', '.join(hits)}). These bypass or rewrite the "
              "repository's safety checks; ask the repository owner to run it themselves.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 64
    mode = argv[0]
    if mode == "--claude-bash":
        return mode_claude_bash()
    deny = load_denylist()
    if mode == "--staged":
        return mode_staged(deny)
    if mode == "--commit-msg":
        return mode_commit_msg(argv[1], deny)
    if mode == "--pre-push":
        return mode_pre_push(deny)
    if mode == "--range":
        return mode_range(argv[1], deny)
    if mode == "--tree":
        return mode_tree(argv[1] if len(argv) > 1 else "HEAD", deny)
    if mode == "--claude-hook":
        return mode_claude_hook(deny)
    print(f"unknown mode {mode}", file=sys.stderr)
    return 64


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:  # fail closed
        print(f"leak-guard: internal error, failing closed: {exc!r}", file=sys.stderr)
        sys.exit(2)
