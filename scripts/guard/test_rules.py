"""Tests for the leak-guard rules. Fake secrets are assembled at runtime so this file
passes its own guard. Run: python3 -m unittest scripts/guard/test_rules.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules  # noqa: E402


def j(*parts: str) -> str:
    return "".join(parts)


def rules_hit(text: str, deny: list[rules.DenyEntry] | None = None) -> set[str]:
    return {f.rule for f in rules.scan_text("x.py", text, deny)}


class SecretFormats(unittest.TestCase):
    def test_tokens(self) -> None:
        cases = {
            "aws-access-key": j("AKIA", "Z" * 16),
            "github-token": j("gh", "p_", "a1" * 18),
            "huggingface-token": j("hf", "_", "Ab3" * 11),
            "modal-token": j("ak", "-", "Q9w" * 8),
            "openai-key": j("sk", "-", "proj-", "x7" * 12),
            "google-api-key": j("AI", "za", "B" * 35),
            "slack-token": j("xo", "xb-", "1234567890-abc"),
            "private-key": j("-----BEGIN ", "RSA PRIVATE", " KEY-----"),
        }
        for rule, value in cases.items():
            with self.subTest(rule=rule):
                self.assertIn(rule, rules_hit(f"x = '{value}'"))

    def test_credential_assignment(self) -> None:
        self.assertIn("credential-assignment", rules_hit(j("api", "_key = 'Zx81kd92JDk20dkLq'")))
        self.assertNotIn("credential-assignment", rules_hit("api_key = 'your-api-key-here'"))
        self.assertNotIn("credential-assignment", rules_hit("password = ${DB_PASSWORD}"))

    def test_db_url(self) -> None:
        self.assertIn("db-url-password", rules_hit(j("postgresql://app:", "Hunter2x9", "@db.internal:5432/x")))
        for ok in ("postgresql://picode:picode@db:5432/p", "postgresql://user:pass@localhost/p",
                   "postgresql://picode:change-me@localhost:5432/p",
                   "postgresql://picode:secure_password@db.example.com:5432/p"):
            self.assertNotIn("db-url-password", rules_hit(ok), ok)

    def test_redaction(self) -> None:
        secret = j("hf", "_", "Zq8" * 11)
        out = str(rules.scan_text("f", secret)[0])
        self.assertNotIn(secret, out)


class PersonalData(unittest.TestCase):
    def test_emails(self) -> None:
        self.assertIn("email", rules_hit(j("contact: jane.doe", "@", "corp.io")))
        for ok in ("semyon.vadishev@gmail.com", "noreply@anthropic.com", "bob@example.com",
                   "12345+vs@users.noreply.github.com"):
            self.assertNotIn("email", rules_hit(ok), ok)

    def test_paths(self) -> None:
        self.assertIn("personal-path", rules_hit(j("/Us", "ers/alice/src/x")))
        self.assertIn("personal-path", rules_hit(j("/ho", "me/bob/.ssh")))
        self.assertNotIn("personal-path", rules_hit("/home/runner/work and /Users/dev/x"))

    def test_allow_marker(self) -> None:
        self.assertEqual(rules_hit(j("jane", "@", "corp.io  # guard", ":allow")), set())

    def test_denylist_kinds(self) -> None:
        deny = rules.parse_denylist(
            "literal\tS3cr3tV4lue\tenv:TOKEN\n"
            "word-ci\tacme-corp\tcompany\n"
            "ticker\tNVDA\tticker\n"
            "regex\tU\\d{7}\taccount\n"
        )
        self.assertIn("denylist:env:TOKEN", rules_hit("x S3cr3tV4lue y", deny))
        self.assertIn("denylist:company", rules_hit("ACME-Corp report", deny))
        self.assertNotIn("denylist:company", rules_hit("acme-corporation", deny))
        self.assertIn("denylist:ticker", rules_hit("bought NVDA", deny))
        self.assertNotIn("denylist:ticker", rules_hit("nvda lower", deny))
        self.assertIn("denylist:account", rules_hit("acct U1234567", deny))
        f = rules.scan_text("x", "x S3cr3tV4lue", deny)[0]
        self.assertNotIn("S3cr3tV4lue", str(f))


class FileRules(unittest.TestCase):
    def check(self, path: str, size: int | None = None) -> set[str]:
        return {f.rule for f in rules.scan_path(path, size)}

    def test_forbidden(self) -> None:
        for p in (".env", "picode-scraper/.env.local", "keys/server.pem", "kaggle.json",
                  ".claude/settings.local.json", "CLAUDE.md", "picode-model/checkpoints/best.pt",
                  "model.safetensors", ".worktrees/x/y.py"):
            self.assertIn("forbidden-file", self.check(p), p)
        self.assertNotIn("forbidden-file", self.check(".env.example"))
        self.assertNotIn("forbidden-file", self.check("picode-model/picode/training/data.py"))

    def test_data_and_images(self) -> None:
        self.assertIn("data-export", self.check("exports/holdings.csv"))
        self.assertNotIn("data-export", self.check("picode-scraper/tests/fixtures/pairs.csv"))
        self.assertIn("image-location", self.check("screenshot.png"))
        self.assertNotIn("image-location", self.check("docs/assets/hero.png"))

    def test_large(self) -> None:
        self.assertIn("large-file", self.check("picode-model/x.bin", 2_000_000))
        self.assertNotIn("large-file", self.check("docs/assets/hero.png", 2_200_000))


class BashRules(unittest.TestCase):
    def test_blocked(self) -> None:
        for cmd in ("git commit --no-verify -m x", "git commit -nm x", "git commit -am x -n",
                    "git push --force origin main", "git push -f", "git push origin +main",
                    "git push --force-with-lease=main:abc origin main",
                    "git config core.hooksPath /dev/null", "git -c core.hooksPath=x commit",
                    "git filter-repo --path x", j("PICODE_GUARD_", "OVERRIDE=1 git commit"),
                    j("env PICODE_GUARD_", "OVERRIDE=1 git push")):
            self.assertTrue(rules.scan_bash(cmd), cmd)

    def test_allowed(self) -> None:
        for cmd in ('git commit -m "docs: explain --no-verify is blocked"',
                    "git commit -F - <<'EOF'\nnever use git push --force\nEOF\n",
                    "git push origin main", "git commit -m 'refactor -n flag'", "grep -n foo x.py",
                    "git log -n 5", "ls -f"):
            self.assertEqual(rules.scan_bash(cmd), [], cmd)


if __name__ == "__main__":
    unittest.main()
