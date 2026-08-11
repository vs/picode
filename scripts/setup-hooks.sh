#!/bin/sh
# Enable the repository's git hooks (leak guard). Run once after cloning.
set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Git hooks enabled (core.hooksPath=.githooks)."
echo "Owner only: build the private denylist with: python3 scripts/guard/refresh_denylist.py"
