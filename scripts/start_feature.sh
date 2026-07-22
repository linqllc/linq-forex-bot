#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <feature-name>" >&2
  exit 2
fi

name="${1// /-}"
name="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"

git diff --quiet && git diff --cached --quiet || {
  echo "Working tree has uncommitted changes. Commit or stash them first." >&2
  exit 1
}

git checkout develop
git pull --ff-only origin develop
git checkout -b "feature/$name"

echo "Created feature/$name"
