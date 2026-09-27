#!/usr/bin/env bash
# Download the Space Weather: Solar + Geomagnetic Indices dataset from Kaggle
# into data/raw/ using the Kaggle CLI via uv (no global install needed).
set -euo pipefail

DATASET="erevear/space-weather-solar-geomagnetic-indices"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$REPO_ROOT/data/raw"

if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  echo "Using KAGGLE_USERNAME / KAGGLE_KEY from the environment."
elif [ -f "$HOME/.kaggle/access_token" ]; then
  chmod 600 "$HOME/.kaggle/access_token"
  echo "Using ~/.kaggle/access_token (Kaggle API token)"
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "Using ~/.kaggle/kaggle.json"
else
  cat >&2 <<'EOF'
Error: Kaggle credentials not found.

Provide one of:
  1. API token:   echo "KGAT_..." > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
  2. Legacy file: ~/.kaggle/kaggle.json (chmod 600)
  3. Env vars:    export KAGGLE_USERNAME=... KAGGLE_KEY=...

Create a token at https://www.kaggle.com/settings -> API -> Create New Token.
EOF
  exit 1
fi

mkdir -p "$DEST"

uvx --from kaggle kaggle datasets download \
  -d "$DATASET" \
  -p "$DEST" \
  --unzip

echo "Files now in $DEST:"
ls -la "$DEST"
