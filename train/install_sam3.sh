#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/sam3.git"
REPO_DIR="sam3"

echo "Cloning repository..."
git clone -q "$REPO_URL" "$REPO_DIR"

echo "Removing pinned numpy dependency from pyproject.toml..."
sed -E -i \
    's/"numpy==[^"]+",?[[:space:]]*//g' \
    "$REPO_DIR/pyproject.toml"

echo "Downloading CLIP vocabulary..."
mkdir -p "$REPO_DIR/assets"
wget -q \
    https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
    -O "$REPO_DIR/assets/bpe_simple_vocab_16e6.txt.gz"

echo "Installing package..."
cd "$REPO_DIR"
python3 -m pip install -e .

echo "Done."
