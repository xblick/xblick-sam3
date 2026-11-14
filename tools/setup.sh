#!/bin/bash
# Sets xblick-sam3 up from scratch or updates it: the venv exactly as uv.lock fixes it (.venv; SAM 3's code from
# GitHub at the locked commit, Durchblick from github.com/xblick/xblick) and, through xblick-sam3 --download-only
# (weights.py), SAM 3.1's checkpoint (3.5 GB, SAM License, from Hugging Face's gated facebook/sam3.1; only if missing,
# checked against its SHA-256) in checkpoints/. The node downloads it at its start too.
#
#     tools/setup.sh
#
# Before it downloads the checkpoint, xblick-sam3 prints the model's terms (weights.TERMS): whoever runs the script
# is the one who uses the model.
#
# Needs uv (UV names it, default uv on PATH), git, and a Hugging Face account with access to
# facebook/sam3.1, logged in once (`.venv/bin/hf auth login`) or its token in HF_TOKEN. Run again, it does only what
# changed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
cd "$ROOT"

"$UV" sync --frozen
.venv/bin/xblick-sam3 --download-only
