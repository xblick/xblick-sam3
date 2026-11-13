#!/bin/bash
# Sets xblick-sam3 up from scratch or updates it: the venv exactly as uv.lock fixes it (.venv; SAM 3's code from
# GitHub at the locked commit, Durchblick from github.com/xblick/xblick) and SAM 3.1's checkpoint (3.5 GB, SAM
# License, from Hugging Face's gated facebook/sam3.1; only if missing, checked against its SHA-256) in checkpoints/.
#
#     tools/setup.sh
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
CKPT=$ROOT/checkpoints/sam3.1_multiplex.pt
if [ ! -e "$CKPT" ]; then
  echo "downloading $CKPT"
  .venv/bin/python -I -c "from huggingface_hub import hf_hub_download as d; d('facebook/sam3.1', \
'sam3.1_multiplex.pt', local_dir='$ROOT/checkpoints')"
fi
echo "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6  $CKPT" | sha256sum --check --quiet - \
  || { echo "$CKPT: wrong checksum (remove it, then run again)" >&2; exit 1; }
