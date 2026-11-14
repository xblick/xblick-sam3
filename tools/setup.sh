#!/bin/bash
# Sets xblick-sam3 up from scratch or updates it: the venv exactly as uv.lock fixes it (.venv; SAM 3's code from
# GitHub at the locked commit, Durchblick from github.com/xblick/xblick) and SAM 3.1's checkpoint (3.5 GB, SAM
# License, from Hugging Face's gated facebook/sam3.1; only if missing, checked against its SHA-256) in checkpoints/.
#
#     tools/setup.sh
#
# Before it downloads a model it prints the model's terms: whoever runs the script is the one who uses the model.
#
# Needs uv (UV names it, default uv on PATH), git, and a Hugging Face account with access to
# facebook/sam3.1, logged in once (`.venv/bin/hf auth login`) or its token in HF_TOKEN. Run again, it does only what
# changed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
cd "$ROOT"

# The terms of what is downloaded, printed before the download. $1: what is missing.
model_terms() {
    cat <<TERMS
xblick-sam3: to download: $1
SAM 3 is the work of Meta, under its terms, and you are the one who uses it:
  * SAM 3's code (github.com/facebookresearch/sam3) and the SAM 3.1 checkpoint (Hugging Face facebook/sam3.1,
    gated) are under Meta's SAM License: LICENSES/SAM-License.txt in this repository.
  * You may not use them for military or warfare purposes, nuclear industries or applications, espionage, the
    development or use of guns or illegal weapons, or activities subject to ITAR (clause 1.b.v).
  * A publication of research done with them must acknowledge their use (clause 1.b.ii).
  * You may pass them or derivative works of them on only under the SAM License (clause 1.b.i).
TERMS
}

CKPT=$ROOT/checkpoints/sam3.1_multiplex.pt
[ -e "$CKPT" ] || model_terms "SAM 3's code and the checkpoint sam3.1_multiplex.pt (3.5 GB)"
"$UV" sync --frozen
if [ ! -e "$CKPT" ]; then
  echo "downloading $CKPT"
  .venv/bin/python -I -c "from huggingface_hub import hf_hub_download as d; d('facebook/sam3.1', \
'sam3.1_multiplex.pt', local_dir='$ROOT/checkpoints')"
fi
echo "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6  $CKPT" | sha256sum --check --quiet - \
  || { echo "$CKPT: wrong checksum (remove it, then run again)" >&2; exit 1; }
