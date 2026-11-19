#!/bin/bash
# Sets xblick-sam3 up from scratch or updates it: the venv exactly as uv.lock fixes it (.venv; SAM 3's code from
# GitHub at the locked commit, Durchblick from github.com/xblick/xblick) and, through xblick-sam3 --download-only
# (weights.py), SAM 3.1's checkpoint (3.5 GB, SAM License, from Hugging Face's gated facebook/sam3.1; only if missing,
# checked against its SHA-256) in ~/.cache/xblick/sam3 (the node's cache directory). The node downloads it at its start too.
#
#     tools/setup.sh
#
# Before it downloads the checkpoint, xblick-sam3 prints the model's terms (weights.TERMS): whoever runs the script
# is the one who uses the model.
#
# Needs uv (UV names it, default uv on PATH), git, and a Hugging Face account with access to
# facebook/sam3.1, logged in once (`.venv/bin/hf auth login`) or its token in HF_TOKEN. Without git or uv it stops
# with the command that installs them. Run again, it does only what changed. It never asks and never uses sudo.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV=${UV:-uv}
PROG=xblick-sam3

usage() {
    cat <<'EOF'
Usage: tools/setup.sh

Sets xblick-sam3 up in this clone, or updates it. It creates .venv exactly as uv.lock fixes it (Python 3.12, PyTorch
for CUDA 13, SAM 3's code from GitHub, the core xblick from PyPI) and downloads SAM 3.1's checkpoint into the node's
cache directory (~/.cache/xblick/sam3). The checkpoint is gated on Hugging Face: it needs an account with access to
facebook/sam3.1, logged in once with .venv/bin/hf auth login, or its token in HF_TOKEN. A second call does only what
changed. It never asks and never uses sudo. Needs git and uv (UV=/path/to/uv names another uv). The node needs an
NVIDIA GPU with driver 580 or newer.

  -h, --help    this text
EOF
}

die() {
    echo "setup.sh: $*" >&2
    exit 1
}

for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    die "the node needs Linux on x86-64 (the core's wheels), this is $(uname -s) $(uname -m)"
command -v git > /dev/null || die "no git: sudo apt install git"
command -v "$UV" > /dev/null ||
    die "no uv ($UV): install it with  wget -qO- https://astral.sh/uv/install.sh | sh  and open a new terminal"
for arg in "$@"; do
    case $arg in
        -h | --help) ;;
        *) usage >&2; die "unknown option: $arg" ;;
    esac
done
MIN_DRIVER=580  # NVIDIA's minimum Linux driver for CUDA 13.0, which the cu130 torch is built for
if ! command -v nvidia-smi > /dev/null; then
    echo "$PROG: no NVIDIA driver found (no nvidia-smi). The node needs an NVIDIA GPU with driver $MIN_DRIVER or" \
        "newer to start (Ubuntu: sudo apt install nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
elif ! driver=$(timeout 30 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null) ||
        [[ -z $driver ]]; then
    echo "$PROG: nvidia-smi failed or lists no GPU. The node needs an NVIDIA GPU with driver $MIN_DRIVER or newer" \
        "to start. The setup goes on."
else
    driver=$(cut -d. -f1 <<< "$driver" | sort -n | head -1 | tr -d ' ')
    ((driver >= MIN_DRIVER)) ||
        echo "$PROG: NVIDIA driver $driver. PyTorch for CUDA 13 needs $MIN_DRIVER or newer (Ubuntu: sudo apt install" \
            "nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
fi
cd "$ROOT"

cat <<EOF
$PROG: setting up $PWD. It downloads what is missing of:
  * .venv as uv.lock fixes it: Python 3.12, the core xblick, and PyTorch for CUDA 13 with NVIDIA's CUDA libraries
    (NVIDIA's licence, License.txt in each nvidia package); about 3 GB, 6 GB on disk
  * SAM 3's code (github.com/facebookresearch/sam3, the commit uv.lock fixes) into .venv
  * SAM 3.1's checkpoint (3.5 GB, Hugging Face facebook/sam3.1, gated) into the node's cache directory, after the
    terms of the SAM License. Without access or login the download stops and names the next step.
EOF
"$UV" sync --frozen
.venv/bin/xblick-sam3 --download-only
echo "$PROG: set up. The commands are in .venv/bin/; the first start: README.md, Install."
