"""The node's weights: which files it reads, where they come from, and their download.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-sam3
--download-only``) both use it. ``STORE.ensure`` touches neither the disk nor the network when every file is there. A
missing file is downloaded under a temporary name in its directory (``<name>.part``), checked and only then renamed,
all under a file lock, so that two nodes started at once download once and never read half a file.

* ``CHECKPOINT``: SAM 3.1's checkpoint, ``sam3.1_multiplex.pt`` in the node's cache directory
  (``~/.cache/xblick/sam3``, ``--weights-dir``; ``STORE.file``), from Hugging Face's gated
  facebook/sam3.1 (an account with access, logged in with ``hf auth login`` or HF_TOKEN), checked by its SHA-256
* ``needed(checkpoint)``: the file a start with ``--checkpoint`` reads (none for a file of one's own)

SAM 3's code is a dependency of the venv (``uv sync``), no weight file.

The download itself (lock, ``.part`` files, checks, progress, ``--download-only``/``--no-download``) is the core's
``xblick.weights``; this module keeps the node's files, its terms notice and its ``STORE``.
"""

from __future__ import annotations

from pathlib import Path

from xblick.weights import HuggingFace, Store, Weight

PROG = "xblick-sam3"
NODE = "sam3"  # the short name: its cache directory is ~/.cache/xblick/sam3
REQUIRED = ()  # (path, what): code from tools/setup.sh that must be there first; none here
REPO = Path(__file__).resolve().parents[2]
FORMER = REPO / "checkpoints"  # the weights' place before 0.2.4 (the checkout): not used, the store names the move
CHECKPOINT = Path("sam3.1_multiplex.pt")  # below the store's root
# Hugging Face repository (gated) and the commit its main was on (2026-11-14); the file there has CHECKPOINT_SHA256
CHECKPOINT_SOURCE = ("facebook/sam3.1", "daa63191845a41281374e725f4c9e51c7a824460")
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
CHECKPOINT_SIZE = 3_502_755_717
# The terms of what is downloaded, printed before the download (whoever downloads is the one who uses the model).
TERMS = """\
SAM 3 is the work of Meta, under its terms, and you are the one who uses it:
  * SAM 3's code (github.com/facebookresearch/sam3) and the SAM 3.1 checkpoint (Hugging Face facebook/sam3.1,
    gated) are under Meta's SAM License: LICENSES/SAM-License.txt in this repository.
  * You may not use them for military or warfare purposes, nuclear industries or applications, espionage, the
    development or use of guns or illegal weapons, or activities subject to ITAR (clause 1.b.v).
  * A publication of research done with them must acknowledge their use (clause 1.b.ii).
  * You may pass them or derivative works of them on only under the SAM License (clause 1.b.i)."""


def needed(checkpoint: str | None = None) -> list[Weight]:
    """The file of ``--checkpoint``: SAM 3.1's for None (the default), none for a file of one's own."""
    if checkpoint is not None:
        return []
    repo, revision = CHECKPOINT_SOURCE
    return [Weight(CHECKPOINT, HuggingFace(repo, revision, CHECKPOINT.name, gated=True),
                   CHECKPOINT_SIZE, CHECKPOINT_SHA256)]


def options(args) -> list[str]:
    """The options that select the weights (and --weights-dir), for the download command the lines name."""
    return [] + (["--weights-dir", args.weights_dir] if args.weights_dir else [])


def store_for(parser, args) -> Store:
    """STORE at --weights-dir (not given: the node's cache directory); a wrong --weights-dir is an argument error."""
    try:
        return STORE.at(args.weights_dir)
    except ValueError as error:
        parser.error(f"--weights-dir: {error}")


# the download (xblick.weights): the root is the node's cache directory, its lock there
STORE = Store(PROG, node=NODE, terms=TERMS, required=REQUIRED, repo=REPO, former=[FORMER])
