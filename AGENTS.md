# AGENTS.md

xblick-sam3: Meta's SAM 3.1 detector as a Durchblick node. Per colour frame it publishes the instance masks of
the prompts (words, exemplar boxes) and, with `--boxes TOPIC`, one mask per given box. Human documentation: README.md
(entry), docs/topics.md, docs/prompts.md, docs/performance.md, docs/design.md. Style of all
documentation: the core's docs/project/writing-docs.md. Transport-wide traps (Zenoh sessions and routers, CUDA rings,
hanging tests, shared filesystems): AGENTS.md of the core repository (github.com/xblick/xblick).

## Commands

```bash
tools/setup.sh                                  # .venv as uv.lock fixes it, then xblick-sam3 --download-only
.venv/bin/python -m pytest -q tests             # GPU tests skip without CUDA, need ~/.cache/xblick/sam3/sam3.1_multiplex.pt
.venv/bin/python -m pytest -q tests/test_weights.py  # CPU, 4 s: the node's files and its start up to the weights (the download: the core's tests)
.venv/bin/xblick-sam3 --download-only           # SAM 3.1 checkpoint (3.5 GB) if missing, SHA-256 checked, exit 0
.venv/bin/python -m pytest -q tests/test_node.py
.venv/bin/ruff check src tests                  # lint, 120 columns (pyproject)
.venv/bin/python tools/bench.py IMAGES_DIR --words cup bottle --compile default --resolution 672   # speed tables
uv lock                                         # after a new core rev in pyproject (pin xblick>=0.5.14,<0.6)
.venv/bin/xblick-sam3 --camera test/camera --words "cup,bottle" --compile default   # needs zenohd on 7447
```

- Checkpoint is gated: Hugging Face account with access to `facebook/sam3.1`, logged in once
  (`.venv/bin/hf auth login`) or `HF_TOKEN`, before the first download (`tools/setup.sh`, `--download-only` or the
  first start). Without: one line naming the page, the login and `HF_TOKEN`, exit 1.
- `--no-download`: a start with a missing checkpoint ends with one line (exit 1). Excludes `--download-only`.
- SAM 3's code: from GitHub at the commit in `uv.lock`. Durchblick (`xblick`): at the commit in `pyproject.toml`.
- `--compile default`: about 45 s of compiling before the first frame.
- Without a camera: a simulation that publishes `robot/camera`, or `xblick play <recording.mcap> --loop`.

## Rules

- Files: the weights are in the node's cache directory (`xblick.paths.cache_dir("sam3")`, `~/.cache/xblick/sam3`), never in
  the checkout. `weights.STORE = Store(PROG, node=NODE, ..., former=[FORMER])`, `Weight` paths relative to its root;
  `store = weights.store_for(parser, args)` (`--weights-dir`) for `download_only`, `ensure` and `fail`, with
  `options=weights.options(args)`; model code takes its paths from `store.root`. `FORMER` (the checkout's `checkpoints/`) is never
  read: the store prints the command that moves it. The core's `docs/reference/files.md` has the rule.
- Fail fast, no silent fallbacks. Refused prompts answer with an error and leave the prompts unchanged. With
  `CUDA_MPS_PIPE_DIRECTORY` set, the node stops at start when no MPS daemon answers. Keep it so.
- The queryable answers in Zenoh's thread at once: `answer` checks a request without the model
  (`segmenter.check_prompts`) and accepts it, and the loop sets it (`apply_prompts`) before the next frame. Never
  answer from the frame loop: a query still open at the caller's timeout gets a `Timeout` reply, which a sender takes
  for a second node. `tests/slow_step.py` slows the model for the test of it.
- Weights: `weights.py` is the one place (`CHECKPOINT`, its repository, SHA-256, size, `TERMS`; `segmenter.py`
  imports `CHECKPOINT`). `weights.STORE.ensure` runs at the start before the model loads; `--download-only` runs it
  alone. File there: no network, no file written (`test_weights.py` checks it). Keep it so.
- Downloads: `<name>.part` in `~/.cache/xblick/sam3/`, renamed after the SHA-256 check, under the lock
  `~/.cache/xblick/sam3/.download.lock`. Never let a partial or unchecked file get its final name. No fallbacks.
- `--download-only` and `--camera`: `xblick.weights.parse_args` lifts `required` when the token `--download-only` is
  spelled out. `download_only` is in `leave_out` of `declare_options`; `no_download` is a read-only parameter.
- A wrong SHA-256 keeps `<name>.part` (the error names it: remove it). Ctrl-C during a download: exit 130, SIGTERM:
  143 (`xblick.weights`), the `.part` file removed.
- Licence separation: SAM 3's code is a dependency, the checkpoint a download. Copy nothing of either into the repo.
- README.md and docs/ are the documentation of record. Change topics, options, speed tables and tests with the code.
  Report measured numbers with what was measured, on which GPU and when.
- Node default and class default differ: node `--resolution 672`, Python `Segmenter` 1008 unless `resolution=672`.
  Do not change one and assume the other followed.
- Any speed change to the segmenter keeps `tests/test_segmenter.py` green (masks against SAM 3's own forward pass,
  IoU > 0.99).
- Match the surrounding code. 120 columns, ruff rules in `pyproject.toml`.

## Traps (each cost hours)

- **bfloat16 and RoPE.** Symptom: SAM 3 finds nothing, no error. Cause: `Module.to(bf16)` also casts the ViT's complex
  rotary-embedding buffers and drops their imaginary part. Fix: cast the image encoder's parameters only, once. The
  decoder turns autocast off on purpose. Leave it.
- **Resolution.** Only multiples of 336 (whole 24x24 attention windows). `segmenter.resize` recomputes everything that
  depends on the grid (global blocks' RoPE, decoder box position-bias coordinates, neck position encodings). Add any
  new grid-dependent cache there.
- **Exemplar boxes.** Symptom: wrong objects on later frames (measured). Cause: re-encoding the box on a later frame
  pools whatever is there by then. Fix: encode on the box's own frame, once, reuse. Prompts have a `stamp`. The node
  keeps `--history` frames (30).
- **`--compile`.** Compiled statically, a new prompt count stalls 2 s, a new exemplar count 20-27 s. Fix in place:
  grounding transformer and mask head compiled `dynamic=True`, `warm_up` compiles both graphs (one prompt, several).
  Tried without gain: `max-autotune`, `reduce-overhead`, FP8, token-major relayout of the attention inputs.
- **Instance encoding.** `mono16`, never `16UC1`. REP 118 makes `16UC1` depth, and viewers draw it as a point cloud.
- **SAM 3's packaging.** Misses imports (psutil, scipy, cv2, einops, pycocotools), needs `setuptools<81`
  (`pkg_resources` for its tokenizer). All listed in `pyproject.toml`. Keep them when bumping SAM 3.
- **SAM 3.1's Object Multiplex tracker.** Unused on purpose: slower, loads a whole video up front, yields batches,
  `start_session` passes `offload_state_to_cpu`, which the multiplex `init_state` lacks. Numbers: docs/design.md.
- **Queries in tests.** Symptom: query gets no answer. Cause: a session with only `--connect` does not route queries
  between two clients. Fix: give the node `--listen` (as `tests/test_node.py`), or no endpoints with a router on 7447.
- **Hanging tests.** Zenoh threads keep a process alive. Run node tests in the background with `timeout`.
- **`--boxes`.** The node then follows the pace of the box messages. It segments each on the colour frame of its
  stamp. It counts and drops box messages whose frame is gone (older than `--history`, or not there within 0.5 s).

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_sam3/segmenter.py` | `Segmenter`, `Prompt`, `BoxHead`: batched words, `resize`, compile and warm-up |
| `src/xblick_sam3/tracking.py` | id tracker: image motion from the masks (`--motion`), overlap, position and size (`--match-iou`, `--match-distance`, `--match-size`), merge and split, `--max-missing`; instance image |
| `src/xblick_sam3/node.py` | the node `xblick-sam3`: streams, boxes, prompts queryable, diagnostics |
| `src/xblick_sam3/weights.py` | `CHECKPOINT`, `TERMS`, `needed`, `STORE` (the core's `xblick.weights.Store`: the download, `--download-only`/`--no-download`) |
| `src/xblick_sam3/prompts.py` | client `xblick-sam3-prompts`, request parsing |
| `tools/setup.sh`, `tools/bench.py` | setup, timing on an image directory |
| `tests/`, `tests/data/office.png` | segmenter, node over real Zenoh sessions, tracking, a TUM RGB-D frame |
| `~/.cache/xblick/sam3/` | `sam3.1_multiplex.pt` (downloaded by the node), `.download.lock`: the node's cache directory. The checkout's `checkpoints/` is the former place. |
| `docs/` | topics, prompts, performance, design |

## Defaults

| What | Value |
| --- | --- |
| Input | `<camera>/color` (`rgb8`, `rgba8`, `bgr8`, `bgra8`, `mono8`) with `camera_info` |
| Outputs | `<camera>/sam3/detections_2d`, `<camera>/sam3/instances` (`mono16`), status `sam3` on `<camera>/diagnostics` |
| Queryable | `<camera>/sam3/prompts` (JSON) |
| `--node` | `sam3` |
| `--threshold` / `--match-iou` / `--max-missing` | 0.5 / 0.3 / 30 |
| `--match-distance` / `--match-size` / `--motion` | 0.5 / 2 / `masks` |
| `--resolution` / `--history` / `--max-hz` | 672 / 30 / 30 |
| `--local` / `--device` / `--compile` | `auto` / `cuda` / off |
| Box message wait for its frame | 0.5 s (`FRAME_WAIT_S`) |
| Environment | `HF_TOKEN`, `HF_ENDPOINT`, `HF_HUB_OFFLINE` (download), `UV` (setup), `CUDA_MPS_PIPE_DIRECTORY` (MPS expected), `XBLICK_HOME`, `XDG_CACHE_HOME` (cache directory) |
| Consumers | xblick-foundationpose and the mesh makers read `instances`, xblick-tapnext seeds on it, `xblick rerun --segmentation` draws it |
