# Changelog

## Branch ros2 (not released)

The node with the core's option `--ros` (ROS 2 on `rmw_zenoh`).

### Changed

- The core comes from its unreleased branch `ros2` (`xblick @ git+https://github.com/xblick/xblick@ros2`), not from
  PyPI. `uv sync` builds its C++ part from source.

### Added

- `--ros`, `--ros-domain`, `--ros-node` and `--ros-params-writable` from the core. With `--ros` the detections and the
  instances go to ROS 2 as well, and the parameters are a ROS node. README: section "ROS 2".
- `tests/test_ros.py`: a stand-in model (no checkpoint, a CUDA device); with `--ros` a ROS subscriber gets each output
  with a ROS type, with ROS's attachment and tokens; without it nothing of ROS is on the wire.

Compatible with earlier clients: nothing on our keys changed.

## 0.2.5 (2026-11-15)

The prompts queryable answers at once, and a fixed GPU.

### Changed

- `--device cuda` is GPU 0, `cuda:N` GPU N. Another value, or a GPU that cannot be used, ends the node with one line.
- The start line says where the model computes (`model on cuda:0`).
- README: how to cite SAM 3.

### Fixed

- The queryable `prompts` answers at once, also while the model works on a frame. New prompts are checked, answered and
  segmented from the next frame on.
- `xblick-sam3-prompts` no longer waits past its timeout when Zenoh does not end a query.

Compatible with earlier clients: nothing on the wire changed.

## 0.2.4 (2026-11-14)

- The weights moved out of the checkout into the node's cache directory, `~/.cache/xblick/sam3/`. `XBLICK_HOME` and
  `XDG_CACHE_HOME` move it, and `--weights-dir DIR` names another directory. Weights at the former place, the
  checkout's `checkpoints/`, are not used. The node prints the command that moves them. In the checkout it is
  `mkdir -p ~/.cache/xblick && mv checkpoints ~/.cache/xblick/sam3`.
- `--weights-dir DIR` is a new start option and a read-only parameter.
- The message about a missing file names the full download command.
- The node needs `xblick>=0.5.5,<0.6`.
- `Segmenter` takes SAM 3.1's checkpoint from the node's cache directory unless given one; the node passes the
  file of its store (`--weights-dir`) or `--checkpoint`.

## 0.2.3 (2026-11-14)

- The download of the weights is the core's `xblick.weights` (xblick 0.5.3). The node's own copy of that code is
  gone. `weights.py` keeps the node's files, its terms notice and its `STORE`. The options, the messages, and the
  files and their places stay as they were.
- The node needs `xblick>=0.5.3,<0.6`.
- `tests/test_weights.py` keeps the node's own checks. The core's tests cover the download itself.

## 0.2.2 (2026-11-14)

- The node downloads a missing SAM 3.1 checkpoint at its start, before the model loads. It prints the terms of the
  SAM License, the file and its size first. With the checkpoint there it uses no network and writes no file.
- `--download-only` downloads and checks the checkpoint, then exits 0. It needs no camera, Zenoh router or GPU, and
  `--camera` may be left out. `tools/setup.sh` calls it after `uv sync`.
- `--no-download` ends a start at once when the checkpoint is missing and names it. It is a read-only parameter.
- Without access to `facebook/sam3.1` or a login, the node ends with one line that names the page, `hf auth login`
  and `HF_TOKEN`.
- `weights.py` is the one place of the checkpoint (`segmenter.CHECKPOINT` comes from there) and of the terms, which
  moved from `tools/setup.sh` unchanged. `tools/setup.sh` now prints them only when the checkpoint is downloaded,
  after `uv sync` has fetched SAM 3's code.
- A download goes to `<name>.part` under the lock `checkpoints/.download.lock` and gets its final name after the
  SHA-256 check.
- The download takes the checkpoint from a fixed commit of `facebook/sam3.1` (daa6319, where `main` was on
  2026-11-14), no longer from `main`. Hugging Face states the same SHA-256 for the file there. A download is also
  checked against the exact size.

## 0.2.1 (2026-11-14)

`src/xblick_sam3/segmenter.py` is now under Meta's SAM License (`LICENSES/SAM-License.txt`): it assembles and adapts
SAM 3's detector from Meta's model code and is a derivative work of SAM 3. Everything else stays under the Apache
License 2.0. The README names the terms of the SAM License that bind whoever uses the node. No change of the code.

`tools/setup.sh` prints these terms before it downloads SAM 3's code and checkpoint.

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-sam3 | github.com/xblick/xblick-sam3 |
| Distribution | `durchblick-sam3` | `xblick-sam3` |
| Python package | `durchblick_sam3` | `xblick_sam3` |
| Commands | `durchblick-sam3`, `durchblick-sam3-prompts` | `xblick-sam3`, `xblick-sam3-prompts` |
| Program name in log lines, the logger and reports | `durchblick-sam3` | `xblick-sam3` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Other nodes named in the docs | `durchblick-foundationpose`, `durchblick-tapnext`, `durchblick-mapping` | `xblick-foundationpose`, `xblick-tapnext`, `xblick-mapping` |

## 0.1.0

The node as `durchblick-sam3` (Python package `durchblick_sam3`) on the core `blick` 0.4.
