# xblick-sam3

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)
[![tests](https://github.com/xblick/xblick-sam3/actions/workflows/tests.yml/badge.svg)](https://github.com/xblick/xblick-sam3/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0%20%2B%20SAM%20License-blue)](LICENSE)
[![device](https://img.shields.io/badge/needs-NVIDIA%20GPU-76b900)](README.md)

xblick-sam3 segments the objects in a camera's colour stream with Meta's
[SAM 3.1](https://github.com/facebookresearch/sam3), an open-vocabulary instance segmentation model. You name the
objects with words or example boxes, and the node publishes one mask per instance for every frame of a 30 Hz camera.

It is a node of [Durchblick](https://github.com/xblick/xblick). Use it when another node needs
masks of objects that no fixed class list covers. You can change the prompts over Zenoh while the node works.

## What it publishes

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/sam3/detections_2d` | `vision_msgs/Detection2DArray` | One detection per instance with its label, score, box in pixels and tracking id. |
| `<camera>/sam3/instances` | image stream, `mono16` | One image of the colour frame's size. A pixel holds the id of its instance, or 0. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | The status `sam3`, once a second. |
| `<camera>/sam3/prompts` | queryable, JSON | The current prompts. A query with a payload replaces them. |

The node reads `<camera>/color` with its `camera_info`. With `--boxes` it also reads a `Detection2DArray` topic and
publishes one mask per given box. Every result has the stamp and the camera frame of its colour frame.

Details: [docs/topics.md](docs/topics.md).

## Requirements

- An NVIDIA GPU with CUDA. We measured on an RTX PRO 6000 Blackwell. Alone on that GPU the node held 4.7 GiB of
  VRAM (four words, `--compile default`, 1280 x 720 at 30 frames per second, 2026-11-02).
- Durchblick 0.5 or newer, below 0.6 (Python package `xblick`). The setup script installs it from source.
- [uv](https://docs.astral.sh/uv/) and git.
- A Hugging Face account with access to the gated model [facebook/sam3.1](https://huggingface.co/facebook/sam3.1).
- SAM 3's code and the SAM 3.1 checkpoint are under Meta's SAM License. Whoever uses the node uses them and is bound
  by it, see [Licence and credits](#licence-and-credits).

## Install

1. Ask for access on <https://huggingface.co/facebook/sam3.1>.
2. Clone this repository and run the setup script. It creates `.venv` exactly as `uv.lock` fixes it. Then it runs
   `xblick-sam3 --download-only`, which prints the terms of the SAM License and downloads the checkpoint.

   ```bash
   tools/setup.sh
   ```

   The first call stops at the download, because you are not logged in yet. The error names the next step.
3. Log in once, or set `HF_TOKEN` to your token.

   ```bash
   .venv/bin/hf auth login
   ```

4. Download the checkpoint (3.5 GB) into `~/.cache/xblick/sam3/`. The command checks its SHA-256.

   ```bash
   .venv/bin/xblick-sam3 --download-only
   ```

The venv contains PyTorch with CUDA 13 and SAM 3's code from GitHub at the commit that `uv.lock` fixes. If `HF_TOKEN`
is set before step 2, the first call already downloads the checkpoint.

The node itself also downloads a missing checkpoint at its start, before it loads the model. It prints the terms,
the file and its size first. `--download-only` serves preparation scripts: it needs no camera, no Zenoh router and
no GPU, and it exits with 0 when the checkpoint is complete. A machine that must never start a download on its own,
such as a robot, starts the node with `--no-download`. The node then ends at once when the checkpoint is missing,
and it names the file.

### Where the weights are

The node keeps its weights in its cache directory, `~/.cache/xblick/sam3/`. `XBLICK_HOME=/dir` moves it to `/dir/cache/sam3/`,
and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory. The place is the same for a
checkout and for a pip install. [Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows how to copy the cache to a
PC without internet.

Up to version 0.2.3, the weights were in the checkout's `checkpoints/`. The node does not read them there. At its start it
prints the command that moves them. In the checkout it is:

```bash
mkdir -p ~/.cache/xblick && mv checkpoints ~/.cache/xblick/sam3
```

## Run

Start the node on a camera. `test/camera` is an example name. The node connects to the Zenoh router of the
PC, which [Sessions](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md)
explains.

```bash
.venv/bin/xblick-sam3 --camera test/camera --words "cup,bottle,tomato soup can" --compile default
```

`--compile default` compiles the networks for about 45 s before the first frame. After that a frame takes about a
third less time.

Look at the result in Durchblick's viewer:

```bash
xblick rerun --spawn --detections test/camera/sam3/detections_2d \
    --segmentation test/camera/sam3/instances test/camera/color
```

Read the prompts and set new ones while the node works:

```bash
.venv/bin/xblick-sam3-prompts --camera test/camera
.venv/bin/xblick-sam3-prompts --camera test/camera --words bowl "soup can"
```

The options that most users change:

| Option | Default | Meaning |
| --- | --- | --- |
| `--words` | none | The text prompts at the start. Separate words with spaces or commas. Live: it changes while the node runs. |
| `--boxes TOPIC` | off | Also publish one mask per box of this `Detection2DArray` topic. |
| `--threshold` | 0.5 | The lowest score of an instance that the node publishes. Live: it changes while the node runs. |
| `--resolution` | 672 | The size to which the node scales a frame. SAM 3's own size is 1008. |
| `--compile MODE` | off | Compile the networks with `torch.compile` in this mode, for example `default`. |
| `--max-hz` | 30 | The highest number of colour frames per second. `0` or `none` takes every frame. |
| `--local` | `auto` | How the node reads a camera on the same host: `auto`, `shm` or `cuda`. |
| `--node` | `sam3` | The `sam3` part of the topic names. |
| `--no-download` | off | End at once when the checkpoint is missing. The node downloads nothing. |
| `--download-only` | off | Download the missing checkpoint, check it and end. `--camera` may be left out. |
| `--weights-dir DIR` | the cache directory | where the weights are and go, in place of `~/.cache/xblick/sam3/` |

All options, the prompt format and the Python class: [docs/prompts.md](docs/prompts.md).

Every option is also a parameter of the node `<camera>/<node>`, for example `test/camera/sam3`. Durchblick's
terminal interface (`xblick`, tab Parameters) and `xblick param` show them. Four of them change while the node
runs: `threshold`, `match_iou`, `max_missing` and `words`. A new list of words replaces the prompts at the next frame,
and prompts set over `<camera>/<node>/prompts` show in `words`. The other options apply at the next start, and
`--param NAME:=VALUE` or `--params-file FILE` set them there
([Parameters](https://github.com/xblick/xblick/blob/main/docs/reference/parameters.md)).

## Why it is fast

With 8 words, a frame takes 24.4 ms. The same segmenter takes 69.9 ms at SAM 3's own 1008 x 1008 pixels without
compiling.

| Change | Effect with 8 words |
| --- | --- |
| All words pass the model as one batch. | SAM 3's own processor calls the model once per word. |
| The node scales a frame to 672 x 672. | 51.1 ms become 24.4 ms, both compiled. |
| `torch.compile` compiles the three networks. | 36.7 ms become 24.4 ms at 672. |
| The mask head computes masks only for detections. | 11 ms less without compiling. |
| The image encoder stores its weights in bfloat16. | 2 ms less. |

At 672 the node finds 98 to 99 % of the instances again that SAM 3 is sure of at 1008. On one GPU with the robot
mask, Boxer and nvblox it keeps up with a 30 Hz camera.

The measurements, the method and what did not help: [docs/performance.md](docs/performance.md).

## Limits

- The node treats every frame on its own. An object that the detector loses for more than 15 frames gets a new id.
- We measured agreement between 672 and 1008. We did not measure accuracy against ground truth.
- The node accepts only resolutions that are multiples of 336.

## Documentation

- [docs/topics.md](docs/topics.md): inputs, messages and ids in detail.
- [docs/prompts.md](docs/prompts.md): all options, the prompt format, `--boxes`, the Python class.
- [docs/performance.md](docs/performance.md): measurements, what we changed, what did not help.
- [docs/design.md](docs/design.md): how the node works inside, the decisions and their reasons, the tests.
- Durchblick: [Streams and topics](https://github.com/xblick/xblick/blob/main/docs/concepts/streams-and-topics.md),
  [Transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md),
  [Timestamps](https://github.com/xblick/xblick/blob/main/docs/concepts/timestamps.md),
  [Sharing a GPU](https://github.com/xblick/xblick/blob/main/docs/guides/sharing-a-gpu.md).

## Licence and credits

This repository has two licences:

- `src/xblick_sam3/segmenter.py` is under Meta's SAM License, see [LICENSES/SAM-License.txt](LICENSES/SAM-License.txt).
  It assembles and adapts SAM 3's detector from Meta's model code, which makes it a derivative work of SAM 3.
- Everything else is under the Apache License 2.0, see [LICENSE](LICENSE).

[NOTICE](NOTICE) names the copyright holder and the parts of others.

SAM 3 and the SAM 3.1 checkpoint are Meta's work under the SAM License. This repository contains no copy of them.
SAM 3's code is a dependency at the commit that `uv.lock` fixes, and the setup script downloads the checkpoint.
Whoever runs the node uses both and is bound by the SAM License. Among its terms:

- SAM 3 may not be used for military or warfare purposes, nuclear industries or applications, espionage, the
  development or use of guns or illegal weapons, or activities subject to ITAR (clause 1.b.v).
- A publication of research done with SAM 3 must acknowledge its use (clause 1.b.ii).
- Whoever passes SAM 3 or a derivative work of it on, `segmenter.py` included, does so under the SAM License and
  with a copy of it (clause 1.b.i).

`tests/data/office.png` is a frame of the sequence `freiburg3_long_office_household` of the TUM RGB-D dataset of the
Technical University of Munich. It is under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0).

## Citation

This node uses SAM 3.1 (facebookresearch/sam3). If you use it in research, cite their work. The entries below are copied
from [its README](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/README.md):

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```
