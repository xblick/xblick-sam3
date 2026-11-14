# xblick-sam3

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
2. Clone this repository and run the setup script. It creates `.venv` exactly as `uv.lock` fixes it.

   ```bash
   tools/setup.sh
   ```

   The first call stops at the download, because you are not logged in yet.
3. Log in once, or set `HF_TOKEN` to your token.

   ```bash
   .venv/bin/hf auth login
   ```

4. Run `tools/setup.sh` again. It downloads the checkpoint (3.5 GB) into `checkpoints/` and checks its SHA-256.

The venv contains PyTorch with CUDA 13 and SAM 3's code from GitHub at the commit that `uv.lock` fixes. If `HF_TOKEN`
is set before step 2, the first call already downloads the checkpoint.

## Run

Start the node on a camera. `robot/camera` is an example name. The node connects to the Zenoh router of the
PC, which [Sessions](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md)
explains.

```bash
.venv/bin/xblick-sam3 --camera robot/camera --words "cup,bottle,tomato soup can" --compile default
```

`--compile default` compiles the networks for about 45 s before the first frame. After that a frame takes about a
third less time.

Look at the result in Durchblick's viewer:

```bash
xblick rerun --spawn --detections robot/camera/sam3/detections_2d \
    --segmentation robot/camera/sam3/instances robot/camera/color
```

Read the prompts and set new ones while the node works:

```bash
.venv/bin/xblick-sam3-prompts --camera robot/camera
.venv/bin/xblick-sam3-prompts --camera robot/camera --words bowl "soup can"
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

All options, the prompt format and the Python class: [docs/prompts.md](docs/prompts.md).

Every option is also a parameter of the node `<camera>/<node>`, for example `robot/camera/sam3`. Durchblick's
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

- Version 0.2.1. We tested the node on recorded frames and on the replayed head camera of a simulation, beside the
  other Durchblick nodes. We have not tested it on a robot.
- We tested prompts over Zenoh and `--boxes` on recorded frames and in the tests. We have not tested `--boxes` live
  on Boxer's boxes.
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
