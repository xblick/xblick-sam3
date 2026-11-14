# Options, prompts and boxes

This page lists every option of `xblick-sam3`. It describes the format in which you set prompts while the node
works, the `--boxes` mode, and the Python class behind the node.

## Options of `xblick-sam3`

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera` | required | The base name of the camera, for example `test/camera`. |
| `--words` | none | The text prompts at the start. Separate them with spaces or commas, for example `cup "soup can"`. Live: it changes while the node runs. |
| `--boxes TOPIC` | off | Segment the boxes of this `Detection2DArray` topic, one mask per box. |
| `--history` | 30 | The number of colour frames that the node keeps. Box messages and exemplar boxes refer to them by stamp. |
| `--color` | `<camera>/color` | The colour stream. |
| `--node` | `sam3` | The part of the topic names after the camera. |
| `--threshold` | 0.5 | The lowest score of an instance. Live: it changes while the node runs. |
| `--match-iou` | 0.3 | The lowest mask overlap with which an instance keeps the id of an object. Live: it changes while the node runs. |
| `--max-missing` | 15 | The number of frames for which an unseen object keeps its id. Live: it changes while the node runs. |
| `--compile MODE` | off | Compile the networks with `torch.compile` in this mode before the first frame. |
| `--resolution` | 672 | The size to which the node scales a frame. It must be a multiple of 336. |
| `--checkpoint` | `checkpoints/sam3.1_multiplex.pt` | The SAM 3.1 checkpoint. |
| `--device` | `cuda` | The GPU of the node. |
| `--no-download` | off | End at once when the checkpoint is missing, and name it. The node downloads nothing. |
| `--download-only` | off | Download the missing checkpoint, check its SHA-256 and end. The node needs no camera, Zenoh router or GPU for it, and `--camera` may be left out. |
| `--max-hz` | 30 | The highest number of colour frames per second. `0` or `none` takes every frame. |
| `--local` | `auto` | The transport for a camera on the same host: `auto`, `shm` or `cuda`. |
| `--connect`, `--listen`, `--config` | the router of the PC | The Zenoh session, as for every Durchblick program. |

**`--max-hz`.** The node takes at most this many colour frames per second, and always the newest one. It does not
decode the frames that it leaves out.
[Writing nodes](https://github.com/xblick/xblick/blob/main/docs/guides/writing-nodes.md)
describes the mechanism.

**`--local`.** With `auto`, the node reads a camera on the same host from its CUDA ring if that ring is on the
node's GPU. Otherwise it reads from the host ring in shared memory. `shm` and `cuda` force one of the two. With
`cuda`, the frames never leave the GPU, and the node stops if the camera offers no CUDA ring.

**MPS.** NVIDIA MPS lets several processes use one GPU at the same time. If you set `CUDA_MPS_PIPE_DIRECTORY`, the
node expects an MPS daemon and stops at start when none answers.
[Sharing a GPU](https://github.com/xblick/xblick/blob/main/docs/guides/sharing-a-gpu.md)
explains MPS.

## Prompts

A prompt tells SAM 3.1 what to segment. It is a word or phrase, a set of example boxes, or both. SAM 3 calls an
example box an *exemplar*. It means "every instance that looks like the object in this box".

### The queryable

The node answers Zenoh queries on `<camera>/sam3/prompts`.

- A query without a payload returns the current prompts as JSON.
- A query with a JSON payload replaces all prompts. The node applies the change between two frames and answers with
  the new prompts.
- If the node refuses the request, it answers with an error and keeps the old prompts.

```json
{"prompts": ["cup",
             {"text": "mug", "boxes": [[x1, y1, x2, y2]]},
             {"label": "my tool", "boxes": [[x1, y1, x2, y2]], "negative": [[x1, y1, x2, y2]]}],
 "stamp": 1730000000123456789}
```

| Field | Meaning |
| --- | --- |
| a string | A word or phrase. |
| `text` | The word or phrase of a prompt that also has boxes. |
| `boxes` | Exemplars as `[x1, y1, x2, y2]` in pixels. The node looks for instances like the boxed ones. |
| `negative` | Boxes around things that you do not mean. SAM 3 calls them negative boxes. |
| `label` | The name of the instances in `class_id`. The default is the text. |
| `stamp` | The stamp in nanoseconds of the colour frame in which you drew the boxes. |

Rules:

- A prompt needs a text or at least one box in `boxes`.
- A prompt without a text needs a `label`. SAM 3 then uses the text "visual" for it.
- The node must still have the frame of `stamp`. It keeps its last `--history` frames. Without `stamp`, the boxes
  refer to the newest frame.
- A `stamp` is valid only in a request that contains boxes.
- The answer contains the stamp of the frame on which the node encoded the boxes.

### The client

`xblick-sam3-prompts` sends these queries.

```bash
.venv/bin/xblick-sam3-prompts --camera test/camera                          # show the prompts
.venv/bin/xblick-sam3-prompts --camera test/camera --words bowl "soup can"  # set words
.venv/bin/xblick-sam3-prompts --camera test/camera --json '{"prompts": ["cup"]}'
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera` | required | The base name of the camera. |
| `--node` | `sam3` | The node's part of the topic name. |
| `--words` | | Set these words as the prompts. |
| `--json` | | Send this request. You cannot combine it with `--words`. |
| `--timeout` | 10 | The seconds to wait for the answer. |

Any Zenoh client can send the same query with `session.get(key, payload=...)`.

### Exemplars belong to one frame

The node encodes an exemplar once, on the frame in which you drew it. Every later frame reuses that encoding, as it
reuses the encoding of a text. [docs/design.md](design.md#exemplars-are-encoded-on-their-own-frame) explains the
reason and the measurement.

### Changing prompts with `--compile`

A change of the prompts compiles nothing. With 8 words at 672, a frame costs 25.4 ms either way.
[docs/performance.md](performance.md#compiling-for-any-number-of-prompts) has the details.

## Boxes

`--boxes TOPIC` gives each box of a `Detection2DArray` message its own mask. An example is Boxer's topic
`<camera>/boxer/detections_2d`. The node segments the object in the box, in the way SAM segments a box prompt. It
does not look for other instances of that object.

**The node follows the pace of the box messages.** A box message arrives after its colour frame. Boxer's boxes
arrive about 30 ms later. The node takes the newest box message and looks up the colour frame with the same stamp.
It then segments that frame once, with the prompts and the boxes together.

**The node drops a box message without its frame.** This happens when the node missed the frame, when the frame is
older than the last `--history` frames, or when the frame does not arrive within 0.5 s. The status counts these
messages as "box messages without their frame".

**The node omits some boxes.** The box head computes an object score for each box. The node leaves out a box in
which the head sees no object, and a box whose mask is empty.

An object from a box keeps its id from frame to frame. [docs/topics.md](topics.md#ids) explains the ids.

[docs/design.md](design.md#the-box-head) describes the network behind `--boxes`, and
[docs/performance.md](performance.md#the-cost-of---boxes) states its cost.

## In Python, without Durchblick

The class `Segmenter` works on tensors and needs no Durchblick session.

```python
from xblick_sam3.segmenter import Prompt, Segmenter

segmenter = Segmenter(["cup", "bottle"], compile="default", resolution=672, box_head=True)
result = segmenter.segment(rgb)  # rgb: (H, W, 3) uint8 RGB torch tensor, in host or GPU memory
result.label, result.score, result.boxes, result.masks, result.instances

# An exemplar: pass the frame in which you drew the box. Later calls work on any frame.
segmenter.set_prompts(["cup", Prompt(label="my mug", boxes=((120, 80, 180, 160),))], rgb)

# Given boxes, shape (M, 4), as x1 y1 x2 y2 in pixels.
result = segmenter.segment(rgb, boxes)
result.box_masks, result.box_scores, result.box_found
```

The results are tensors on the GPU. `result.label` holds the index of the prompt of each instance.

The class uses SAM 3's resolution of 1008 unless you pass `resolution=672`. The node passes 672 by default.

## Parameters

Every option is also a parameter of the node `<camera>/<node>`, for example `test/camera/sam3`. The live ones
change while the node runs: `threshold`, `match_iou`, `max_missing`, `words`; the tables mark them. The others are
read-only and apply at the next start, where `--param NAME:=VALUE` or `--params-file FILE` set them. New words replace
the prompts at the next frame, and prompts set over `<camera>/<node>/prompts` show in `words`. Durchblick's terminal
interface (`xblick`, tab Parameters) and `xblick param` show them.
