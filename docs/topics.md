# Topics and messages

This page lists what xblick-sam3 reads and what it publishes, field by field. It also explains how an object
keeps its id from frame to frame.

## Inputs

| Topic | Type | Use |
| --- | --- | --- |
| `<camera>/color` | image stream with `camera_info` | The frames to segment. The encoding is `rgb8`, `rgba8`, `bgr8`, `bgra8` or `mono8`. |
| the topic of `--boxes` | `vision_msgs/Detection2DArray` | Boxes to segment, in pixels of the colour image. Each message has the stamp of its colour frame. |

`--color` names another colour stream. The node needs the `camera_info` of the colour stream. It counts the frames
that arrive without one and reports the count in its status.

## Outputs

Every result has the stamp of its colour frame and the frame id of the camera.
[Timestamps](https://github.com/xblick/xblick/blob/main/docs/concepts/timestamps.md) explains
why a result keeps the stamp of its input. `--node` changes the `sam3` part of the topic names.

A frame without instances publishes an empty array and an image of zeros. A viewer then clears its view.

### `<camera>/sam3/detections_2d`

The message is a `vision_msgs/Detection2DArray` with one detection per instance.

| Field | Content |
| --- | --- |
| `class_id` | The label of the prompt. For a word, the label is the word. |
| `score` | The score that SAM 3.1 gives the instance. |
| box | The box of the instance in pixels of the colour image. |
| `id` | The tracking id of the object, from "1" to "65535". The instance image holds the same value. |

The detections of the given boxes come first, in the order of the box message. Each of them keeps its own
hypotheses and its own box. The detections of the prompts follow, sorted by score with the best first.

### `<camera>/sam3/instances`

The node publishes all masks as one image stream with the encoding `mono16`. The image has the size of the colour
frame and the `camera_info` of the colour stream. A pixel holds 0 where no instance is. Elsewhere it holds the id of
the object. Where two masks overlap, the pixel holds the id of the instance with the higher score.

A reader on the same host gets the image on the GPU through the `cuda` transport. Other readers get it `raw` or
`compressed` as PNG.
[Transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md) explains
the choice.

`xblick rerun --segmentation` draws the instance image over the colour image, with one colour per instance.
[docs/design.md](design.md) states why the encoding is `mono16`.

### `<camera>/diagnostics`

The node publishes the status `sam3` once a second as a `diagnostic_msgs/DiagnosticArray`. The status contains:

- the frames per second,
- the instances and the given boxes per frame,
- the milliseconds per frame,
- the number of frames without `camera_info`,
- the number of box messages without their frame,
- the prompts.

The start log and every report in the log also state which transport each stream uses. The report states the frame
rate in effect and the number of frames that `--max-hz` left out.

### `<camera>/sam3/prompts`

This is a Zenoh queryable that answers in JSON. A query without a payload returns the prompts. A query with a
payload sets them. [docs/prompts.md](prompts.md) describes the format. The node answers at once, also while the
model works on a frame. It checks new prompts and answers with them, and the node segments them from the next frame
on. The queryable exists once the model is loaded.

## Ids

An object keeps its id from frame to frame. The node matches each frame's instances to the known objects of the
same prompt. It compares the mask of an instance with the last mask of each object. Two masks match when their
intersection over union is at least `--match-iou` (default 0.3). The node chooses the assignment with the largest
total overlap. An instance without a match gets a new id.

An object ends when the detector has not seen it for more than `--max-missing` frames (default 15). The matching
takes 0.4 ms per frame. The code is in `src/xblick_sam3/tracking.py`.

An object from a given box keeps its id in the same way. The node tracks these objects apart from the objects of
the prompts. An object of the box class "cup" and an object of the word "cup" are two objects, even if both are the
same cup.

### How stable the ids are

We replayed 901 frames of a simulation in which the head and the hands move. The words were box, can and bottle.

- An object that the detector finds in every frame keeps its id.
- An object that the detector loses and finds again gets a new id in two cases. Either it returns more than 15
  frames later, or it returns at another place in the image.
- This happens to objects whose score is close to `--threshold`. The word "can" mostly matched another object, with
  scores of 0.50 to 0.58 in some frames.

| Word | Ids in 901 frames |
| --- | --- |
| can | 46 |
| box | 8 |
| bottle | 5 |

A higher `--threshold` removes such detections.

## A camera that changes

When the colour camera returns with another image size or another frame id, the node starts its tracks anew. All
objects then get new ids.
