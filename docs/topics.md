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
- the new ids in the last second, the known objects per frame (seen and unseen) and the frames with an estimate
  of the image's motion,
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
same prompt in four steps. The code is in `src/xblick_sam3/tracking.py`.

1. The motion of the image. When the camera pans, tilts, drives or zooms, all masks move at once, and their overlap
   with the last masks drops. With `--motion masks` (the default) the node estimates one shift and one scale of the
   whole image from the masks. Each pair of a known object and an instance of the same prompt and a similar size
   proposes a shift and a scale. The proposal that most objects agree with wins. It needs two objects that agree;
   with fewer, the node takes the image as still. The masks of all known objects then move with the image, also
   those of objects that are unseen at the moment. In-plane rotation is not estimated. `--motion none` skips this
   step.
2. The matching. An instance keeps the id of an object when the intersection over union of their masks is at least
   `--match-iou` (default 0.3). It also keeps it when its centre is at most `--match-distance` object sizes from
   where the object is expected (default 0.5; the size is the square root of the area) and their areas differ by at
   most the factor `--match-size` (default 2). The node chooses the assignment with the largest total score: overlap
   first, then closeness, then objects seen more recently.
3. Merge and split. An instance that covers at least half of the masks of two objects keeps the id of the larger
   object. The smaller object waits as an unseen object and gets its id back when the masks part again. When the
   mask of an object splits into two instances, the larger part keeps the id.
4. An instance without a match gets a new id. An object ends when the detector has not seen it for more than
   `--max-missing` frames (default 30). Until then it takes an instance at the place where it is expected.

The node works on copies of the masks that are at most 160 pixels wide. The ids take about 1.3 ms per frame
(0.6 ms with the overlap alone), with about six instances per frame on an RTX PRO 6000.

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

On TUM RGB-D's `fr3/long_office_household` (2585 frames of a hand-held camera, words keyboard, cup, book, monitor
and teddy bear) the node issued 168 ids with the rules above and 384 with the overlap alone (0.2.6). On every second
frame, as with a slower detector, it issued 142 instead of 502. The rest are mostly new instances that SAM 3 finds
for a few frames only, and parts of a mask that SAM 3 reports as an object of their own (one book of a pile).

## A camera that changes

When the colour camera returns with another image size or another frame id, the node starts its tracks anew. All
objects then get new ids.
