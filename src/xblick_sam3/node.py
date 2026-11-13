"""xblick-sam3: SAM 3.1's open-vocabulary instance segmentation (segmenter.py) on a camera's colour stream.

    xblick-sam3 --camera robot/camera [--words cup bottle "tomato soup can"] \\
                    [--boxes <camera>/boxer/detections_2d] [--history 30] \\
                    [--color <camera>/color] [--node sam3] [--threshold 0.5] [--compile max-autotune-no-cudagraphs]
                    [--resolution 672] [--local shm|cuda]
                    [--connect ...] [--listen ...] [--config FILE]   (default: the PC's Zenoh router)

What it segments:

* prompts: words (--words), example boxes (exemplars: every instance like the boxed one) or both, set at the start
  (--words) and at any time over the queryable ``<camera>/<node>/prompts`` (prompts.py, ``xblick-sam3-prompts``),
  which also answers with the prompts now
* with --boxes, the boxes of a vision_msgs/Detection2DArray topic, e.g. Boxer's: one mask per box, that box's object
  (SAM's interactive head); each box message's frame (the colour frame of its stamp, one of the last --history) is
  segmented, prompts too: the node then runs at the pace of the boxes

and publishes, stamped as the frame and in its camera frame:

* ``<camera>/<node>/detections_2d``  vision_msgs/Detection2DArray: one detection per instance, the given boxes' first
  (in their order, each with its own hypotheses and box; boxes the head finds no object in left out), then the
  prompts' (best first, the prompt's label as class_id with its score, its box in pixels of the colour image); its id
  the instance's tracking id ("1" .. "65535"): the same object keeps its id from frame to frame (tracking.py: its mask
  overlaps its last one of the same label and kind by at least --match-iou; unseen for more than --max-missing frames,
  its id ends), and it is the instance's value in ``instances``
* ``<camera>/<node>/instances``      image stream, mono16: the instance masks as one image of the colour frame's size,
  0 where no instance is, else the instance's id (the first instance of the detections where masks overlap); the
  colour stream's camera_info; mono16, not 16UC1: REP 118 makes 16UC1 depth in millimetres (Durchblick's Rerun bridge
  shows it so). On the GPU (``cuda`` transport, Durchblick's CUDA ring) for readers on the same host, else ``raw`` or
  ``compressed`` (PNG)

A frame without instances publishes an empty array and an all-zero image (a reader's view clears). Without --boxes it
processes as fast as SAM 3.1 runs, always the newest frame; frames per second, instances per frame and the time from
frame to results go to ``<camera>/diagnostics`` (status ``sam3``) once a second.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import xblick as db
from xblick import mps, node, tensors, topics
from xblick.camera import check_local
from xblick.history import FrameHistory
from xblick.session import add_arguments, open_from_args

from .prompts import PROMPTS, describe, parse

FRAME_WAIT_S = 0.5  # how long a box message waits for its frame (newer than the newest received) before it is dropped


def to_message(result, ids: list[int], words: list[str], header: db.Header) -> db.Detection2DArray:
    boxes, scores, labels = result.boxes.tolist(), result.score.tolist(), result.label.tolist()
    return db.Detection2DArray(header, [
        db.Detection2D(header, [db.ObjectHypothesisWithPose(db.ObjectHypothesis(words[label], score))],
                       db.BoundingBox2D(db.Pose2D(db.Point2D((x1 + x2) / 2, (y1 + y2) / 2), 0.0), x2 - x1, y2 - y1),
                       str(i))
        for i, (x1, y1, x2, y2), score, label in zip(ids, boxes, scores, labels, strict=True)])


def boxes_of(message: db.Detection2DArray) -> list[tuple[float, float, float, float]]:
    """A Detection2DArray's boxes as x1, y1, x2, y2 (pixels; theta ignored: SAM's boxes are upright)."""
    return [(d.bbox.center.position.x - d.bbox.size_x / 2, d.bbox.center.position.y - d.bbox.size_y / 2,
             d.bbox.center.position.x + d.bbox.size_x / 2, d.bbox.center.position.y + d.bbox.size_y / 2)
            for d in message.detections]


class LatestBoxes:
    """The newest Detection2DArray of a topic (``take``: once each), from Zenoh's thread; sets ``wake``."""

    def __init__(self, session, key: str, wake: threading.Event):
        self._lock = threading.Lock()
        self._message = None
        self._wake = wake
        self.received = 0
        self._subscriber = db.DetectionSubscriber(session, key, self._put)

    def _put(self, message):
        if not isinstance(message, db.Detection2DArray):
            raise TypeError(f"{type(message).__name__}: Detection2DArray (boxes in an image) expected")
        with self._lock:
            self._message = message
            self.received += 1
        self._wake.set()

    def take(self):
        self._subscriber.check()
        with self._lock:
            message, self._message = self._message, None
        return message

    def close(self):
        self._subscriber.close()


# the parameters that change while the node runs (Parameters.declare_options); every other option applies at the start
LIVE = {"threshold": {"range": (0.0, 1.0, 0.01)},
        "match_iou": {"range": (0.01, 1.0, 0.01)},
        "max_missing": {"range": (0, 300, 1), "unit": "frames"},
        "words": {}}


def split_words(given: list[str]) -> list[str]:
    """Prompt words as the option takes them: each also comma-separated (``cup,bottle`` or ``cup "soup can"``)."""
    return [s.strip() for word in given for s in word.split(",") if s.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xblick-sam3", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="the camera's base, e.g. robot/camera")
    parser.add_argument("--words", nargs="*", default=[],
                        help="text prompts at the start: words, each also comma-separated, e.g. 'cup,bottle' or cup "
                             "\"soup can\" (while it runs: the parameter words, or <camera>/<node>/prompts with "
                             "exemplar boxes)")
    parser.add_argument("--boxes", default=None, metavar="TOPIC",
                        help="segment the boxes of this Detection2DArray topic, one mask each, e.g. "
                             "<camera>/boxer/detections_2d")
    parser.add_argument("--history", type=int, default=30,
                        help="colour frames kept for box messages and exemplar boxes to refer to (by stamp)")
    parser.add_argument("--color", default=None, help="the colour stream (default <camera>/color)")
    parser.add_argument("--node", default="sam3", help="the results' part of the topic: <camera>/<node>/...")
    parser.add_argument("--threshold", type=float, default=0.5, help="the least score of an instance")
    parser.add_argument("--match-iou", type=float, default=0.3,
                        help="the least overlap (intersection over union) of an instance's mask with an object's last "
                             "one for it to keep that object's id")
    parser.add_argument("--max-missing", type=int, default=15,
                        help="frames an object may go unseen and keep its id")
    parser.add_argument("--compile", default=None, metavar="MODE",
                        help="torch.compile the networks in this mode (e.g. max-autotune-no-cudagraphs; README, "
                             "\"Speed\"); compiles before the first frame, minutes")
    parser.add_argument("--resolution", type=int, default=672,
                        help="the size frames are scaled to: 672 (half the time a frame), or SAM 3's own 1008 (README, "
                             "\"Speed\")")
    parser.add_argument("--checkpoint", default=None,
                        help="SAM 3.1's checkpoint (default: checkpoints/sam3.1_multiplex.pt, from tools/setup.sh)")
    parser.add_argument("--device", default="cuda")
    add_arguments(parser)  # --connect, --listen, --config; default: the PC's Zenoh router
    # --max-hz: at most 30 frames/s by default (0 or none: every frame)
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, max_hz=30.0, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = parser.parse_args(argv)
    mps.check()  # with CUDA_MPS_PIPE_DIRECTORY (MPS meant) its daemon has to run: else stop here
    camera = topics.check_name(args.camera)
    # the options as parameters (the terminal interface shows them): read-only, but those LIVE, which the loop reads
    session = open_from_args(args)
    params = node.Parameters(session, args.node, camera)
    params.configure(args)
    params.declare_options(parser, args, live=LIVE, leave_out=("camera", "node"))
    params.check_overrides()
    if args.history < 1:
        parser.error("--history: at least 1")
    words = split_words(args.words)
    color_base = args.color or f"{camera}/color"

    # sam3's own imports: pkg_resources' deprecation (setuptools)
    node.start("xblick-sam3", ignore=["pkg_resources is deprecated"], max_hz=args.max_hz, local=args.local)

    import torch  # noqa: PLC0415 - with the model, after the arguments are checked

    from .segmenter import Segmenter  # noqa: PLC0415 - loads sam3
    from .tracking import Tracker  # noqa: PLC0415

    device = torch.device(args.device)
    cuda_index = device.index if device.index is not None else 0
    segmenter = Segmenter(words, threshold=args.threshold, device=device, compile=args.compile,
                          resolution=args.resolution, box_head=args.boxes is not None,
                          **({"checkpoint": args.checkpoint} if args.checkpoint else {}))
    prompts_stamp = None  # the frame the prompts' boxes are in
    tracker = Tracker(min_iou=args.match_iou, max_missing=args.max_missing)
    wake = threading.Event()
    color = db.ImageSubscriber(session, color_base, transport="auto", camera_info=True, local=args.local,
                               max_hz=args.max_hz, device=device)
    history = FrameHistory(color, args.history, wake.set)
    box_topic = topics.check_name(args.boxes) if args.boxes else None
    boxes_in = LatestBoxes(session, box_topic, wake) if box_topic else None
    prompts_key = f"{camera}/{args.node}/{PROMPTS}"
    queryable = session.declare_queryable(prompts_key)  # answered between frames, in this thread
    detections_pub = db.DetectionPublisher(session, topics.detections(camera, args.node, "detections_2d"),
                                           db.Detection2DArray)
    instances_pub = None  # with the first frame's camera_info: the colour camera's
    instances_key = None  # (width, height, frame) of the frames the tracks are of
    report = node.Reporter(session, camera, args.node, "xblick-sam3", max_hz=args.max_hz)
    report.watch(color)
    print(f"xblick-sam3: {color_base}" + (f" + {box_topic}" if box_topic else "")
          + f" -> {camera}/{args.node}/{{detections_2d,instances}}, prompts {segmenter.words} ({prompts_key})",
          flush=True)
    counts = report.counts

    def answer(query):
        """Sets the prompts of a query's payload (if any) and answers with the prompts; errors leave them as they
        were."""
        nonlocal prompts_stamp
        try:
            payload = query.payload.to_string() if query.payload is not None else ""
            if payload.strip():
                prompts, stamp = parse(json.loads(payload))
                frame = None
                if any(p.exemplars for p in prompts):
                    frame = (history.newest() or (None, None))[1] if stamp is None else history.at(stamp)
                    if frame is None:
                        raise ValueError(f"no colour frame {'yet' if stamp is None else f'of stamp {stamp}'} (the "
                                         f"node keeps its last {args.history})")
                segmenter.set_prompts(prompts, tensors.rgb(frame, device) if frame is not None else None)
                prompts_stamp = frame.header.stamp.to_ns() if frame is not None else None
                applied_words[0] = segmenter.words  # the parameter follows: it shows the prompts' names
                params.set("words", segmenter.words)
                print(f"xblick-sam3: prompts {segmenter.words}", flush=True)
            query.reply(prompts_key, json.dumps(describe(segmenter.prompts, prompts_stamp)))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            query.reply_err(f"{type(error).__name__}: {error}")

    def process(frame, box_message):
        nonlocal instances_pub, instances_key, tracker
        info = color.camera_info
        if info is None or (info.height, info.width) != tuple(frame.image.shape[:2]):
            counts["no_info"] += 1  # none yet, or the camera's new model came before its frames (or after)
            return
        frame_id = info.header.frame_id or frame.header.frame_id
        key = (info.width, info.height, frame_id)
        if instances_pub is not None and key != instances_key:  # the camera came back with another size or frame
            print(f"xblick-sam3: {color_base} is now {info.width} x {info.height} in {frame_id}: the tracks start "
                  "anew (new ids)", flush=True)
            tracker = Tracker(min_iou=args.match_iou, max_missing=args.max_missing)
            instances_pub.camera_info, instances_pub.frame_id = info, frame_id
        instances_key = key
        if instances_pub is None:
            if args.compile is not None:
                print(f"xblick-sam3: compiling ({args.compile})", flush=True)
                segmenter.warm_up(*frame.image.shape[:2])
            instances_pub = db.CameraPublisher(
                session, f"{camera}/{args.node}/instances", encoding="mono16", frame_id=frame_id,
                camera_info=info, source="sam3", cuda_device=cuda_index, labels=True, codec="png")
        started = time.monotonic()
        given = boxes_of(box_message) if box_message is not None else None
        boxes = torch.tensor(given, dtype=torch.float32).reshape(-1, 4) if given is not None else None
        result = segmenter.segment(tensors.rgb(frame, device), boxes)
        header = db.Header(frame.header.stamp, frame_id)
        found = result.box_found.tolist() if given is not None else []
        kept = [d for d, f in zip(box_message.detections, found, strict=True) if f] if given is not None else []
        box_labels = [("box", d.results[0].hypothesis.class_id if d.results else "") for d in kept]
        masks = torch.cat([result.box_masks[result.box_found], result.masks]) if given is not None else result.masks
        names = segmenter.words
        ids = tracker.update(box_labels + [("prompt", names[i]) for i in result.label.tolist()], masks)
        # ordered on this thread's stream after the masks: publish copies it once it is complete
        instances = Tracker.instances_image(ids, masks)
        instances_pub.publish(instances, stamp_ns=frame.header.stamp.to_ns(), stream=torch.cuda.current_stream(device))
        message = to_message(result, ids[len(kept):], names, header)
        message.detections[:0] = [db.Detection2D(header, d.results, d.bbox, str(i))
                                  for d, i in zip(kept, ids[:len(kept)], strict=True)]
        detections_pub.publish(message)
        counts["frames"] += 1
        counts["instances"] += len(ids)
        counts["boxes"] += len(found)
        counts["busy"] += time.monotonic() - started

    def apply_parameters():
        """The live parameters' values now, in this thread (between frames): threshold and tracking at once, new
        words as the prompts (exemplar prompts set on the prompts key give way to them)."""
        segmenter.threshold = params["threshold"]
        tracker.min_iou, tracker.max_missing = params["match_iou"], params["max_missing"]
        wanted = split_words(params["words"])
        if wanted != applied_words[0]:
            applied_words[0] = wanted
            segmenter.set_prompts(wanted)
            print(f"xblick-sam3: prompts {segmenter.words} (parameter words)", flush=True)

    applied_words = [words]
    pending, pending_since, last_stamp = None, 0.0, None
    try:
        while True:
            wake.wait(0.05)
            wake.clear()
            history.check()
            while (query := queryable.try_recv()) is not None:
                answer(query)
            apply_parameters()
            frame = (history.newest() or (None, None))[1]
            if frame is not None:
                check_local(color, args.local)
            if boxes_in is not None:
                taken = boxes_in.take()
                if taken is not None:
                    if pending is not None:
                        counts["no_frame"] += 1  # a newer one came before its frame
                    pending, pending_since = taken, time.monotonic()
                frame = None
                if pending is not None:
                    stamp = pending.header.stamp.to_ns()
                    frame = history.at(stamp)
                    if frame is None and (stamp < history.newest_stamp
                                          or time.monotonic() - pending_since > FRAME_WAIT_S):
                        counts["no_frame"] += 1  # its frame was missed or is no longer kept
                        pending = None
            elif frame is not None and frame.header.stamp.to_ns() == last_stamp:
                frame = None
            if frame is not None:
                taken_boxes, pending = pending, None
                last_stamp = frame.header.stamp.to_ns()
                process(frame, taken_boxes)
            if report.due():
                d, seconds = report.take()
                n, ms = max(d["frames"], 1), d["busy"] / max(d["frames"], 1) * 1e3
                problems = ([f"{d['no_info']} without camera_info"] if d["no_info"] else []) + \
                    ([f"{d['no_frame']} box messages without their frame"] if d["no_frame"] else [])
                waiting = None if d["frames"] else f"{color_base}: {color.describe()}" + (
                    f"; {box_topic}: {boxes_in.received} messages" if boxes_in is not None else "")
                level, message = node.state(f"{d['frames'] / seconds:.2f} frames/s", problems, waiting)
                report.publish(level, message,
                               {"frames/s": d["frames"] / seconds, "instances/frame": d["instances"] / n,
                                "boxes/frame": d["boxes"] / n, "ms/frame": ms, "no camera_info": d["no_info"],
                                "boxes without their frame": d["no_frame"], "prompts": ", ".join(segmenter.words)},
                               log_extra=f", {ms:.0f} ms a frame")
    except KeyboardInterrupt:
        pass
    finally:
        history.close()  # and its subscriber
        for closable in (boxes_in, detections_pub, instances_pub, report):
            if closable is not None:
                closable.close()
        queryable.undeclare()
        params.close()
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
