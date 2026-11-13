"""The node: its conversions (no GPU), and the node itself as its own process on a published colour stream over real
Zenoh sessions (needs a CUDA device and access to facebook/sam3.1)."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import xblick as db
from xblick import topics
from xblick.session import open_session

from xblick_sam3.node import boxes_of, to_message
from xblick_sam3.prompts import describe, parse
from xblick_sam3.segmenter import Prompt, Segmentation

CAM = "robot/camera"
WORDS = ["chair", "cup", "book"]


def test_detections_message_carries_words_scores_boxes_and_tracking_ids():
    result = Segmentation(label=torch.tensor([2, 0]), score=torch.tensor([0.9, 0.6]),
                          boxes=torch.tensor([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 4.0, 2.0]]),
                          masks=torch.zeros(2, 64, 64, dtype=torch.bool),
                          instances=torch.zeros(64, 64, dtype=torch.uint16))
    header = db.Header(db.Time.from_ns(5_000_000_123), "cam")
    message = to_message(result, [12, 4], WORDS, header)
    assert message.header == header and len(message.detections) == 2
    first, second = message.detections
    assert first.id == "12" and second.id == "4"
    assert first.results[0].hypothesis.class_id == "book"
    assert first.results[0].hypothesis.score == pytest.approx(0.9)
    assert (first.bbox.center.position.x, first.bbox.center.position.y) == (20.0, 40.0)
    assert (first.bbox.size_x, first.bbox.size_y) == (20.0, 40.0)
    assert second.results[0].hypothesis.class_id == "chair"


def test_prompt_requests_parse_and_describe_back():
    request = {"prompts": ["cup", {"text": "mug", "boxes": [[1, 2, 30, 40]]},
                          {"label": "my tool", "boxes": [[5, 5, 9, 9]], "negative": [[0, 0, 4, 4.5]]}], "stamp": 12}
    prompts, stamp = parse(request)
    assert stamp == 12 and [p.name for p in prompts] == ["cup", "mug", "my tool"]
    assert prompts[1] == Prompt(text="mug", boxes=((1.0, 2.0, 30.0, 40.0),))
    assert prompts[2].negative == ((0.0, 0.0, 4.0, 4.5),) and prompts[2].text is None
    assert parse(json.loads(json.dumps(describe(prompts, stamp)))) == (prompts, stamp)
    assert parse({"prompts": []}) == ([], None)


@pytest.mark.parametrize("request_, message", [
    ({"words": ["cup"]}, "a request is"),
    ({"prompts": "cup"}, "a list"),
    ({"prompts": [""]}, "empty"),
    ({"prompts": [{"boxes": [[0, 0, 1, 1]]}]}, "needs a label"),
    ({"prompts": [{"text": "cup", "boxes": [[5, 0, 1, 1]]}]}, "x1 < x2"),
    ({"prompts": [{"text": "cup", "boxes": [[0, 0, 1]]}]}, "x1, y1, x2, y2"),
    ({"prompts": [{"label": "x", "negative": [[0, 0, 1, 1]]}]}, "at least one"),
    ({"prompts": ["cup"], "stamp": 5}, "stamp without boxes"),
    ({"prompts": [{"text": "cup", "colour": "red"}]}, "an object of"),
])
def test_bad_prompt_requests_say_why(request_, message):
    with pytest.raises(ValueError, match=message):
        parse(request_)


def test_box_messages_become_corner_boxes():
    header = db.Header()
    message = db.Detection2DArray(header, [db.Detection2D(header, [], db.BoundingBox2D(
        db.Pose2D(db.Point2D(20.0, 40.0), 0.0), 10.0, 30.0))])
    assert boxes_of(message) == [(15.0, 25.0, 25.0, 55.0)]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_node_segments_a_published_stream():
    """Publishes the office frame (bgr8, as the robot's cameras) at 10 Hz, each with its own stamp; the node, its own
    process, answers with detections and the instances image of the same stamps, consistent with each other, and the
    same ids in every frame (the scene does not move)."""
    import cv2  # noqa: PLC0415

    bgr = cv2.imread(str(Path(__file__).parent / "data" / "office.png"))
    h, w = bgr.shape[:2]
    port = free_port()
    session = open_session(listen=[f"127.0.0.1:{port}"])
    info = db.CameraInfo.pinhole(w, h, 525.0, 525.0, w / 2 - 0.5, h / 2 - 0.5, frame_id="azurekinect")
    publisher = db.CameraPublisher(session, f"{CAM}/color", encoding="bgr8", frame_id="azurekinect", camera_info=info,
                                   source="test")
    detections, lock = {}, threading.Lock()

    def on_detections(message):
        with lock:
            detections[message.header.stamp.to_ns()] = message

    det_sub = db.DetectionSubscriber(session, topics.detections(CAM, "sam3", "detections_2d"), on_detections)
    # a reader on the CPU: without device= local auto would take the node's CUDA ring on GPU 0 (Durchblick 0.3.38)
    inst_sub = db.ImageSubscriber(session, f"{CAM}/sam3/instances", transport="auto", device="cpu")
    node = shutil.which("xblick-sam3", path=os.path.dirname(sys.executable))
    process = subprocess.Popen([node, "--camera", CAM, "--words", ",".join(WORDS), "--connect", f"127.0.0.1:{port}"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    try:
        instances, stamp_ns, deadline = {}, 1_000_000_000, time.monotonic() + 120.0
        while len(instances) < 5 and time.monotonic() < deadline:
            assert process.poll() is None, "".join(output)
            publisher.publish(bgr, stamp_ns=stamp_ns)
            stamp_ns += 100_000_000
            frame = inst_sub.latest(timeout=0.1)
            if frame is not None:
                instances[frame.header.stamp.to_ns()] = frame
            det_sub.check()
        assert len(instances) >= 5, "".join(output)
        time.sleep(0.5)  # the detections of the last instances image
        published = set(range(1_000_000_000, stamp_ns, 100_000_000))
        id_sets = set()
        for stamp, frame in instances.items():
            assert stamp in published
            assert frame.encoding == "mono16" and frame.header.frame_id == "azurekinect"
            image = np.asarray(frame.image)
            assert image.shape == (h, w)
            with lock:
                message = detections[stamp]
            n = len(message.detections)
            assert n >= 5  # chairs, cups, books
            ids = [int(d.id) for d in message.detections]
            assert len(set(ids)) == n and all(1 <= i <= 65535 for i in ids)
            id_sets.add(frozenset(ids))
            assert {d.results[0].hypothesis.class_id for d in message.detections} == set(WORDS)
            scores = [d.results[0].hypothesis.score for d in message.detections]
            assert scores == sorted(scores, reverse=True)
            assert set(np.unique(image)) <= {0, *ids}
            for d in message.detections:
                ys, xs = np.nonzero(image == int(d.id))
                if len(xs) == 0:
                    continue  # covered by better instances
                c, sx, sy = d.bbox.center.position, d.bbox.size_x, d.bbox.size_y
                # the instance's pixels within its box, up to two cells of the mask head's grid (masks and boxes are
                # separate heads; at the node's 672 the grid is 192 x 192: 3.3 pixels of this frame a cell)
                slack = 2 * w / 192
                assert xs.min() >= c.x - sx / 2 - slack and xs.max() <= c.x + sx / 2 + slack
                assert ys.min() >= c.y - sy / 2 - slack and ys.max() <= c.y + sy / 2 + slack
        assert len(id_sets) == 1, id_sets  # the same objects, the same ids

        # its options are parameters: read-only, but the live ones; a new word list is the prompts from the next frame
        from xblick.parameters import ParameterClient  # noqa: PLC0415

        client = ParameterClient(session, f"{CAM}/sam3")
        descriptors, values = client.all()
        by_name = {d.name: (d, v.value()) for d, v in zip(descriptors, values, strict=True)}
        assert by_name["words"][1] == [",".join(WORDS)] and not by_name["words"][0].read_only
        assert by_name["threshold"][1] == 0.5 and not by_name["threshold"][0].read_only
        assert by_name["resolution"][0].read_only and by_name["resolution"][1] == 672
        assert "camera" not in by_name and "connect" not in by_name
        assert client.set({"words": [WORDS[0]]})[0].successful
        assert not client.set({"threshold": 1.5})[0].successful
        deadline = time.monotonic() + 30.0
        found = set()
        while time.monotonic() < deadline:
            publisher.publish(bgr, stamp_ns=stamp_ns)
            stamp_ns += 100_000_000
            time.sleep(0.2)
            det_sub.check()
            with lock:
                newest = detections[max(detections)]
            found = {d.results[0].hypothesis.class_id for d in newest.detections}
            if found == {WORDS[0]}:
                break
        assert found == {WORDS[0]}, found
        client.close()
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        inst_sub.close()
        det_sub.close()
        publisher.close()
        session.close()
    assert process.returncode == 0, "".join(output)


def box_iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return w * h / (area(a) + area(b) - w * h)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_node_segments_given_boxes_and_takes_prompts_over_its_queryable():
    """With --boxes the node segments each box message's frame (the colour frame of its stamp): one instance per box,
    first in the detections, with the box's own hypothesis and box; the boxes here are the node's own word detections
    of an earlier frame. Its queryable answers with the prompts, sets new ones (an exemplar box of a chair on a frame
    it still has: its instances are chairs) and refuses bad ones, keeping its prompts."""
    import cv2  # noqa: PLC0415

    bgr = cv2.imread(str(Path(__file__).parent / "data" / "office.png"))
    h, w = bgr.shape[:2]
    port = free_port()
    session = open_session(listen=[f"127.0.0.1:{port}"])
    info = db.CameraInfo.pinhole(w, h, 525.0, 525.0, w / 2 - 0.5, h / 2 - 0.5, frame_id="azurekinect")
    publisher = db.CameraPublisher(session, f"{CAM}/color", encoding="bgr8", frame_id="azurekinect", camera_info=info,
                                   source="test")
    box_key = topics.detections(CAM, "given", "detections_2d")
    boxes_pub = db.DetectionPublisher(session, box_key, db.Detection2DArray)
    detections, lock = {}, threading.Lock()

    def on_detections(message):
        with lock:
            detections[message.header.stamp.to_ns()] = message

    det_sub = db.DetectionSubscriber(session, topics.detections(CAM, "sam3", "detections_2d"), on_detections)
    # a reader on the CPU: without device= local auto would take the node's CUDA ring on GPU 0 (Durchblick 0.3.38)
    inst_sub = db.ImageSubscriber(session, f"{CAM}/sam3/instances", transport="auto", device="cpu")
    node = shutil.which("xblick-sam3", path=os.path.dirname(sys.executable))
    node_port = free_port()  # the prompts client's way to the node: this test's peer session routes no queries
    process = subprocess.Popen([node, "--camera", CAM, "--words", ",".join(WORDS), "--boxes", box_key,
                                "--connect", f"127.0.0.1:{port}", "--listen", f"127.0.0.1:{node_port}"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    key = f"{CAM}/sam3/prompts"

    def ask(payload=None):
        replies = list(session.get(key, payload=payload, timeout=10.0))
        assert len(replies) == 1, "".join(output)
        reply = replies[0]
        return (json.loads(reply.ok.payload.to_string()), None) if reply.ok is not None else \
            (None, reply.err.payload.to_string())

    header = db.Header(db.Time.from_ns(0), "azurekinect")
    given: list[db.Detection2D] = []  # the boxes sent with each frame: word detections of an earlier one
    stamp_ns = 1_000_000_000

    def step():
        nonlocal stamp_ns
        assert process.poll() is None, "".join(output)
        publisher.publish(bgr, stamp_ns=stamp_ns)
        time.sleep(0.03)  # the box message after its frame, as from a detector
        boxes_pub.publish(db.Detection2DArray(db.Header(db.Time.from_ns(stamp_ns), "azurekinect"), list(given)))
        stamp_ns += 100_000_000
        time.sleep(0.07)
        inst_sub.latest(timeout=0.0)
        det_sub.check()

    def latest_message():
        with lock:
            return detections[max(detections)] if detections else None

    try:
        deadline = time.monotonic() + 120.0
        while latest_message() is None and time.monotonic() < deadline:
            step()
        first = latest_message()
        assert first is not None, "".join(output)
        assert ask() == ({"prompts": WORDS, "stamp": None}, None)
        # the chairs and cups of that frame as given boxes, labelled apart from the words
        for d in first.detections:
            if d.results[0].hypothesis.class_id in ("chair", "cup"):
                given.append(db.Detection2D(header, [db.ObjectHypothesisWithPose(
                    db.ObjectHypothesis("given " + d.results[0].hypothesis.class_id, 0.25))], d.bbox, ""))
        assert len(given) >= 3
        sent = stamp_ns
        while time.monotonic() < deadline:
            step()
            with lock:
                if any(s >= sent for s in detections):
                    break
        with lock:
            message = detections[min(s for s in detections if s >= sent)]
        boxes = message.detections[:len(given)]
        assert [d.results[0].hypothesis.class_id for d in boxes] == [g.results[0].hypothesis.class_id for g in given]
        assert all(d.results[0].hypothesis.score == pytest.approx(0.25) for d in boxes)
        assert [boxes_of(db.Detection2DArray(header, [d]))[0] for d in boxes] == boxes_of(
            db.Detection2DArray(header, given))
        ids = [int(d.id) for d in message.detections]
        assert len(set(ids)) == len(ids)
        assert {d.results[0].hypothesis.class_id for d in message.detections[len(given):]} <= set(WORDS)

        # an exemplar: one chair box of the first frame, still kept by the node
        chair = next(boxes_of(db.Detection2DArray(header, [d]))[0] for d in first.detections
                     if d.results[0].hypothesis.class_id == "chair")
        request = {"prompts": [{"label": "seat", "boxes": [list(chair)]}], "stamp": first.header.stamp.to_ns()}
        got, error = ask(json.dumps(request))
        assert error is None and got == request, (error, "".join(output))
        given.clear()
        sent = stamp_ns
        while time.monotonic() < deadline:
            step()
            with lock:
                if any(s >= sent for s in detections):
                    break
        with lock:
            message = detections[min(s for s in detections if s >= sent)]
        seats = [boxes_of(db.Detection2DArray(header, [d]))[0] for d in message.detections]
        assert {d.results[0].hypothesis.class_id for d in message.detections} == {"seat"}
        chairs = [boxes_of(db.Detection2DArray(header, [d]))[0] for d in first.detections
                  if d.results[0].hypothesis.class_id == "chair"]
        assert all(max(box_iou(s, c) for c in chairs) > 0.5 for s in seats), (seats, chairs)

        # refused: a box outside the frame, and a frame the node no longer has; the prompts stay
        for bad, why in ((dict(request, prompts=[{"label": "x", "boxes": [[0, 0, w + 50, 10]]}]), "not inside"),
                         (dict(request, stamp=1), "no colour frame of stamp 1")):
            got, error = ask(json.dumps(bad))
            assert got is None and why in error, error
        assert ask()[0] == request
        client = shutil.which("xblick-sam3-prompts", path=os.path.dirname(sys.executable))
        shown = subprocess.run([client, "--camera", CAM, "--connect", f"127.0.0.1:{node_port}"], capture_output=True,
                               text=True, timeout=60)
        assert shown.returncode == 0 and json.loads(shown.stdout) == request, shown.stderr
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        inst_sub.close()
        det_sub.close()
        boxes_pub.close()
        publisher.close()
        session.close()
    assert process.returncode == 0, "".join(output)
