"""``--ros`` (the core's branch ros2): the node as its own process with a stand-in segmenter (no checkpoint; a CUDA
device, as the node needs one). With ``--ros`` its detections and instances also appear on rmw_zenoh's keys, with
ROS's 33-byte attachment and liveliness tokens, and a ROS subscriber reads them; the prompts queryable (JSON) is no
ROS service. Without ``--ros`` nothing of ROS goes on the wire."""

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
from PIL import Image
from xblick import ros, topics
from xblick.session import RosOptions, open_session

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

CAM = "test/camera"
DOMAIN = 97
OFFICE = Path(__file__).parent / "data" / "office.png"
TOPICS = {f"/{topics.detections(CAM, 'sam3', 'detections_2d')}": "vision_msgs/msg/Detection2DArray",
          f"/{CAM}/sam3/instances/image_raw": "sensor_msgs/msg/Image",
          f"/{CAM}/diagnostics": "diagnostic_msgs/msg/DiagnosticArray"}
OURS = topics.detections(CAM, "sam3", "detections_2d")

# the node with a segmenter that finds one square per word, without SAM 3 and its checkpoint
STAND_IN = """
import sys, torch
from xblick_sam3 import node, segmenter

class StandIn:
    def __init__(self, prompts, threshold=0.5, device="cuda", checkpoint=None, compile=None, resolution=672,
                 box_head=False, **_):
        self.device = torch.device(device)
        self.set_prompts(prompts)

    @property
    def words(self):
        return [p.name for p in self.prompts]

    def set_prompts(self, prompts, image=None):
        self.prompts = segmenter.check_prompts(prompts, None if image is None else tuple(image.shape[:2]))

    def warm_up(self, h, w):
        pass

    def segment(self, image, boxes=None):
        h, w = image.shape[:2]
        n, d = len(self.prompts), self.device
        masks = torch.zeros((n, h, w), dtype=torch.bool, device=d)
        for i in range(n):
            masks[i, 10 + 40 * i:40 + 40 * i, 10:40] = True
        boxes_ = torch.tensor([[10.0, 10.0 + 40 * i, 40.0, 40.0 + 40 * i] for i in range(n)], device=d).reshape(-1, 4)
        instances = torch.zeros((h, w), dtype=torch.int32, device=d)
        for i in range(n):
            instances[masks[i]] = i + 1
        return segmenter.Segmentation(torch.arange(n, device=d), torch.full((n,), 0.9, device=d), boxes_, masks,
                                      instances.to(torch.uint16))

segmenter.Segmenter = StandIn
sys.exit(node.main(sys.argv[1:]))
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_every_default_key_is_a_ros_name():
    for key in (f"{CAM}/color/image_raw", OURS, f"{CAM}/sam3/instances/image_raw", f"{CAM}/sam3/prompts",
                f"{CAM}/diagnostics"):
        ros.topic_of(key)


def run_node(with_ros: bool, seconds: float = 90.0):
    """The stand-in node (and ``--ros``) on a published office frame. A reader that takes part in ROS subscribes to
    ``TOPICS`` (a mirror sends only to a ROS subscriber); an observer that takes part in nothing records every key with
    its attachment and every ROS liveliness token. Returns ({ROS topic: [payload]}, {key: attachment}, {token}, the
    node's output)."""
    bgr = np.ascontiguousarray(np.asarray(Image.open(OFFICE).convert("RGB"))[..., ::-1])
    h, w = bgr.shape[:2]
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    session = open_session(listen=[endpoint], ros_options=RosOptions(DOMAIN, "xblick_reader") if with_ros else None)
    observer = open_session(connect=[endpoint], declare=False)
    keys: dict[str, bytes | None] = {}
    tokens: set[str] = set()
    got: dict[str, list[bytes]] = {topic: [] for topic in TOPICS}
    watch = observer.declare_subscriber(
        "**", lambda s: keys.setdefault(str(s.key_expr), s.attachment.to_bytes() if s.attachment else None))
    live = observer.liveliness().declare_subscriber("@ros2_lv/**", lambda s: tokens.add(str(s.key_expr)),
                                                    history=True)
    readers = [session.ros.node.subscriber(topic, kind, lambda payload, *_, topic=topic: got[topic].append(payload))
               for topic, kind in TOPICS.items()] if with_ros else []
    info = db.CameraInfo.pinhole(w, h, 525.0, 525.0, w / 2 - 0.5, h / 2 - 0.5, frame_id="office")
    publisher = db.CameraPublisher(session, f"{CAM}/color", encoding="bgr8", frame_id="office", camera_info=info,
                                   source="test")
    # --ros-node: the stand-in is python -c, whose program name is not xblick-sam3
    ros_options = ["--ros", "--ros-domain", str(DOMAIN), "--ros-node", "xblick_sam3"] if with_ros else []
    process = subprocess.Popen([sys.executable, "-c", STAND_IN, "--camera", CAM, "--words", "chair,cup",
                                "--checkpoint", "stand_in.pt", "--no-download", "--local", "shm", "--connect",
                                endpoint, *ros_options], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    try:
        stamp_ns, deadline, seen = 1_000_000_000, time.monotonic() + seconds, 0
        while time.monotonic() < deadline:
            assert process.poll() is None, "".join(output)
            publisher.publish(bgr, stamp_ns=stamp_ns)
            stamp_ns += 100_000_000
            time.sleep(0.1)
            seen += OURS in keys
            if all(got.values()) if with_ros else seen >= 15:
                break
        time.sleep(0.3)
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
        for reader in readers:
            reader.close()
        publisher.close()
        watch.undeclare()
        live.undeclare()
        observer.close()
        session.close()
    assert process.returncode == 0, "".join(output)
    return got, keys, tokens, "".join(output)


def test_with_ros_the_detections_and_the_instances_reach_ros():
    got, keys, tokens, output = run_node(with_ros=True)
    assert [topic for topic, payloads in got.items() if not payloads] == [], output
    message = db.Detection2DArray.deserialize(got[f"/{OURS}"][0])
    assert [d.results[0].hypothesis.class_id for d in message.detections] == ["chair", "cup"]
    instances = db.Image.deserialize(got[f"/{CAM}/sam3/instances/image_raw"][0])
    assert (instances.encoding, instances.header.frame_id) == ("mono16", "office")
    for topic, type_name in TOPICS.items():
        key = ros.data_key(DOMAIN, topic, type_name, ros.hash_of(type_name))
        assert len(keys[key]) == 33, key
        seq, stamp, _ = ros.decode_attachment(keys[key])
        assert seq >= 1 and abs(stamp - time.time_ns()) < 60 * 10**9
    parsed = [ros.parse_token(t) for t in tokens]
    publishers = {t.topic for t in parsed if t.kind == ros.PUBLISHER and t.fully_qualified_node == "/xblick_sam3"}
    assert set(TOPICS) <= publishers, publishers
    assert not [t for t in tokens if "prompts" in t]  # the queryable (JSON): no ROS service
    assert OURS in keys  # our keys as before


def test_without_ros_nothing_of_ros_goes_on_the_wire():
    _, keys, tokens, _ = run_node(with_ros=False)
    assert OURS in keys
    assert not [k for k in keys if "::dds_::" in k or k[:1].isdigit()] and tokens == set()
