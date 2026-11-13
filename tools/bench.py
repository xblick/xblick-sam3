"""Times the segmenter on a directory of images: milliseconds per frame (median, 90th percentile) from the frame in
host memory to the instances on the GPU, and instances per frame. The README's numbers come from it.

    .venv/bin/python tools/bench.py IMAGES_DIR --words cup bottle chair [--compile default] [--resolution 1008]
                                    [--frames 50]

IMAGES_DIR: PNG or JPEG frames, every one of the same size (e.g. TUM RGB-D's rgb/ directory).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from xblick_sam3.segmenter import Segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path)
    parser.add_argument("--words", nargs="+", required=True)
    parser.add_argument("--compile", default=None, metavar="MODE")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=50, help="frames timed (spread over the directory)")
    args = parser.parse_args()
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(files) < args.frames:
        raise SystemExit(f"{args.images}: {len(files)} images, --frames {args.frames}")
    files = files[:: len(files) // args.frames][: args.frames]
    frames = [np.ascontiguousarray(cv2.imread(str(f))[..., ::-1]) for f in files]
    if len({f.shape for f in frames}) != 1:
        raise SystemExit(f"{args.images}: images of different sizes")
    started = time.perf_counter()
    segmenter = Segmenter(args.words, threshold=args.threshold, compile=args.compile, resolution=args.resolution)
    segmenter.warm_up(*frames[0].shape[:2])
    print(f"model and warm-up {time.perf_counter() - started:.0f} s", flush=True)
    ms, instances = [], []
    for frame in frames:
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = segmenter.segment(torch.from_numpy(frame))
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1e3)
        instances.append(len(result.score))
    print(f"{len(args.words)} words, compile {args.compile}, {args.resolution} px: {np.median(ms):.1f} ms per frame "
          f"(90th percentile "
          f"{np.percentile(ms, 90):.1f}), {np.mean(instances):.1f} instances per frame, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}", flush=True)


if __name__ == "__main__":
    main()
