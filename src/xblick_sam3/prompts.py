"""xblick-sam3-prompts: shows or sets a running xblick-sam3's prompts over its queryable
``<camera>/<node>/prompts``.

    xblick-sam3-prompts --camera test/camera                       # the prompts now
    xblick-sam3-prompts --camera test/camera --words cup "soup can"
    xblick-sam3-prompts --camera test/camera --json '{"prompts": [...], "stamp": ...}'
                            [--node sam3] [--timeout 10] [--connect ...] [--listen ...] [--config FILE]

The queryable answers a query without payload with the prompts (JSON); one with a JSON payload replaces them all and
answers with the new ones (or an error, the prompts unchanged)::

    {"prompts": ["cup",                                               # a word: its text
                 {"text": "mug", "boxes": [[x1, y1, x2, y2]]},         # text and exemplars (instances like these)
                 {"label": "my tool", "boxes": [[...]], "negative": [[...]]}],  # exemplars alone: label required
     "stamp": 1730000000123456789}

Boxes are in pixels of the colour frame of ``stamp`` (nanoseconds, its header's), which the node still has to have (it
keeps its last --history frames); without ``stamp``, of its newest frame. The answer carries the stamp of the frame
the boxes were encoded on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .segmenter import Prompt

PROMPTS = "prompts"  # <camera>/<node>/prompts
ASK_AGAIN_S = 0.2  # a query that reached no queryable: asked again after this


def parse(request: dict) -> tuple[list[Prompt], int | None]:
    """The prompts and stamp of a request; ValueError for anything not as documented above."""
    if not isinstance(request, dict) or set(request) - {"prompts", "stamp"} or "prompts" not in request:
        raise ValueError('a request is {"prompts": [...], "stamp": ns (optional)}')
    if not isinstance(request["prompts"], list):
        raise ValueError("prompts: a list")
    stamp = request.get("stamp")
    if stamp is not None and (not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0):
        raise ValueError(f"stamp {stamp!r}: nanoseconds, an integer")
    prompts = []
    for item in request["prompts"]:
        if isinstance(item, str):
            prompts.append(Prompt(text=item.strip()))
            continue
        if not isinstance(item, dict) or set(item) - {"text", "label", "boxes", "negative"}:
            raise ValueError(f"prompt {item!r}: a string or an object of text, label, boxes, negative")
        boxes = {}
        for key in ("boxes", "negative"):
            given = item.get(key, [])
            if not isinstance(given, list) or not all(
                    isinstance(b, list) and len(b) == 4 and all(isinstance(v, int | float) for v in b) for b in given):
                raise ValueError(f"prompt {item!r}: {key} a list of [x1, y1, x2, y2]")
            boxes[key] = tuple(tuple(float(v) for v in b) for b in given)
        for key in ("text", "label"):
            if key in item and not isinstance(item[key], str):
                raise ValueError(f"prompt {item!r}: {key} a string")
        prompts.append(Prompt(text=item.get("text"), label=item.get("label"), **boxes))
    if stamp is not None and not any(p.exemplars for p in prompts):
        raise ValueError("a stamp without boxes: it is the frame the boxes are in")
    return prompts, stamp


def describe(prompts: list[Prompt], stamp: int | None) -> dict:
    """The prompts as a request that would set them (the answer of the queryable)."""
    items = []
    for p in prompts:
        if not p.exemplars and p.label is None:
            items.append(p.text)
            continue
        item = {k: v for k, v in (("text", p.text), ("label", p.label)) if v is not None}
        item.update({k: [list(b) for b in v] for k, v in (("boxes", p.boxes), ("negative", p.negative)) if v})
        items.append(item)
    return {"prompts": items, "stamp": stamp}


def main(argv=None) -> int:
    from xblick import topics  # noqa: PLC0415
    from xblick.session import add_arguments, open_from_args  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="xblick-sam3-prompts", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="the camera's base, e.g. test/camera")
    parser.add_argument("--node", default="sam3", help="the node's part of the topic: <camera>/<node>/prompts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--words", nargs="+", help="set these words as the prompts (each also comma-separated)")
    group.add_argument("--json", help="set the prompts of this request (above)")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="seconds to wait for the answer (the node answers between frames)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    key = f"{topics.check_name(args.camera)}/{args.node}/{PROMPTS}"
    payload = None
    if args.words:
        payload = json.dumps({"prompts": [s.strip() for w in args.words for s in w.split(",") if s.strip()]})
    elif args.json:
        parse(json.loads(args.json))  # fails here, not in the node
        payload = args.json
    session = open_from_args(args)
    try:
        # right after the session opens a query reaches no queryable yet (Durchblick's find_streams: their declarations
        # arrive some 0.1-0.3 s later): asked again until an answer comes or the timeout ends
        replies, end = [], time.monotonic() + args.timeout
        while not replies and (remaining := end - time.monotonic()) > 0:
            replies = list(session.get(key, payload=payload, timeout=remaining))
            if not replies:
                time.sleep(min(ASK_AGAIN_S, max(0.0, end - time.monotonic())))
    finally:
        session.close()
    if not replies:
        print(f"xblick-sam3-prompts: no answer on {key} within {args.timeout} s", file=sys.stderr)
        return 1
    failed = False
    for reply in replies:
        if reply.ok is not None:
            print(reply.ok.payload.to_string())
        else:
            print(f"xblick-sam3-prompts: {reply.err.payload.to_string()}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
