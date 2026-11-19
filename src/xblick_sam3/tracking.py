"""Ids that stay with an object from frame to frame: each frame's instances matched to the objects of the same word
seen before, by the overlap of their masks, by position and size, after the motion of the whole image.

The segmenter sees every frame on its own; its instances come in order of score, which changes from frame to frame.
``Tracker.update`` keeps each object's last mask in a smaller copy (``work_width`` pixels wide at most) and per frame:

1. *Motion of the image* (``motion="masks"``): when the camera pans, tilts, drives or zooms, every object's mask moves
   at once and its overlap with the last one drops. The tracker estimates one shift and scale of the whole image from
   the masks themselves: each pair of an object and an instance of its word with a similar size proposes the shift
   and scale that maps the one onto the other; the proposal that the most objects agree with (their centres land
   within a quarter of their size of an instance) wins, refined by least squares over those objects. It needs at
   least two objects that agree; with fewer, the image counts as still (one object alone cannot tell the camera's
   motion from its own). Every object's mask, also of those unseen for some frames, then moves with the image.
   ``motion="none"``: no estimate, the masks stay where they were.
2. *Matching*: an instance keeps the id of the object whose moved mask overlaps it most (intersection over union at
   least ``min_iou``) or, failing that, whose moved centre is within ``max_distance`` times the object's size (the
   square root of its area) of the instance's centre with areas at most ``max_size_ratio`` apart. Each object to at
   most one instance: the assignment of the largest total score (overlap first, then closeness, then objects seen more
   recently).
3. *Merge and split*: an instance that covers at least ``COVER`` of the masks of two objects (two masks merged into
   one) keeps the id of the larger of them; the other waits as unseen. An object whose mask split into two instances
   keeps its id on the larger part.
4. An instance matched to no object is a new object, a new id. An object unseen for more than ``max_missing`` frames
   ends; until then it can take an instance where it is expected (its moved mask, or its moved centre and size).

In-plane rotation of the image is not estimated (a head pans and tilts; a roll of a few degrees moves masks little).
On the GPU, about 1.3 ms a frame for six instances (docs/topics.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

MAX_ID = 65535  # ids are the values of the mono16 instance image: 1 .. 65535, then from 1 again
MOTIONS = ("masks", "none")  # --motion: the image's motion estimated from the masks, or none
COVER = 0.5  # the share of an object's mask an instance has to cover to count as that object merged into it
AGREE = 0.25  # a proposal of the image's motion: an object agrees when its centre lands this close (times its size)
MAX_SCALE = 1.5  # the most a scale may change between two frames (a zoom); larger proposals are refused
WORK_WIDTH = 160  # the width of the tracker's copies of the masks (at most; the smaller images as they are)


@dataclass
class Track:
    id: int
    word: object  # its instances' label
    mask: torch.Tensor  # (h, w) bool: the last matched one, moved with the image since, in the working size
    centre: np.ndarray  # (2,) x, y of ``mask`` (working pixels)
    area: float  # pixels of the last matched mask (working pixels)
    missing: int = 0  # frames since it was last matched


def _stats(masks: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Centres (N, 2) x, y and areas (N,) of masks (N, h, w)."""
    if len(masks) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    m = masks.float()
    area = m.sum((1, 2))
    h, w = masks.shape[1:]
    xs = torch.arange(w, device=m.device, dtype=torch.float32)
    ys = torch.arange(h, device=m.device, dtype=torch.float32)
    cx = (m.sum(1) * xs).sum(1) / area.clamp(min=1)
    cy = (m.sum(2) * ys).sum(1) / area.clamp(min=1)
    out = torch.stack([cx, cy, area], 1).cpu().numpy()
    return out[:, :2].astype(np.float64), out[:, 2].astype(np.float64)


def estimate_motion(before: list[tuple[object, np.ndarray, float]], now: list[tuple[object, np.ndarray, float]],
                    size: tuple[int, int], max_size_ratio: float) -> tuple[float, np.ndarray, int]:
    """The image's shift and scale from the objects ``before`` (word, centre, area) to the instances ``now``: a
    point p moves to ``scale * (p - c) + c + shift`` with c the image's centre (``size`` h, w). Returns scale, shift
    and the number of objects that agree; (1, 0, 0) with fewer than two that agree."""
    if len(before) < 2 or len(now) < 2:
        return 1.0, np.zeros(2), 0
    c = np.array([size[1] / 2, size[0] / 2])
    pb = np.array([b[1] for b in before]) - c
    pn = np.array([n[1] for n in now]) - c
    ab = np.array([b[2] for b in before])
    an = np.array([n[2] for n in now])
    same = np.array([[b[0] == n[0] for n in now] for b in before])
    ratio = an[None, :] / np.maximum(ab[:, None], 1)
    pairs = same & (ratio <= max_size_ratio) & (ratio >= 1 / max_size_ratio)
    best = (0, 1.0, np.zeros(2), None)
    for j, i in zip(*np.nonzero(pairs), strict=True):
        s = float(np.sqrt(ratio[j, i]))
        if not 1 / MAX_SCALE <= s <= MAX_SCALE:
            continue
        t = pn[i] - s * pb[j]
        moved = s * pb + t  # each object's centre under this proposal
        dist = np.linalg.norm(moved[:, None, :] - pn[None, :, :], axis=2) / np.sqrt(np.maximum(ab, 1))[:, None] / s
        ok = pairs & (dist <= AGREE)
        agree = int(ok.any(1).sum())
        if agree > best[0]:
            best = (agree, s, t, ok)
    agree, s, t, ok = best
    if agree < 2:
        return 1.0, np.zeros(2), 0
    # least squares over the agreeing objects (each with its nearest instance): pn = s * pb + t
    js = np.nonzero(ok.any(1))[0]
    moved = s * pb + t
    is_ = [int(np.argmin(np.where(ok[j], np.linalg.norm(moved[j] - pn, axis=1), np.inf))) for j in js]
    x, y = pb[js], pn[is_]
    xm, ym = x.mean(0), y.mean(0)
    var = ((x - xm) ** 2).sum()
    s_ls = float(((x - xm) * (y - ym)).sum() / var) if var > 1e-6 else s
    s_ls = s_ls if 1 / MAX_SCALE <= s_ls <= MAX_SCALE else s
    return s_ls, ym - s_ls * xm, agree


def warp(masks: torch.Tensor, scale: float, shift: np.ndarray) -> torch.Tensor:
    """Masks (N, h, w) bool moved as ``estimate_motion`` says (nearest pixel; what moves out of the image is lost)."""
    if len(masks) == 0 or (scale == 1.0 and not shift.any()):
        return masks
    h, w = masks.shape[1:]
    # output pixel q comes from p = (q - c - shift) / scale + c, in normalised coordinates (align_corners=False)
    theta = torch.tensor([[1 / scale, 0.0, -2 * shift[0] / (w * scale)],
                          [0.0, 1 / scale, -2 * shift[1] / (h * scale)]], dtype=torch.float32, device=masks.device)
    grid = F.affine_grid(theta[None].expand(len(masks), 2, 3), [len(masks), 1, h, w], align_corners=False)
    return F.grid_sample(masks.float()[:, None], grid, mode="nearest", align_corners=False)[:, 0] > 0.5


class Tracker:
    def __init__(self, min_iou: float = 0.3, max_missing: int = 30, max_distance: float = 0.5,
                 max_size_ratio: float = 2.0, motion: str = "masks", work_width: int = WORK_WIDTH):
        if not 0.0 < min_iou <= 1.0:
            raise ValueError(f"min_iou {min_iou}: in (0, 1]")
        if max_distance < 0 or max_size_ratio < 1 or max_missing < 0:
            raise ValueError(f"max_distance {max_distance}, max_size_ratio {max_size_ratio}, max_missing "
                             f"{max_missing}: not negative, at least 1, not negative")
        if motion not in MOTIONS:
            raise ValueError(f"motion {motion!r}: one of {', '.join(MOTIONS)}")
        self.min_iou = min_iou
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.max_size_ratio = max_size_ratio
        self.motion = motion
        self.work_width = work_width
        self.tracks: list[Track] = []
        self.shape: tuple[int, int] | None = None  # the full masks' (H, W)
        self.last_motion = (1.0, np.zeros(2), 0)  # the last frame's estimate (scale, shift in full pixels, agreeing)
        self._next = 1
        self.issued = 0  # ids given so far (new objects)

    def _new_id(self) -> int:
        taken = {t.id for t in self.tracks}
        while self._next in taken:
            self._next = self._next % MAX_ID + 1
        new, self._next = self._next, self._next % MAX_ID + 1
        self.issued += 1
        return new

    def _small(self, masks: torch.Tensor) -> torch.Tensor:
        factor = max(1, masks.shape[2] // self.work_width)
        if factor == 1 or len(masks) == 0:
            return masks if factor == 1 else masks[:, ::factor, ::factor]
        return F.avg_pool2d(masks.float()[:, None], factor)[:, 0] >= 0.5

    def update(self, label, masks: torch.Tensor) -> list[int]:
        """The ids of a frame's instances (``label``: what each is of, a tensor or a list of any comparable values,
        e.g. prompt names; ``masks``: (N, H, W) bool), in their order."""
        labels = label.tolist() if isinstance(label, torch.Tensor) else list(label)
        if self.shape is not None and tuple(masks.shape[1:]) != self.shape:
            raise ValueError(f"masks of {tuple(masks.shape[1:])}, the tracks' are {self.shape}")
        self.shape = tuple(masks.shape[1:])
        small = self._small(masks)
        centres, areas = _stats(small)
        ids: list[int | None] = [None] * len(labels)
        size = tuple(small.shape[1:])
        # 1. the image's motion, and every object's mask moved with it
        if self.motion == "masks" and self.tracks:
            scale, shift, agree = estimate_motion(
                [(t.word, t.centre, t.area) for t in self.tracks],
                [(w, centres[i], areas[i]) for i, w in enumerate(labels) if areas[i] > 0], size, self.max_size_ratio)
            if agree:
                moved = warp(torch.stack([t.mask for t in self.tracks]), scale, shift)
                c = np.array([size[1] / 2, size[0] / 2])
                for t, m in zip(self.tracks, moved, strict=True):
                    t.mask, t.centre, t.area = m, scale * (t.centre - c) + c + shift, t.area * scale * scale
            factor = self.shape[1] / size[1]
            self.last_motion = (scale, shift * factor, agree)
        # 2. matching, word by word; the overlaps of all instances with all objects at once (one copy from the GPU)
        matched: dict[int, int] = {}  # track index -> instance index
        if self.tracks and labels:
            a = small.flatten(1).float()
            b = torch.stack([t.mask for t in self.tracks]).flatten(1).float()
            inter_all, b_area_all = (x.cpu().numpy() for x in (a @ b.T, b.sum(1)))
        for word in set(labels):
            now = [i for i, w in enumerate(labels) if w == word]
            before = [k for k, t in enumerate(self.tracks) if t.word == word]
            if not before:
                continue
            inter_np = inter_all[np.ix_(now, before)]
            b_area = b_area_all[before]
            iou = inter_np / np.maximum(areas[now][:, None] + b_area[None] - inter_np, 1)
            tb = [self.tracks[k] for k in before]
            dist = np.linalg.norm(centres[now][:, None, :] - np.array([t.centre for t in tb])[None], axis=2) \
                / np.sqrt(np.maximum([t.area for t in tb], 1))[None]
            ratio = areas[now][:, None] / np.maximum([t.area for t in tb], 1)[None]
            near = (dist <= self.max_distance) & (ratio <= self.max_size_ratio) & (ratio >= 1 / self.max_size_ratio)
            ok = (iou >= self.min_iou) | near
            missing = np.array([t.missing for t in tb])[None]
            score = np.where(ok, 2 * iou + np.where(near, 1 - dist / max(self.max_distance, 1e-6), 0)
                             - 1e-3 * missing, -1e6)
            rows, cols = linear_sum_assignment(-score)
            for r, c in zip(rows, cols, strict=True):
                if ok[r, c]:
                    matched[before[c]] = now[r]
            # 3. merge: an instance covering COVER of several objects' masks keeps the largest one's id
            cover = inter_np / np.maximum(b_area, 1)[None]
            for r, i in enumerate(now):
                covered = [before[c] for c in range(len(before))
                           if cover[r, c] >= COVER and self.tracks[before[c]].missing == 0]
                if len(covered) < 2:
                    continue
                largest = max(covered, key=lambda k: self.tracks[k].area)
                holder = next((k for k, v in matched.items() if v == i), None)
                if holder is not None and holder != largest and largest not in matched:
                    del matched[holder]
                    matched[largest] = i
            # split: an object whose mask holds two instances keeps its id on the larger
            for c, k in enumerate(before):
                if k not in matched:
                    continue
                inside = [now[r] for r in range(len(now)) if inter_np[r, c] >= COVER * max(areas[now[r]], 1)]
                if len(inside) < 2:
                    continue
                larger = max(inside, key=lambda i: areas[i])
                if matched[k] != larger and larger not in matched.values():
                    matched[k] = larger
        # 4. ids, new objects, unseen objects
        for k, i in matched.items():
            t = self.tracks[k]
            ids[i] = t.id
            t.mask, t.centre, t.area, t.missing = small[i], centres[i], float(areas[i]), 0
        kept = []
        for k, track in enumerate(self.tracks):
            if k not in matched:
                track.missing += 1
            if track.missing <= self.max_missing:
                kept.append(track)
        self.tracks = kept
        for i, word in enumerate(labels):
            if ids[i] is None:
                ids[i] = self._new_id()
                self.tracks.append(Track(ids[i], word, small[i], centres[i], float(areas[i])))
        return ids  # type: ignore[return-value]

    @staticmethod
    def instances_image(ids: list[int], masks: torch.Tensor) -> torch.Tensor:
        """The masks as one uint16 image of their ids, 0 where none is; where masks overlap, the first (the better
        instance: the segmenter orders by score)."""
        h, w = masks.shape[1:]
        if not ids:
            return torch.zeros(h, w, dtype=torch.uint16, device=masks.device)
        first = torch.argmax(masks.to(torch.uint8), dim=0)
        values = torch.as_tensor(np.asarray(ids, np.int32), device=masks.device)
        return torch.where(masks.any(dim=0), values[first], 0).to(torch.uint16)
