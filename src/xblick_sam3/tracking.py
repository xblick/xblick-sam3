"""Ids that stay with an object from frame to frame: each frame's instances matched to the last ones of the same word
by the overlap of their masks.

The segmenter sees every frame on its own; its instances come in order of score, which changes from frame to frame.
``Tracker.update`` gives each instance the id of the track whose last mask overlaps it most (intersection over union
at least ``min_iou``, each track to at most one instance: the assignment of the largest total overlap), else a new id.
A track not matched for more than ``max_missing`` frames ends. On the GPU, a few tenths of a millisecond a frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

MAX_ID = 65535  # ids are the values of the mono16 instance image: 1 .. 65535, then from 1 again


@dataclass
class Track:
    id: int
    word: object  # its instances' label
    mask: torch.Tensor  # (H, W) bool, the last matched one
    missing: int = 0  # frames since it was last matched


class Tracker:
    def __init__(self, min_iou: float = 0.3, max_missing: int = 15):
        if not 0.0 < min_iou <= 1.0:
            raise ValueError(f"min_iou {min_iou}: in (0, 1]")
        self.min_iou = min_iou
        self.max_missing = max_missing
        self.tracks: list[Track] = []
        self._next = 1

    def _new_id(self) -> int:
        taken = {t.id for t in self.tracks}
        while self._next in taken:
            self._next = self._next % MAX_ID + 1
        new, self._next = self._next, self._next % MAX_ID + 1
        return new

    def update(self, label, masks: torch.Tensor) -> list[int]:
        """The ids of a frame's instances (``label``: what each is of, a tensor or a list of any comparable values,
        e.g. prompt names; ``masks``: (N, H, W) bool), in their order."""
        labels = label.tolist() if isinstance(label, torch.Tensor) else list(label)
        ids: list[int | None] = [None] * len(labels)
        if self.tracks and masks.shape[1:] != self.tracks[0].mask.shape:
            raise ValueError(f"masks of {tuple(masks.shape[1:])}, the tracks' are {tuple(self.tracks[0].mask.shape)}")
        matched: set[int] = set()
        for word in set(labels):
            now = [i for i, w in enumerate(labels) if w == word]
            before = [k for k, t in enumerate(self.tracks) if t.word == word]
            if not before:
                continue
            a = masks[now].flatten(1).float()
            b = torch.stack([self.tracks[k].mask for k in before]).flatten(1).float()
            inter = a @ b.T
            union = a.sum(1, keepdim=True) + b.sum(1) - inter
            iou = (inter / union.clamp(min=1)).cpu().numpy()
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols, strict=True):
                if iou[r, c] >= self.min_iou:
                    track = self.tracks[before[c]]
                    ids[now[r]] = track.id
                    track.mask, track.missing = masks[now[r]], 0
                    matched.add(before[c])
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
                self.tracks.append(Track(ids[i], word, masks[i]))
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
