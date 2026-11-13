"""Tracker: ids follow the masks, not the order of the scores (no GPU)."""

import pytest
import torch

from xblick_sam3.tracking import MAX_ID, Tracker

H, W = 40, 60


def box(x, y, w=10, h=10) -> torch.Tensor:
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[y:y + h, x:x + w] = True
    return mask


def frame(*objects):
    """(label, masks) of instances given as (word, mask)."""
    return torch.tensor([w for w, _ in objects], dtype=torch.long), torch.stack([m for _, m in objects])


def test_ids_follow_moving_objects_whatever_their_order():
    tracker = Tracker()
    a, b = tracker.update(*frame((0, box(5, 5)), (0, box(40, 20))))
    assert a != b
    # both moved a little and swapped their order (scores changed)
    assert tracker.update(*frame((0, box(42, 21)), (0, box(7, 6)))) == [b, a]


def test_new_object_gets_a_new_id_and_words_never_mix():
    tracker = Tracker()
    (a,) = tracker.update(*frame((0, box(5, 5))))
    # the same place, another word: another object
    c, a2 = tracker.update(*frame((1, box(5, 5)), (0, box(6, 5))))
    assert a2 == a and c != a


def test_too_little_overlap_is_another_object():
    tracker = Tracker(min_iou=0.5)
    (a,) = tracker.update(*frame((0, box(5, 5))))
    (b,) = tracker.update(*frame((0, box(12, 5))))  # 3 of 10 columns overlap: IoU 30 / 170
    assert b != a


def test_unseen_object_keeps_its_id_for_max_missing_frames():
    tracker = Tracker(max_missing=2)
    (a,) = tracker.update(*frame((0, box(5, 5))))
    empty = (torch.zeros(0, dtype=torch.long), torch.zeros(0, H, W, dtype=torch.bool))
    tracker.update(*empty)
    tracker.update(*empty)
    assert tracker.update(*frame((0, box(5, 5)))) == [a]
    for _ in range(3):
        tracker.update(*empty)
    assert tracker.update(*frame((0, box(5, 5)))) != [a]


def test_ids_wrap_and_skip_live_ones():
    tracker = Tracker()
    tracker._next = MAX_ID
    first, second = tracker.update(*frame((0, box(0, 0)), (0, box(30, 20))))
    assert (first, second) == (MAX_ID, 1)


def test_instances_image_holds_ids_best_first():
    masks = torch.stack([box(5, 5), box(10, 10)])  # overlap at [10:15, 10:15]
    image = Tracker.instances_image([7, 3], masks)
    assert image.dtype == torch.uint16
    image = image.int()
    assert (image[5:10, 5:15] == 7).all() and (image[10:15, 10:15] == 7).all()
    assert (image[15:20, 10:20] == 3).all() and (image[:5] == 0).all()
    assert int(Tracker.instances_image([], torch.zeros(0, H, W, dtype=torch.bool)).int().max()) == 0


def test_mask_size_must_stay():
    tracker = Tracker()
    tracker.update(*frame((0, box(5, 5))))
    with pytest.raises(ValueError, match="masks of"):
        tracker.update(torch.tensor([0]), torch.zeros(1, H + 1, W, dtype=torch.bool))
