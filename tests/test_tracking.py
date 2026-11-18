"""Tracker: ids follow the masks, not the order of the scores (no GPU)."""

import numpy as np
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


# Image motion, gaps, merges and splits: synthetic sequences in a larger image (the tracker works on a half-size copy)
BH, BW = 240, 320


def big(x, y, w=20, h=20) -> torch.Tensor:
    mask = torch.zeros(BH, BW, dtype=torch.bool)
    mask[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = True
    return mask


def scene(boxes, word=0):
    return frame(*[(word, big(*b)) for b in boxes])


EMPTY = (torch.zeros(0, dtype=torch.long), torch.zeros(0, BH, BW, dtype=torch.bool))
OBJECTS = [(150, 40), (220, 60), (170, 140), (260, 170)]


def pan(dx, dy=0, objects=OBJECTS):
    return [(x + dx, y + dy) for x, y in objects]


def test_a_pan_moves_every_mask_at_once_and_the_ids_stay():
    # 15 px a frame on 20 px objects: overlap 5 / 35 and a shift of 0.75 sizes, too far without the image's motion
    for motion, kept in (("masks", True), ("none", False)):
        tracker = Tracker(motion=motion)
        first = tracker.update(*scene(OBJECTS))
        ids = [tracker.update(*scene(pan(-15 * k))) for k in (1, 2, 3)]
        assert all(i == first for i in ids) is kept, motion
    assert np.allclose(tracker.last_motion[1], 0)  # "none" estimates nothing


def test_a_pan_with_one_object_moving_on_its_own():
    tracker = Tracker()
    first = tracker.update(*scene(OBJECTS))
    for k in (1, 2, 3):
        boxes = pan(-15 * k)
        boxes[3] = (boxes[3][0] + 8 * k, boxes[3][1])  # it moves 8 px a frame against the pan
        assert tracker.update(*scene(boxes)) == first
        scale, shift, agree = tracker.last_motion
        # the pan (with four objects a slight zoom can explain 8 px too: the estimate stays near the pan)
        assert agree >= 3 and abs(shift[0] + 15) < 2 and abs(shift[1]) < 1 and abs(scale - 1) < 0.05


def test_a_zoom_keeps_the_ids():
    tracker = Tracker()
    objects = [(80, 60), (220, 60), (80, 160), (220, 160)]
    first = tracker.update(*frame(*[(0, big(x, y, 20, 20)) for x, y in objects]))
    s, c = 1.3, np.array([BW / 2, BH / 2])
    zoomed = []
    for x, y in objects:  # each box scaled about the image's centre
        x0, y0 = s * (np.array([x, y]) - c) + c
        zoomed.append((0, big(round(x0), round(y0), round(20 * s), round(20 * s))))
    assert tracker.update(*frame(*zoomed)) == first
    assert abs(tracker.last_motion[0] - s) < 0.05


def test_a_gap_during_a_pan():
    tracker = Tracker(max_missing=5)
    first = tracker.update(*scene(OBJECTS))
    tracker.update(*EMPTY)  # two frames without instances while the camera goes on (-12 px a frame)
    tracker.update(*EMPTY)
    assert tracker.update(*scene(pan(-36))) == first


def test_one_object_alone_follows_its_position_and_size():
    tracker = Tracker()
    (a,) = tracker.update(*scene([(50, 50)]))
    (b,) = tracker.update(*scene([(58, 50)]))  # 8 of 20 px: overlap 12 / 28 < ... then
    (c,) = tracker.update(*scene([(67, 50)]))  # 9 px: overlap 11 / 29, a shift of 0.45 sizes
    assert a == b == c
    assert tracker.last_motion[2] == 0  # one object: no motion of the image


def test_a_shifted_object_of_another_size_is_another_object():
    tracker = Tracker()
    (a,) = tracker.update(*scene([(50, 50)]))
    (b,) = tracker.update(*frame((0, big(58, 50, 60, 40))))  # near, but six times the area
    assert a != b


def test_an_object_missing_for_some_frames_comes_back_where_expected():
    tracker = Tracker(max_missing=10)
    first = tracker.update(*scene(OBJECTS))
    for _ in range(6):  # the last one hidden, the others pan 10 px a frame
        tracker.update(*scene(pan(-10 * (_ + 1), objects=OBJECTS[:3])))
    assert tracker.update(*scene(pan(-70))) == first


def test_merged_masks_keep_the_larger_id_and_the_smaller_comes_back_on_the_split():
    tracker = Tracker()
    big_one, small_one = big(30, 40, 40, 30), big(70, 40, 15, 30)
    a, b = tracker.update(*frame((0, big_one), (0, small_one)))
    (m,) = tracker.update(*frame((0, big_one | small_one)))  # the segmenter sees one object
    assert m == a
    assert tracker.update(*frame((0, small_one), (0, big_one))) == [b, a]


def test_a_split_mask_keeps_its_id_on_the_larger_part():
    tracker = Tracker()
    (a,) = tracker.update(*frame((0, big(30, 40, 60, 30))))
    small_part, large_part = big(30, 40, 20, 30), big(52, 40, 38, 30)
    s, large = tracker.update(*frame((0, small_part), (0, large_part)))
    assert large == a and s != a


def test_the_nodes_motion_choices_are_the_trackers():
    from xblick_sam3 import node  # noqa: PLC0415
    from xblick_sam3.tracking import MOTIONS  # noqa: PLC0415
    assert node.MOTIONS == MOTIONS
