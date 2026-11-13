"""The segmenter against SAM 3's own forward pass (needs a CUDA device and access to facebook/sam3.1).

tests/data/office.png: the first colour frame of TUM RGB-D's freiburg3_long_office_household (CC BY 4.0).
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from xblick_sam3.segmenter import Prompt, Segmenter  # noqa: E402

WORDS = ["keyboard", "monitor", "cup", "book", "chair", "teddy bear", "plant", "computer mouse"]


def office() -> torch.Tensor:
    import cv2  # noqa: PLC0415

    bgr = cv2.imread(str(Path(__file__).parent / "data" / "office.png"))
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))


@pytest.fixture(scope="module")
def segmenter():
    return Segmenter(WORDS, threshold=0.5)


@pytest.fixture(scope="module")
def at_672():
    return Segmenter(WORDS, threshold=0.5, resolution=672, box_head=True)


def iou_matrix(a, b):
    inter = a.flatten(1).float() @ b.flatten(1).float().T
    return inter / (a.flatten(1).sum(1)[:, None] + b.flatten(1).sum(1)[None] - inter).clamp(min=1)


def test_masks_of_the_detections_alone_equal_sams_mask_head(segmenter):
    """_mask_logits (the mask head for the detections alone) gives SAM 3's masks: Sam3Image.forward_grounding's,
    computed for every word and query."""
    from sam3.model.data_misc import interpolate  # noqa: PLC0415, F401 - sam3 importable

    image = office().cuda()
    result = segmenter.segment(image)
    assert len(result.score) >= 5  # chairs, cups, books
    assert sorted({WORDS[i] for i in result.label.tolist()}) == ["book", "chair", "cup"]
    m = segmenter.model
    with torch.inference_mode(), segmenter._autocast():
        backbone_out = m.backbone.forward_image(segmenter.preprocess(image), need_interactive_out=False,
                                                need_propagation_out=False)
        backbone_out.update(segmenter._text)
        out = m.forward_grounding(backbone_out=backbone_out, find_input=segmenter._find, find_target=None,
                                  geometric_prompt=segmenter._prompt)
    score = out["pred_logits"].squeeze(-1).float().sigmoid()
    word, query = torch.nonzero(score >= 0.5, as_tuple=True)
    order = torch.argsort(score[word, query], descending=True)
    word, query = word[order], query[order]
    assert torch.equal(word, result.label)
    torch.testing.assert_close(score[word, query], result.score)
    h, w = image.shape[:2]
    full = torch.nn.functional.interpolate(out["pred_masks"][word, query].float()[:, None], size=(h, w),
                                           mode="bilinear", align_corners=False)[:, 0] > 0
    for a, b in zip(full, result.masks, strict=True):
        iou = (a & b).sum() / (a | b).sum()
        assert iou > 0.99, iou


def test_instances_image_has_each_masks_index_best_first(segmenter):
    result = segmenter.segment(office().cuda())
    instances = result.instances.cpu().numpy().astype(np.int64)
    masks = result.masks.cpu().numpy()
    assert instances.dtype == np.int64 and set(np.unique(instances)) <= set(range(len(masks) + 1))
    assert ((instances > 0) == masks.any(axis=0)).all()
    for i, mask in enumerate(masks):
        on = instances[mask]
        assert (on <= i + 1).all()  # no worse instance over a better one
        better = masks[:i].any(axis=0) if i else np.zeros_like(mask)
        assert (instances[mask & ~better] == i + 1).all()


def test_frame_without_the_words_gives_no_instances(segmenter):
    result = segmenter.segment(torch.zeros(48, 64, 3, dtype=torch.uint8))
    assert len(result.score) == 0 and result.masks.shape == (0, 48, 64)
    assert result.instances.shape == (48, 64) and int(result.instances.int().max()) == 0


def test_at_672_the_sure_instances_stay(segmenter):
    """At 672 x 672 (half the time a frame) the instances SAM 3 is sure of at 1008 (score >= 0.6) are found again (score
    >= 0.4, same word, mask IoU >= 0.5); the README's numbers for 80 frames: 98 to 99 %."""
    image = office().cuda()
    full = segmenter.segment(image)
    small = Segmenter(WORDS, threshold=0.4, resolution=672).segment(image)
    sure = torch.nonzero(full.score >= 0.6)[:, 0].tolist()
    assert len(sure) >= 3
    for i in sure:
        same = small.masks[small.label == full.label[i]]
        iou = (same & full.masks[i]).flatten(1).sum(1) / (same | full.masks[i]).flatten(1).sum(1)
        assert len(iou) and iou.max() >= 0.5, (WORDS[full.label[i]], float(full.score[i]), iou)


def test_resolution_has_to_be_whole_attention_windows():
    with pytest.raises(ValueError, match="multiple of 336"):
        Segmenter(WORDS, resolution=840)


def test_box_head_segments_each_box_as_the_detector_does(at_672):
    """The detector's own boxes as box prompts: one mask each, the same object (median mask IoU 0.92 over 100 boxes of
    10 TUM frames, 2 % below 0.5)."""
    image = office().cuda()
    result = at_672.segment(image)
    boxed = at_672.segment(image, result.boxes)
    assert boxed.box_masks.shape == result.masks.shape and bool(boxed.box_found.all())
    iou = torch.diagonal(iou_matrix(boxed.box_masks, result.masks))
    assert float(iou.median()) > 0.85 and float((iou < 0.5).float().mean()) <= 0.2, iou
    assert torch.equal(boxed.label, result.label)  # the prompts' instances as without boxes
    empty = at_672.segment(image, torch.zeros(0, 4))
    assert empty.box_masks.shape == (0, *image.shape[:2]) and len(empty.box_found) == 0


def test_exemplar_box_finds_instances_like_it(at_672):
    """A chair's box as the one exemplar ("seat", no text): its instances are the chairs, the boxed one among them;
    words beside it keep their instances."""
    image = office().cuda()
    words = at_672.segment(image)
    chairs = words.masks[words.label == WORDS.index("chair")]
    box = tuple(words.boxes[int(torch.nonzero(words.label == WORDS.index("chair"))[0])].tolist())
    try:
        at_672.set_prompts([Prompt(label="seat", boxes=(box,))], image)
        seats = at_672.segment(image)
        assert len(seats.score) >= 1 and at_672.words == ["seat"]
        overlap = iou_matrix(seats.masks, chairs)
        assert bool((overlap.max(dim=1).values > 0.5).all()), overlap  # every seat a chair
        assert float(overlap.max()) > 0.8  # the boxed one
        at_672.set_prompts([*WORDS, Prompt(label="seat", boxes=(box,))], image)
        both = at_672.segment(image)
        for i, word in enumerate(WORDS):  # the words' instances as without the exemplar prompt
            assert int((both.label == i).sum()) == int((words.label == i).sum()), word
        assert int((both.label == len(WORDS)).sum()) == len(seats.score)
    finally:
        at_672.set_prompts(WORDS)


def test_bad_prompts_are_refused_and_change_nothing(at_672):
    image = office().cuda()
    h, w = image.shape[:2]
    with pytest.raises(ValueError, match="need the frame"):
        at_672.set_prompts([Prompt(label="x", boxes=((0.0, 0.0, 5.0, 5.0),))])
    with pytest.raises(ValueError, match="not inside the frame"):
        at_672.set_prompts(["cup", Prompt(label="x", boxes=((0.0, 0.0, w + 1.0, 5.0),))], image)
    with pytest.raises(ValueError, match="named twice"):
        at_672.set_prompts(["cup", Prompt(text="mug", label="cup")])
    assert at_672.words == WORDS
    at_672.set_prompts([])
    try:
        nothing = at_672.segment(image)
        assert len(nothing.score) == 0 and nothing.box_masks is None
        assert len(at_672.segment(image, torch.tensor([[10.0, 10.0, 100.0, 100.0]])).box_masks) == 1
    finally:
        at_672.set_prompts(WORDS)
