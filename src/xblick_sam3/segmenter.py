# SPDX-License-Identifier: LicenseRef-SAM-License
#
# This file is a derivative work of the SAM Materials (SAM 3 by Meta, github.com/facebookresearch/sam3): it assembles
# and adapts SAM 3's detector from Meta's model code. It is distributed under the terms of Meta's SAM License, see
# LICENSES/SAM-License.txt. Modifications: Copyright 2026 David Schneider. The other files of this repository are
# under the Apache License 2.0 (LICENSE).

"""SAM 3.1's detector on single images: instance masks of every prompt of a list, all prompts in one batch; and,
optionally, SAM's interactive head: one mask per given box.

SAM 3.1 (facebookresearch/sam3, checkpoint facebook/sam3.1) is a detector (an image encoder, a text encoder, a
transformer that grounds each prompt in the image, a mask head) and a video tracker that follows its detections. This
module uses the detector alone: every frame on its own, no state between frames. A prompt is a text, example boxes
(SAM 3's exemplars: "instances like this one", or negative: "not like this") or both (``Prompt``). The prompts are
encoded once (``set_prompts``; exemplar boxes on the frame they were drawn in); a frame then costs one pass of the image
encoder plus one batched pass of the grounding transformer and mask head for all prompts (SAM 3's ``Sam3Processor``
runs the latter once per prompt).

With ``box_head`` the tracker's interactive head (SAM's prompt encoder and mask decoder, from the same checkpoint) gives
one mask per box of ``segment(image, boxes)``: that box's object, as SAM segments a box (no search for others).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# SAM 3.1's checkpoint (Hugging Face's gated facebook/sam3.1): where weights.py downloads it, the node's cache directory
from .weights import STORE, needed

RESOLUTION = 1008  # SAM 3's input: the image scaled to 1008 x 1008 (no padding), 72 x 72 patches of 14 pixels
PATCH, WINDOW = 14, 24  # the ViT's patches (pixels) and its attention windows (patches)
VISUAL = "visual"  # SAM 3's text of a prompt of boxes alone (Sam3Processor.add_geometric_prompt)
WARM_UP_WORDS = ("object", "thing")  # Segmenter.warm_up's prompts
# what Segmenter.set_prompts sets (warm_up keeps and restores it)
PROMPT_STATE = ("prompts", "_text", "_find", "_prompt", "_empty", "_geo", "_geo_mask", "_fill", "_words")

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Prompt:
    """What to segment: ``text`` (a word or phrase), ``boxes`` (exemplars: instances like these, x1 y1 x2 y2 in pixels
    of the frame they were drawn in), ``negative`` (boxes of what is not meant) or text and boxes together. ``label``
    names its instances (default: the text; required without one)."""

    text: str | None = None
    boxes: tuple[Box, ...] = ()
    negative: tuple[Box, ...] = ()
    label: str | None = None

    def __post_init__(self):
        if self.text is not None and not self.text.strip():
            raise ValueError("a prompt's text is empty")
        if self.text is None and not self.boxes:
            raise ValueError(f"prompt {self.label!r}: a text or at least one (positive) box")
        if self.text is None and not self.label:
            raise ValueError("a prompt of boxes alone needs a label")
        for box in (*self.boxes, *self.negative):
            if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError(f"prompt {self.name!r}: box {box} is no x1 y1 x2 y2 with x1 < x2, y1 < y2")

    @property
    def name(self) -> str:
        return self.label or self.text  # type: ignore[return-value]

    @property
    def exemplars(self) -> bool:
        return bool(self.boxes or self.negative)


@dataclass
class Segmentation:
    """The instances of one frame, best first; on the segmenter's device.

    ``label`` indexes the segmenter's prompts; ``boxes`` are x1, y1, x2, y2 in pixels of the frame; ``masks`` the
    frame's size. ``instances`` (uint16, the frame's size) is the masks as one image: 0 where no instance is, else the
    instance's index + 1, the best scoring instance where masks overlap. With boxes given to ``segment``: ``box_masks``
    (M, H, W) one mask per box, in their order; ``box_scores`` (M,) the head's estimate of its mask's IoU;
    ``box_found`` (M,) whether the head sees an object in the box (its object score) and the mask is not empty."""

    label: torch.Tensor  # (N,) int64
    score: torch.Tensor  # (N,) float32
    boxes: torch.Tensor  # (N, 4) float32
    masks: torch.Tensor  # (N, H, W) bool
    instances: torch.Tensor  # (H, W) uint16
    box_masks: torch.Tensor | None = None  # (M, H, W) bool
    box_scores: torch.Tensor | None = None  # (M,) float32
    box_found: torch.Tensor | None = None  # (M,) bool


def load_checkpoint(checkpoint: str | os.PathLike) -> dict:
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"{checkpoint}: no SAM 3.1 checkpoint (xblick-sam3 --download-only downloads it)")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    return state.get("model", state)


def build_detector(checkpoint: str | os.PathLike, device: torch.device, rope_real: bool = False,
                   resolution: int = RESOLUTION):
    """SAM 3.1's detector as ``sam3.model_builder.build_sam3_multiplex_video_predictor`` builds it, its weights from
    the checkpoint's ``detector.`` part; every one of its weights has to be there (no partial loading). ``rope_real``:
    the ViT's rotary embedding in real numbers (torch.compile cannot compile the complex ones). ``resolution``: the
    input size it runs at, a multiple of 336 (whole attention windows of 24 x 24 patches)."""
    import pkg_resources  # noqa: PLC0415 - sam3 finds its tokenizer's vocabulary so
    from sam3 import model_builder as mb  # noqa: PLC0415 - heavy
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector  # noqa: PLC0415
    from sam3.model.vl_combiner import SAM3VLBackboneTri  # noqa: PLC0415

    if resolution <= 0 or resolution % (PATCH * WINDOW):
        raise ValueError(f"resolution {resolution}: a multiple of {PATCH * WINDOW} (whole attention windows of "
                         f"{WINDOW} x {WINDOW} patches of {PATCH} pixels); e.g. 672 or SAM 3's 1008")
    state = load_checkpoint(checkpoint)
    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    backbone = SAM3VLBackboneTri(scalp=0, visual=mb._create_multiplex_tri_backbone(use_rope_real=rope_real),
                                 text=mb._create_text_encoder(bpe))
    detector = Sam3MultiplexDetector(
        num_feature_levels=1, backbone=backbone, transformer=mb._create_sam3_transformer(),
        segmentation_head=mb._create_segmentation_head(), semantic_segmentation_head=None,
        input_geometry_encoder=mb._create_geometry_encoder(), use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=mb._create_dot_product_scoring(), supervise_joint_box_scores=True, is_multiplex=True)
    prefix = "detector."
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not weights:
        raise ValueError(f"{checkpoint}: no detector weights (keys {prefix}*); a SAM 3.1 checkpoint expected")
    missing, unexpected = detector.load_state_dict(weights, strict=False)
    # the tri-head backbone's tracker heads (interactive, propagation) are the tracker's: not in detector.*, unused
    missing = [k for k in missing if ".interactive_" not in k and ".propagation_" not in k
               and not k.startswith(("backbone.vision_backbone.interactive", "backbone.vision_backbone.propagation"))
               # computed from freqs_cis (which is in the checkpoint) at construction
               and not (rope_real and k.endswith((".freqs_cis_real", ".freqs_cis_imag")))]
    if missing or unexpected:
        raise ValueError(f"{checkpoint}: the detector's weights do not match: missing {missing[:8]} "
                         f"({len(missing)}), unexpected {unexpected[:8]} ({len(unexpected)})")
    if resolution != RESOLUTION:
        resize(detector, resolution, device)
    return detector.to(device).eval(), os.path.realpath(checkpoint)


def resize(detector, resolution: int, device: torch.device) -> None:
    """The detector at another input size than SAM 3's 1008: what depends on the size of the patch grid, recomputed as
    SAM 3 computes it at construction. The ViT's windowed blocks do not depend on it; its 4 global blocks' rotary
    embedding is the checkpoint's for 72 x 72 patches (sam3's ViT: ``img_size`` only sets the rotary embedding), the
    decoder's box position bias keeps the grid's coordinates for torch.compile (``compilable_cord_cache``), the
    neck's sine position encoding its encodings of the 1008 sizes (``cache``)."""
    grid = resolution // PATCH
    vision = detector.backbone.vision_backbone
    for block in vision.trunk.blocks:
        if block.window_size == 0:  # global attention: over the whole grid
            block.attn.input_size = (grid, grid)
            block.attn._setup_rope_freqs()
    decoder = detector.transformer.decoder
    decoder.compilable_cord_cache = decoder._get_coords(grid, grid, device)
    decoder.compilable_stored_size = (grid, grid)
    encoding = vision.position_encoding
    for scale in (4, 2, 1):  # the tri-neck's levels (scale factors 4, 2, 1)
        encoding(torch.zeros(1, 1, grid * scale, grid * scale, device=device))
        encoding.cache[(grid * scale,) * 2] = encoding.cache[(grid * scale,) * 2].clone()


class BoxHead(torch.nn.Module):
    """SAM 3.1's interactive head (the tracker's SAM prompt encoder and mask decoder, as
    ``VideoTrackingMultiplex._build_sam_heads`` and ``build_sam3_multiplex_video_predictor`` build them) on the image
    encoder's interactive neck: a mask for a box. Its weights: the checkpoint's ``tracker.model.interactive_sam_*`` and
    ``interactivity_no_mem_embed`` (added to the image's features on frames without memory, as the tracker does)."""

    def __init__(self, state: dict, resolution: int):
        from sam3.sam.mask_decoder import MaskDecoder  # noqa: PLC0415
        from sam3.sam.prompt_encoder import PromptEncoder  # noqa: PLC0415
        from sam3.sam.transformer import TwoWayTransformer  # noqa: PLC0415

        super().__init__()
        grid = resolution // PATCH
        self.resolution = resolution
        self.prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(grid, grid),
                                            input_image_size=(resolution, resolution), mask_in_chans=16)
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=True,
            iou_prediction_use_sigmoid=False, pred_obj_scores=True, pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True, dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98)
        for module, prefix in ((self.prompt_encoder, "tracker.model.interactive_sam_prompt_encoder."),
                               (self.mask_decoder, "tracker.model.interactive_sam_mask_decoder.")):
            weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not weights:
                raise ValueError(f"no weights {prefix}*: a SAM 3.1 checkpoint expected")
            module.load_state_dict(weights, strict=True)
        self.register_buffer("no_mem_embed", state["tracker.model.interactivity_no_mem_embed"].reshape(1, -1, 1, 1))

    def forward(self, fpn: list[torch.Tensor], boxes: torch.Tensor):
        """The masks' logits (M, 4g, 4g) at a quarter of the input's size, the estimated IoU (M,) and object score
        logits (M,) of ``boxes`` (M, 4: x1 y1 x2 y2 in pixels of the input) on the interactive neck's 3 levels."""
        high = [self.mask_decoder.conv_s0(fpn[0]), self.mask_decoder.conv_s1(fpn[1])]
        corners = boxes.reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device).expand(len(corners), 2)  # SAM's box
        sparse, dense = self.prompt_encoder(points=(corners, labels), boxes=None, masks=None)
        low, iou, _, objects = self.mask_decoder(
            image_embeddings=fpn[2] + self.no_mem_embed, image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False, repeat_image=True,
            high_res_features=high)
        return low[:, 0], iou[:, 0], objects[:, 0]


class Segmenter:
    """Instance masks of ``prompts`` (``Prompt``s, or words) in RGB frames (``segment``), each instance scored; those of
    at least ``threshold`` are kept. ``set_prompts`` changes them. ``dtype``: the precision the networks run in
    (autocast; bfloat16 by default, SAM 3's own choice). ``compile``: torch.compile's mode for the image encoder and the
    grounding transformer (None: not compiled); the first frames then take long (compilation; ``warm_up``, after which
    new prompts compile nothing). ``resolution``: the size frames are scaled
    to, SAM 3's 1008 or a smaller multiple of 336 (672: half the time a frame; ``build_detector``). ``box_head``: also
    segment given boxes (``segment(image, boxes)``; ``BoxHead``)."""

    def __init__(self, prompts: list[Prompt | str], threshold: float = 0.5, device: str | torch.device = "cuda",
                 dtype: torch.dtype | None = torch.bfloat16, checkpoint: str | os.PathLike | None = None,
                 compile: str | None = None, resolution: int = RESOLUTION, box_head: bool = False):
        self.threshold = threshold
        self.device = torch.device(device)
        self.dtype = dtype
        self.compile = compile
        self.resolution = resolution
        if checkpoint is None:  # SAM 3.1's, in the node's cache directory (the node passes its --weights-dir's)
            checkpoint = STORE.file(needed()[0])
        self.model, self.checkpoint = build_detector(checkpoint, self.device, rope_real=compile is not None,
                                                     resolution=resolution)
        self.box_head = BoxHead(load_checkpoint(checkpoint), resolution).to(self.device).eval() if box_head else None
        if dtype is not None:
            # the image encoder's weights in the autocast precision once, not cast at every frame (2 ms of 23); its
            # parameters alone: Module.to would also cast the rotary embedding's complex buffers (dropping their
            # imaginary part). The rest keeps float32: the decoder's feed-forward layers turn autocast off.
            for parameter in self.model.backbone.vision_backbone.parameters():
                parameter.data = parameter.data.to(dtype)
        if compile is not None:
            m = self.model
            m.backbone.vision_backbone = torch.compile(m.backbone.vision_backbone, mode=compile, dynamic=False)
            # the grounding transformer for any number and length of prompts (set_prompts at run time; statically, a
            # new number cost 2 s at the next frame, a new number of exemplar boxes 20 to 27 s): one graph for 1
            # prompt, one for more (warm_up compiles both)
            m.transformer.encoder = torch.compile(m.transformer.encoder, mode=compile, dynamic=True)
            m.transformer.decoder = torch.compile(m.transformer.decoder, mode=compile, dynamic=True)
        # the mask head for as many prompts as have detections: compiled once for any number (dynamic)
        self._mask_embeddings_fn = (self._mask_embeddings if compile is None
                                    else torch.compile(self._mask_embeddings, mode=compile, dynamic=True))
        self._mean = torch.tensor(0.5, device=self.device)
        self.set_prompts(prompts)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None)

    @property
    def words(self) -> list[str]:
        """The prompts' labels (their texts for words)."""
        return [p.name for p in self.prompts]

    @torch.inference_mode()
    def set_prompts(self, prompts: list[Prompt | str], image: torch.Tensor | None = None) -> None:
        """Segments ``prompts`` from now on (strings: words). Exemplar boxes are encoded on ``image``, the (H, W, 3)
        uint8 RGB frame they were drawn in (required with them): the geometry encoder pools the image's features in the
        box and attends to the image, so its encoding describes the boxed instance; later frames reuse it, as they do
        the text's. Encoding them again on a later frame would pool whatever is now in the box. Invalid prompts raise
        ValueError and leave the prompts as they were."""
        from sam3.model.data_misc import FindStage  # noqa: PLC0415
        from sam3.model.geometry_encoders import Prompt as Geometry  # noqa: PLC0415

        prompts = [p if isinstance(p, Prompt) else Prompt(text=p) for p in prompts]
        names = [p.name for p in prompts]
        if len(set(names)) != len(names):
            raise ValueError(f"prompts named twice: {names}")
        exemplars = [i for i, p in enumerate(prompts) if p.exemplars]
        if exemplars and image is None:
            raise ValueError(f"exemplar boxes ({', '.join(names[i] for i in exemplars)}) need the frame they are in")
        if exemplars:
            h, w = image.shape[:2]
            for i in exemplars:
                for box in (*prompts[i].boxes, *prompts[i].negative):
                    if min(box[0], box[1]) < 0 or box[2] > w or box[3] > h:
                        raise ValueError(f"prompt {names[i]!r}: box {box} not inside the frame ({w} x {h})")
        n, d = len(prompts), self.device
        self.prompts = prompts
        if not n:
            return
        with self._autocast():
            self._text = self.model.backbone.forward_text([p.text or VISUAL for p in prompts], device=d)
        # every prompt against the one image: prompt i is text i, all of image 0
        self._find = FindStage(img_ids=torch.zeros(n, device=d, dtype=torch.long),
                               text_ids=torch.arange(n, device=d, dtype=torch.long),
                               input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                               input_points=None, input_points_mask=None)
        # SAM 3's geometric prompt without boxes, for its own forward pass (tests)
        self._prompt = Geometry(box_embeddings=torch.zeros(0, n, 4, device=d),
                                box_mask=torch.zeros(n, 0, device=d, dtype=torch.bool))
        # the same for one prompt: no boxes is the same geometric prompt for every word (_encode_prompt)
        self._empty = Geometry(box_embeddings=torch.zeros(0, 1, 4, device=d),
                               box_mask=torch.zeros(1, 0, device=d, dtype=torch.bool))
        self._geo = self._geo_mask = self._fill = None
        if not exemplars:
            return
        k = max(len(prompts[i].boxes) + len(prompts[i].negative) for i in exemplars)
        boxes = torch.zeros(k, len(exemplars), 4, device=d)  # SAM 3's: cx, cy, w, h of the frame's size
        labels = torch.zeros(k, len(exemplars), device=d, dtype=torch.long)  # 1 positive, 0 negative
        padding = torch.ones(len(exemplars), k, device=d, dtype=torch.bool)
        for j, i in enumerate(exemplars):
            given = [(b, 1) for b in prompts[i].boxes] + [(b, 0) for b in prompts[i].negative]
            for slot, ((x1, y1, x2, y2), positive) in enumerate(given):
                boxes[slot, j] = torch.tensor([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
                labels[slot, j], padding[j, slot] = positive, False
        with self._autocast():
            backbone_out = self._forward_image(self.preprocess(image))
            _, feats, pos, sizes = self.model._get_img_feats(backbone_out, self._find.img_ids[:len(exemplars)])
            geo, geo_mask = self.model.geometry_encoder(
                geo_prompt=Geometry(box_embeddings=boxes, box_mask=padding, box_labels=labels), img_feats=feats,
                img_sizes=sizes, img_pos_embeds=pos)
        # the geometric part of every prompt, (k + 1, n, C): the exemplar prompts' encodings, padded; the words' one
        # token (the empty prompt's, per frame) first, the rest padding
        self._geo = geo.new_zeros(geo.shape[0], n, geo.shape[2])
        self._geo_mask = torch.ones(n, geo.shape[0], device=d, dtype=torch.bool)
        self._fill = torch.zeros(geo.shape[0], n, 1, device=d, dtype=torch.bool)
        self._geo[:, exemplars], self._geo_mask[exemplars] = geo, geo_mask
        self._words = [i for i in range(n) if i not in exemplars]
        self._geo_mask[self._words, 0] = False
        self._fill[0, self._words] = True

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """An (H, W, 3) uint8 RGB frame as SAM 3's input: (1, 3, resolution, resolution), scaled to [-1, 1]."""
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"an (H, W, 3) uint8 RGB image expected, got {tuple(image.shape)} {image.dtype}")
        x = image.to(self.device, non_blocking=True).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(self.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True)
        return (x / 255.0 - self._mean) / self._mean

    def _forward_image(self, x: torch.Tensor):
        """The image encoder; with the box head also its interactive neck (always the same outputs: one compiled
        graph)."""
        return self.model.backbone.forward_image(x, need_interactive_out=self.box_head is not None,
                                                 need_propagation_out=False)

    def _detect(self, backbone_out):
        """Grounding encoder and decoder (Sam3Image.forward_grounding without its mask head) on the image encoder's
        output: the backbone's output, the encoder's output, the prompt and the decoder's queries and scores, all
        prompts."""
        m = self.model
        backbone_out = {**backbone_out, **self._text}
        prompt, prompt_mask = self._encode_prompt(backbone_out)
        backbone_out, encoder_out, _ = m._run_encoder(backbone_out, self._find, prompt, prompt_mask)
        out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
        out, hs = m._run_decoder(memory=encoder_out["encoder_hidden_states"], pos_embed=encoder_out["pos_embed"],
                                 src_mask=encoder_out["padding_mask"], out=out, prompt=prompt, prompt_mask=prompt_mask,
                                 encoder_out=encoder_out)
        return backbone_out, encoder_out["encoder_hidden_states"], prompt, prompt_mask, hs, out

    def _encode_prompt(self, backbone_out) -> tuple[torch.Tensor, torch.Tensor]:
        """Sam3Image._encode_prompt: each prompt is its text's features and its geometric prompt's encoding. A word's
        geometric prompt is empty; its encoding (a token of its own, which attends to the image) is the same for
        every word (one image, no boxes): computed once a frame, not once per word as SAM 3 does (compiled: 1 ms less
        with 8 words at 672, 4 ms with 16 at 1008). Exemplar prompts keep the encoding of their frame
        (``set_prompts``)."""
        m = self.model
        n = len(self.prompts)
        text = backbone_out["language_features"][:, self._find.text_ids]
        text_mask = backbone_out["language_mask"][self._find.text_ids]
        if self._geo is not None and not self._words:  # exemplar prompts alone
            return torch.cat([text, self._geo], dim=0), torch.cat([text_mask, self._geo_mask], dim=1)
        _, feats, pos, sizes = m._get_img_feats(backbone_out, self._find.img_ids[:1])
        geo, geo_mask = m.geometry_encoder(geo_prompt=self._empty, img_feats=feats, img_sizes=sizes,
                                           img_pos_embeds=pos)
        if self._geo is None:  # words alone
            return (torch.cat([text, geo.expand(-1, n, -1)], dim=0),
                    torch.cat([text_mask, geo_mask.expand(n, -1)], dim=1))
        return (torch.cat([text, torch.where(self._fill, geo.to(self._geo.dtype), self._geo)], dim=0),
                torch.cat([text_mask, self._geo_mask], dim=1))

    def _mask_embeddings(self, fpn: list[torch.Tensor], states, prompt, prompt_mask) -> torch.Tensor:
        """SAM 3's mask head (UniversalSegmentationHead.forward) up to its per-pixel embeddings (B, C, 288, 288), for
        the B prompts given (their encoder states, prompts); compiled with ``compile``, for any B."""
        head = self.model.segmentation_head
        if head.cross_attend_prompt is not None:
            states = states + head.cross_attend_prompt(query=head.cross_attn_norm(states), key=prompt, value=prompt,
                                                       key_padding_mask=prompt_mask)[0]
        image_ids = torch.zeros(states.shape[1], device=states.device, dtype=torch.long)  # one image: unused
        pixels = head._embed_pixels(backbone_feats=fpn, image_ids=image_ids, encoder_hidden_states=states)
        return head.instance_seg_head(pixels)

    def _mask_logits(self, backbone_out, encoder_states, prompt, prompt_mask, hs, word, query) -> torch.Tensor:
        """The mask logits (N, 288, 288) of the detections (word, query) alone: the mask head for the prompts with a
        detection, its mask predictor for the detections' queries; SAM 3 runs it for every prompt and all of its 200
        queries (most of the frame's time with many prompts)."""
        from sam3.model.data_misc import NestedTensor  # noqa: PLC0415

        words, inverse = torch.unique(word, return_inverse=True)
        fpn = [x.tensors if isinstance(x, NestedTensor) else x for x in backbone_out["backbone_fpn"]]
        embeds = self._mask_embeddings_fn(fpn, encoder_states[:, words], prompt[:, words], prompt_mask[words])
        mask_embed = self.model.segmentation_head.mask_predictor.mask_embed
        queries = mask_embed(hs[-1][word, query])  # (N, C); hs: (layers, prompts, queries, C)
        return torch.einsum("nc,nchw->nhw", queries, embeds[inverse])

    def _box_logits(self, backbone_out, boxes: torch.Tensor, h: int, w: int):
        """The box head on the interactive neck for ``boxes`` (M, 4) in pixels of the (h, w) frame."""
        from sam3.model.data_misc import NestedTensor  # noqa: PLC0415

        fpn = [x.tensors if isinstance(x, NestedTensor) else x for x in backbone_out["interactive"]["backbone_fpn"]]
        scale = torch.tensor([self.resolution / w, self.resolution / h] * 2, device=self.device)
        return self.box_head(fpn, boxes.to(self.device, torch.float32) * scale)

    def warm_up(self, height: int, width: int, frames: int = 3) -> None:
        """Runs a few frames of the given size: compiles (with ``compile``) and lets CUDA pick its kernels; with the box
        head, a box each frame. Then the grounding transformer and the mask head for one prompt and for two, whatever
        the prompts now (torch.compile's dynamic shapes treat a size of 1 apart: else the first frame with one prompt,
        or with detections of one, compiles, up to 30 s); the prompts stay as they were. A blank frame has no
        detections, so the mask head runs on its own."""
        image = torch.zeros(height, width, 3, dtype=torch.uint8, device=self.device)
        boxes = (torch.tensor([[0.0, 0.0, width / 2, height / 2]], device=self.device)
                 if self.box_head is not None else None)
        for _ in range(frames):
            self.segment(image, boxes)
        kept = {k: v for k, v in vars(self).items() if k in PROMPT_STATE}
        try:
            for n in (1, 2):
                self.set_prompts(list(WARM_UP_WORDS[:n]))
                self.segment(image)
                with torch.inference_mode(), self._autocast():
                    for m in sorted({1, n}):
                        if self.compile == "reduce-overhead":
                            torch.compiler.cudagraph_mark_step_begin()
                        detected = self._detect(self._forward_image(self.preprocess(image)))[:5]
                        word = torch.arange(m, device=self.device)
                        self._mask_logits(*detected, word, torch.zeros_like(word))
        finally:
            vars(self).update(kept)
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def segment(self, image: torch.Tensor, boxes: torch.Tensor | None = None) -> Segmentation:
        """The instances of the prompts in an (H, W, 3) uint8 RGB frame (host or device memory); with ``boxes`` (M, 4:
        x1 y1 x2 y2 in pixels; the box head) also each box's mask."""
        if boxes is not None and self.box_head is None:
            raise ValueError("boxes to segment, but the segmenter has no box head (box_head=True)")
        h, w = image.shape[:2]
        if self.compile == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()  # a new frame: CUDA graphs may reuse the last frame's memory
        if not self.prompts and boxes is None:  # nothing to segment: the networks not run
            return Segmentation(torch.zeros(0, dtype=torch.long, device=self.device),
                                torch.zeros(0, device=self.device), torch.zeros(0, 4, device=self.device),
                                torch.zeros(0, h, w, dtype=torch.bool, device=self.device),
                                torch.zeros(h, w, dtype=torch.uint16, device=self.device))
        with self._autocast():
            backbone_out = self._forward_image(self.preprocess(image))
            detected = self._detect(backbone_out) if self.prompts else None
        if detected is not None:
            backbone_out, states, prompt, prompt_mask, hs, out = detected
            # (prompts, queries): the score already includes the decoder's presence score (supervise_joint_box_scores)
            score = out["pred_logits"].squeeze(-1).float().sigmoid()
            word, query = torch.nonzero(score >= self.threshold, as_tuple=True)  # waits for the GPU
            score = score[word, query]
            order = torch.argsort(score, descending=True)
            word, query, score = word[order], query[order], score[order]
            found = out["pred_boxes_xyxy"][word, query].float() * torch.tensor([w, h, w, h], device=self.device)
        else:
            word = torch.zeros(0, dtype=torch.long, device=self.device)
            score, found = torch.zeros(0, device=self.device), torch.zeros(0, 4, device=self.device)
        if len(score):
            with self._autocast():
                logits = self._mask_logits(backbone_out, states, prompt, prompt_mask, hs, word, query).float()
            masks = F.interpolate(logits[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0] > 0
            # the best instance where masks overlap: the first (best) of the covering masks
            first = torch.argmax(masks.to(torch.uint8), dim=0)
            instances = torch.where(masks.any(dim=0), first + 1, 0).to(torch.uint16)
        else:
            masks = torch.zeros(0, h, w, dtype=torch.bool, device=self.device)
            instances = torch.zeros(h, w, dtype=torch.uint16, device=self.device)
        result = Segmentation(word, score, found, masks, instances)
        if boxes is not None:
            if len(boxes):
                with self._autocast():
                    logits, iou, objects = self._box_logits(backbone_out, boxes, h, w)
                result.box_masks = F.interpolate(logits.float()[:, None], size=(h, w), mode="bilinear",
                                                 align_corners=False)[:, 0] > 0
                result.box_scores = iou.float()
                result.box_found = (objects > 0) & result.box_masks.flatten(1).any(dim=1)
            else:
                result.box_masks = torch.zeros(0, h, w, dtype=torch.bool, device=self.device)
                result.box_scores = torch.zeros(0, device=self.device)
                result.box_found = torch.zeros(0, dtype=torch.bool, device=self.device)
        return result
