# Performance

This page states how fast xblick-sam3 is, how we measured it, which changes made it fast, and which changes did
not help. The [README](../README.md) has the short version.

## Result

With 8 words, the node needs 24.4 ms per frame at its default settings. The same segmenter needs 69.9 ms at SAM 3's
own resolution without compilation.

| Words | 1008, eager | 1008, `--compile default` | 672, eager | 672, `--compile default` | Instances per frame |
| --- | --- | --- | --- | --- | --- |
| 1 | 43.8 ms | 29.1 ms | 28.1 ms | 16.2 ms | 1.0 |
| 8 | 69.9 ms | 51.1 ms | 36.7 ms | 24.4 ms | 12.2 |
| 16 | 96.7 ms | 71.2 ms | 47.0 ms | 31.7 ms | 17.0 |

Setup: one RTX PRO 6000 Blackwell, 50 frames of TUM RGB-D (640 x 480), median, measured on 2026-10-31 with
`tools/bench.py`. The time runs from the frame in host memory to the masks on the GPU.

Two terms in the table need an explanation.

- *Eager* is PyTorch's normal mode. Python executes the model operation by operation.
- *`torch.compile`* translates a PyTorch model into optimised code. The option `--compile default` turns it on.

The segmenter scales every frame to the input size, which is 1008 x 1008 for SAM 3 and 672 x 672 for the node. The
words are the first 1, 8 or 16 of this list: `box can bottle cup chair table person keyboard monitor mouse book
plant phone bag lamp door`.

We measured the compiled columns after the change "The empty geometric prompt, encoded once" below. We measured the
eager columns before it. With that change, the eager time with 8 words is about 1 ms lower.

To repeat the measurement on a directory of images:

```bash
.venv/bin/python tools/bench.py IMAGES_DIR --words cup bottle --compile default --resolution 672
```

## Where the time goes

SAM 3.1's detector has four parts. The *image encoder* is a vision transformer (ViT-L, 32 blocks) that turns the
frame into one feature vector per 14 x 14 pixel patch. We call such a vector a *token*. The *grounding encoder*
fuses each word with all image tokens. The *decoder* turns 200 queries per word into boxes and scores. The *mask
head* computes a mask per query.

| Part, at 1008, compiled | Time |
| --- | --- |
| Image encoder, once per frame, on 72 x 72 patches | 21 ms |
| Grounding encoder, 6 layers over the 5184 image tokens | 1.5 ms per word |
| Decoder, 200 queries | 0.5 ms per word |
| All work per word, with the masks | about 3 ms per word |

At 672 the grid has 48 x 48 patches, which are 2304 tokens. The image encoder then takes less than half the time.
The attention of the grounding encoder over the tokens takes a fifth.

The GPU is idle for 4.5 ms of each frame. At 672 with 8 words, it is busy for 22 ms of 27 ms. A *kernel* is one
function that the GPU executes, for example one matrix product. A frame starts about 1700 kernels, and the CPU
needs time to start each of them. The segmenter also waits for the scores before it starts the mask head.

## What we changed

SAM 3 comes with a class `Sam3Processor` that runs the model. Our `Segmenter` replaces it and differs in six
points.

### 1. All words in one pass

The segmenter encodes the words once, at start. For each frame it calls the image encoder once. It then calls the
grounding encoder and the decoder once for all words, as one batch. SAM 3's processor calls them once per word.

### 2. Masks only for the detections

SAM 3's mask head computes masks for all 200 queries of every word. The segmenter reads the scores first. It then
calls the mask head only for the words that have a detection, and only for the queries of these detections. In
eager mode with 8 words, this saves 11 ms.

`tests/test_segmenter.py` checks that these masks equal the masks of SAM 3's own forward pass. The intersection
over union must be above 0.99.

### 3. The empty geometric prompt, encoded once

Every prompt consists of a word and SAM 3's encoding of the *geometric prompt*, which holds the boxes. The
geometric prompt has a token of its own that attends to the image. This token is present even when the prompt has
no boxes, because SAM 3 was trained with it.

For one image and no boxes, the encoding is the same for every word. SAM 3 computes it once per word. The segmenter
computes it once per frame.

| Setting, compiled | Saving | Frame time |
| --- | --- | --- |
| 8 words at 672 | 1 ms | 25 ms |
| 16 words at 1008 | 4 ms | 75 ms |

### 4. The weights of the image encoder in bfloat16

*bfloat16* is a 16-bit number format that modern GPUs compute much faster than 32-bit floats. PyTorch's *autocast*
converts values to it on the fly, inside a block of code. With autocast alone, the image encoder converted its
32-bit weights in every frame. The segmenter converts them once at start, which saves 2 ms per frame.

The segmenter converts only the parameters. The image encoder also holds its rotary position embedding as buffers
of complex numbers. A conversion of the whole module would convert these buffers and drop their imaginary part.
SAM 3 then finds nothing and reports no error.

The other networks keep 32-bit weights and work under autocast, as SAM 3 does. SAM 3's decoder turns autocast off
on purpose.

### 5. Compilation

`--compile` applies `torch.compile` to the image encoder, the grounding transformer and the mask head. The
compilation takes about 45 s before the first frame. After it, a frame takes about a third less time.

The mode `default` is the one to use.

| Mode | Result |
| --- | --- |
| `default` | 36.7 ms become 24.4 ms with 8 words at 672. |
| `max-autotune` | The compilation takes 3 minutes and gains nothing over `default`. |
| `reduce-overhead` | It gains 1 ms of 50 ms with 8 words at 1008, and 1 ms of 24 ms at 672. |

`reduce-overhead` uses CUDA graphs. A *CUDA graph* records a fixed sequence of kernels and starts it with one call.

### 6. The resolution 672

The node scales frames to 672 x 672, and you change this with `--resolution`. The lower resolution costs some of
SAM 3's detail. We measured how much on 2026-10-31, with 8 words, on two sets of 40 frames.

| | TUM RGB-D, 640 x 480 | Simulated head camera, 1280 x 720 |
| --- | --- | --- |
| Instances that are sure at 1008 and found again at 672 | 99 % | 98 % |
| Mask IoU between 672 and 1008 | 0.93 | 0.97 |
| Found again when both sides use the threshold 0.5 | 91 % | 90 % |

An instance is "sure" when its score at 1008 is at least 0.6. The node "finds it again" at 672 when the same word
has an instance with a score of at least 0.4 and a mask IoU of at least 0.5.

Most of the difference at the threshold 0.5 comes from instances whose score is close to the threshold. Such an
instance is present in one frame and absent in the next, at 1008 as well.
[docs/topics.md](topics.md#how-stable-the-ids-are) shows the effect on the ids.

These numbers measure agreement with 1008. We did not measure accuracy against ground truth.

Only multiples of 336 are possible. The ViT computes attention in windows of 24 x 24 patches, and a window must be
whole. The sizes 840 and 756 would need padded windows. We tried 756, and it agreed less with 1008 than 672 does.

`segmenter.resize` recomputes everything that depends on the grid, as SAM 3 does at construction:

- the rotary embedding of the 4 global attention blocks,
- the coordinates of the decoder's box position bias,
- the position encodings of the neck.

## Compiling for any number of prompts

A compiled network is normally fixed to the tensor shapes that it saw. A new number of prompts would then start a
new compilation in the middle of the stream. The segmenter therefore compiles the grounding transformer and the
mask head with `dynamic=True`, which makes the compiled code valid for any number and length of prompts.
`Segmenter.warm_up` compiles both graphs that PyTorch needs, one for a single prompt and one for several.

| | Static shapes | `dynamic=True` |
| --- | --- | --- |
| A new number of prompts | 2 s pause at the next frame | no pause |
| A new number of exemplar boxes | 20 to 27 s pause | no pause |
| Warm-up | about 40 s | 45 s |
| Frame time, 8 words at 672 | 25.4 ms | 25.4 ms |
| Frame time, 1 word at 672 | 15.4 ms | 17.0 ms |

The two times for 1 word come from two different GPUs of the same kind, so 1.6 ms is an upper bound of the cost.

## The cost of `--boxes`

Setup: TUM RGB-D at 672, median of 90 frames, measured in the same way as `tools/bench.py` does.

| Step | Compiled | Eager |
| --- | --- | --- |
| The interactive neck, with 8 words | 1.5 ms (26.9 against 25.4 ms) | within the noise |
| 1 to 8 boxes in addition | 2.6 ms | 2.6 ms |
| 32 boxes in addition | 4.9 ms | 6.5 ms |
| Boxes alone, no prompts, 1 to 8 boxes | 14.1 ms per frame | |

The matching of the ids takes 0.4 ms per frame.

## With other nodes on one GPU

A camera node rarely has a GPU to itself. We measured this node beside the robot mask node and nvblox with
`tools/bench_nodes.py --sam3-dir` of xblick-mapping.

Setup: the head camera of a simulation, replayed at 30 frames per second (1280 x 720, colour and depth). nvblox
uses the mask. This node uses the words `box,can,bottle,cup` and `--compile default`. All nodes share one
RTX PRO 6000 under MPS. *MPS* is NVIDIA's service that lets several processes use one GPU at the same time. Each
measurement lasts 60 s after 30 s of warm-up. Dates: 2026-10-30, and 2026-10-31 for the rows at 672.

The cells state frames per second and the median latency from the stamp of the frame to the results. For SAM 3.1,
the cell also states the time per frame.

**Without Boxer.** We left Boxer out with `--no-boxer`, because its GPU time would slow SAM 3.1.

| | Robot mask | nvblox | SAM 3.1 | GPU load |
| --- | --- | --- | --- | --- |
| Without SAM 3.1 | 30/s, 21 ms | 29/s, 34 ms | | 4 % |
| SAM 3.1 at 1008 | 30/s, 22 ms | 29/s, 36 ms | 22.7/s, 44 ms per frame, 60 ms | 95 % |
| SAM 3.1 at 672 | 30/s, 21 ms | 29/s, 35 ms | 30/s, 24 ms per frame, 24 ms | 63 % |

At 1008, SAM 3.1 takes the newest frame as soon as it has finished the last one. It fills the spare time of the
GPU, and the other nodes still keep up with the camera at the same latency. At 672, SAM 3.1 keeps up with the
camera as well and leaves a third of the GPU free.

**All nodes with Boxer.** nvblox runs with its `--features`.

| | Robot mask | Boxer | nvblox | SAM 3.1 | GPU load |
| --- | --- | --- | --- | --- | --- |
| Without SAM 3.1 | 30/s, 22 ms | 30/s, 32 ms | 29/s, 35 ms | | 42 % |
| SAM 3.1 at 1008 | 30/s, 22 ms | 30/s, 87 ms | 29/s, 35 ms | 18.5/s, 54 ms per frame, 70 ms | 98 % |
| SAM 3.1 at 672 | 30/s, 21 ms | 30/s, 34 ms | 29/s, 35 ms | 29.7/s, 29 ms per frame, 29 ms | 87 % |
| SAM 3.1 at 672 with `--boxes` on Boxer's boxes | 30/s, 21 ms | 30/s, 37 ms | 29/s, 35 ms | 28.5/s, 34 ms per frame, 75 ms | 93 % |

At 672 every node keeps up with the camera. Boxer's results arrive about as early as without SAM 3.1.

The row with `--boxes` also reads Boxer's boxes through `--boxes`. It had 1.0 box and
3.8 instances per frame, and no box message without its frame. The latency of SAM 3.1 includes Boxer's latency in
this row, because the node segments a frame only after Boxer's boxes for it arrive.

**Without MPS.** The GPU then gives each process a time slice in turn. We measured this at 1008. Boxer fell to 16
frames per second and nvblox to 19. Run the nodes under MPS.
[Sharing a GPU](https://github.com/xblick/xblick/blob/main/docs/guides/sharing-a-gpu.md)
shows how.

## What did not help

| Change | Result |
| --- | --- |
| FP8 matrix products in the image encoder | No faster, with 3.7 % error. |
| A token-major layout for the attention inputs of the grounding encoder | 4 ms less in eager mode, 8 ms more compiled. |
| `max-autotune` | 3 minutes of compilation, no gain. |
| `reduce-overhead` (CUDA graphs) | 1 ms of 50 ms. |

**FP8.** FP8 is an 8-bit float format. The bfloat16 matrix products of the image encoder already reach 310 to 380
TFLOPS. FP8 with one scale per row was no faster once it also quantised the activations.

**The layout of the attention inputs.** SAM 3 hands the image tokens to the attention projections as views of the
shape (tokens, words, channels). PyTorch multiplies such a view as 5184 small products with a slow kernel. This
costs 5.5 ms per frame with 8 words. A copy into a better layout saves 4 ms in eager mode. In compiled mode, the
copy splits the compiled graphs and costs 8 ms. The segmenter therefore keeps SAM 3's layout.

## Status and limits

- We did not try to fuse all words into one prompt. It would need one grounding pass for all words. It also asks
  the model a different question, and we do not know the quality of the answer.
- We did not measure accuracy against ground truth at any resolution.
- We measured the node without MPS only at 1008.
- SAM 3.1's own video tracker is slower than the node's detector with its id matching.
  [docs/design.md](design.md#why-the-node-does-not-use-sam-31s-tracker) has the numbers.
