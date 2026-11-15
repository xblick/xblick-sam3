"""The node with SAM 3.1's segmentation of a frame made slower by SLOW_S seconds (its first argument), for
tests/test_node.py: its queryable answers while the model works. It prints "slow step" when one begins."""

import sys
import time

from xblick_sam3 import node, segmenter

SLOW_S = float(sys.argv.pop(1))
segment = segmenter.Segmenter.segment


def slow_segment(self, *args, **kwargs):
    print("slow step", flush=True)
    time.sleep(SLOW_S)
    return segment(self, *args, **kwargs)


segmenter.Segmenter.segment = slow_segment
sys.exit(node.main())
