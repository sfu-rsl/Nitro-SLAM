#!/usr/bin/env python3
"""ORB-SLAM3 vs Nitro-SLAM Local Mapping breakdown, side by side per sequence.

The Local Mapping counterpart of tracking_comparison.py: same grouping, fills and
styling, with the phases from localmapping_breakdown.py. One group per sequence, a
stacked bar per system, solid for ORB-SLAM3 and striped for Nitro-SLAM, over the
sequences in SEQUENCES.

Usage:
    ./localmapping_comparison.py
    ./localmapping_comparison.py --out figures --sequences MH01 room1
"""

import sys

from localmapping_breakdown import CHART as BREAKDOWN
from tracking_breakdown import SEQUENCES
from tracking_comparison import run

CHART = dict(BREAKDOWN, out='localmapping_comparison.png')


if __name__ == '__main__':
    sys.exit(run(CHART, SEQUENCES, __doc__))
