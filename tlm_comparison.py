#!/usr/bin/env python3
"""ORB-SLAM3 vs Nitro-SLAM TrackLocalMap breakdown, side by side per sequence.

The TrackLocalMap counterpart of tracking_comparison.py: same grouping, fills and
styling, with the phases tracking_breakdown.py draws in tlm_breakdown.png. One
group per sequence, a stacked bar per system, solid for ORB-SLAM3 and striped for
Nitro-SLAM, over the sequences in SEQUENCES.

Where tracking_comparison.py shows Track Local map as a single band of the
frametime, this expands that band into its three phases, so a change inside
TrackLocalMap is attributable rather than just visible in the total.

Usage:
    ./tlm_comparison.py
    ./tlm_comparison.py --out figures --sequences MH01 room1
"""

import sys

from tracking_breakdown import CHARTS, SEQUENCES
from tracking_comparison import run

CHART = dict(CHARTS['tlm'], out='tlm_comparison.png')


if __name__ == '__main__':
    sys.exit(run(CHART, SEQUENCES, __doc__))
