#!/usr/bin/env python3
"""ORB-SLAM3 vs Nitro-SLAM Loop Closing breakdown, closures only, per sequence.

The Loop Closing counterpart of tracking_comparison.py: same grouping, fills and
styling, with the phases and the loopClosed == 1 filter from
loopclosing_breakdown.py, so a bar is the mean cost of one closure on that system.

Defaults to the sequences that close reliably on both systems and both platforms
rather than the four the other comparisons use; a system that never closed a loop
on a sequence has no bar there, and how many runs each bar rests on is printed
rather than drawn.

Both platforms are drawn by default, into one directory with the platform in the
filename -- analysis_out/loopclosing_comparison_{desktop,jetson}.png.

Usage:
    ./loopclosing_comparison.py
    ./loopclosing_comparison.py --sequences room3 outdoors5 --platform jetson
"""

import sys

from loopclosing_breakdown import CHART as BREAKDOWN, CLOSING_SEQUENCES
from tracking_comparison import run

CHART = dict(BREAKDOWN, out='loopclosing_comparison.png')


if __name__ == '__main__':
    sys.exit(run(CHART, CLOSING_SEQUENCES, __doc__))
