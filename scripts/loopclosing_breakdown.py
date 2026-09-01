#!/usr/bin/env python3
"""Stacked-bar time breakdown for the Loop Closing thread -- closures only.

Like localmapping_breakdown.py, this reuses the loading, aggregation and drawing in
tracking_breakdown.py and only names the phases, so all three figures share one set
of styling decisions.

    <results root>/<system>/<machine>/<dataset>/<iteration>/LoopClosing/data/

Draws one figure, loopclosing_breakdown.png. Only the iterations that actually
closed a loop are counted -- rows where loopClosed == 1 -- so a bar is the mean cost
of one closure, not of a place-recognition query that found nothing. A run that
never closed a loop has nothing to contribute and is skipped, and a sequence where
no run closed a loop is left off the axis entirely; both are reported on stdout.

Both platforms are drawn by default, into one directory with the platform in the
filename -- analysis_out/loopclosing_breakdown_{desktop,jetson}.png.

Usage:
    ./loopclosing_breakdown.py
    ./loopclosing_breakdown.py --sequences corridor1 MH05 --platform jetson
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from tracking_breakdown import (OUT_ROOT, PLATFORMS, SURFACE, SYSTEM,
                                aggregate, figure, find_runs, out_path,
                                select_platforms, write_csv)

SUBDIR = ('LoopClosing', 'data')

# Loop closing needs sequences that actually close a loop, which the four the other
# figures use do not -- none of MH01, V101, room1 or corridor1 closes one on both
# systems and both platforms. These four do, in most runs of both systems on the
# desktop and the Jetson alike. outdoors7 is the loosest of them: it closed in 3/5
# ORB-SLAM3 runs and 1/5 Nitro-SLAM runs on the desktop, so its bars rest on far
# less data than the other three -- the run counts are printed on stdout.
CLOSING_SEQUENCES = ['room3', 'magistrale1', 'outdoors5', 'outdoors7']

# (label, [series filenames]) in the order the thread runs them, which is also the
# order the palette assigns hues -- see the note on SLOTS in tracking_breakdown.py.
# Loop fusion and graph optimization are the two halves of loopCorrection, so the
# parent is replaced by its children and "Other" is the rest of the correction.
CHART = dict(
    out='loopclosing_breakdown.png',
    subdir=SUBDIR,
    title='Loop Closing Breakdown',
    total='loopClosing_time',
    unit='Mean Loop Closure Time (ms)',
    only_where='loopClosed',       # count only the iterations that closed a loop
    show_run_count=False,          # how many runs closed a loop is reported on stdout
    empty_reason='no loop closed in this run',
    phases=[
        ('Region Detection',  ['placeRecognition_time']),
        ('Loop Fusion',        ['loopFusion_time']),
        ('Graph Optimization', ['graphOptimization_time']),
    ],
    note='',
    # note='Closures only (loopClosed == 1); global BA runs on its own thread and is '
        #  'not part of the total.',
)


def build(args, platform, roots, machine):
    """Draw the figure for one platform. Returns the number of figures written."""
    runs, missing = find_runs(roots, args.sequences, args.system, machine,
                              subdir=SUBDIR)
    for d in missing:
        print(f'  warning: "{d}" not found under any of {", ".join(roots)}')
    if not runs:
        print(f'  no runs found for {args.system}/{machine}')
        return 0

    n_expected = max(len({r['iteration'] for r in runs if r['dataset'] == d})
                     for d in {r['dataset'] for r in runs})
    print(f'  {len(runs)} run directories · {args.system} · {machine} · '
          f'{", ".join(args.sequences)}')
    os.makedirs(args.out, exist_ok=True)

    agg, skipped, absent = aggregate(runs, CHART, CHART['phases'],
                                     args.allow_missing_series)
    # Closures are rare, so say plainly how much data each bar rests on rather than
    # letting a one-closure average pass for a five-run mean.
    for dataset, entries in sorted(skipped.items()):
        print(f'  {dataset}: {len(entries)} run(s) contributed nothing '
              f'({entries[0][1]})')
    for dataset in args.sequences:
        if dataset in agg:
            e = agg[dataset]
            print(f'  {dataset}: {e["n_runs"]}/{n_expected} runs closed a loop, '
                  f'{e["n_iters"]:.1f} closure(s) per run on average')
    if absent:
        print(f'  series absent in some runs: {", ".join(absent)}')
    if not agg:
        print('  no sequence closed a loop -- nothing to draw')
        return 0

    written = 0
    path = out_path(args.out, CHART['out'], platform)
    if figure(agg, CHART, CHART['phases'], args.sequences, path, n_expected, None):
        print(f'  wrote {path}')
        written = 1
    if args.csv:
        csv_path = out_path(args.out, 'loopclosing_breakdown.csv', platform)
        write_csv(csv_path, {'loopclosing': agg})
        print(f'  wrote {csv_path}')
    return written


def main():
    sns.set_context("paper")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--platform', nargs='*', choices=sorted(PLATFORMS),
                    help='platforms to draw (default: all of them)')
    ap.add_argument('--results', nargs='*',
                    help="results roots to search, in order; overrides --platform's")
    ap.add_argument('--sequences', nargs='*', default=CLOSING_SEQUENCES,
                    help='sequences to plot, in this order')
    ap.add_argument('--system', default=SYSTEM, help='system directory to read')
    ap.add_argument('--machine', help="machine directory to read; overrides --platform's")
    ap.add_argument('--out', default=OUT_ROOT, help='directory to write the figure into')
    ap.add_argument('--csv', action='store_true',
                    help='also write the per-phase table beside the figure')
    ap.add_argument('--allow-missing-series', action='store_true',
                    help='keep runs whose stats dump is incomplete, counting the '
                         'absent phases as zero instead of dropping the run')
    args = ap.parse_args()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})

    # A platform with no results tree is reported and stepped over rather than
    # failing the run: the two sweeps finish at different times.
    written = 0
    for name, roots, machine in select_platforms(args):
        print(f'{name}:')
        written += build(args, name, roots, machine)
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
