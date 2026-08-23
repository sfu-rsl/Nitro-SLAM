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

Usage:
    ./loopclosing_breakdown.py
    ./loopclosing_breakdown.py --sequences corridor1 MH05 --out figures
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from tracking_breakdown import (MACHINE, RESULTS_ROOTS, SURFACE, SYSTEM,
                                aggregate, figure, find_runs, write_csv)

SUBDIR = ('LoopClosing', 'data')

# Loop closing needs sequences that actually close a loop, which the four sequences
# the other figures use largely do not: of those, only corridor1 ever closes one,
# in 2 of its 5 runs. These four do, in 5/5 runs each except corridor1.
CLOSING_SEQUENCES = ['room3', 'corridor1', 'magistrale2', 'outdoors5']

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


def main():
    sns.set_context("paper")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', nargs='*', default=RESULTS_ROOTS,
                    help='results roots to search, in order')
    ap.add_argument('--sequences', nargs='*', default=CLOSING_SEQUENCES,
                    help='sequences to plot, in this order')
    ap.add_argument('--system', default=SYSTEM, help='system directory to read')
    ap.add_argument('--machine', default=MACHINE, help='machine directory to read')
    ap.add_argument('--out', default='.', help='directory for the figure and CSV')
    ap.add_argument('--allow-missing-series', action='store_true',
                    help='keep runs whose stats dump is incomplete, counting the '
                         'absent phases as zero instead of dropping the run')
    args = ap.parse_args()

    runs, missing = find_runs(args.results, args.sequences, args.system, args.machine,
                              subdir=SUBDIR)
    for d in missing:
        print(f'  warning: "{d}" not found under any of {", ".join(args.results)}')
    if not runs:
        print(f'no runs found for {args.system}/{args.machine}')
        return 1

    n_expected = max(len({r['iteration'] for r in runs if r['dataset'] == d})
                     for d in {r['dataset'] for r in runs})
    print(f'{len(runs)} run directories · {args.system} · {args.machine} · '
          f'{", ".join(args.sequences)}')
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})

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
        return 1

    path = os.path.join(args.out, CHART['out'])
    if figure(agg, CHART, CHART['phases'], args.sequences, path, n_expected, None):
        print(f'  wrote {path}')
    csv_path = os.path.join(args.out, 'loopclosing_breakdown.csv')
    write_csv(csv_path, {'loopclosing': agg})
    print(f'  wrote {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
