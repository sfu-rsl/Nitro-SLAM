#!/usr/bin/env python3
"""Stacked-bar time breakdown for the Local Mapping thread, across EuRoC and TUM-VI.

The tracking counterpart of this script, tracking_breakdown.py, owns the loading,
aggregation and drawing; this one just points it at LocalMapping/data and names the
phases. Styling and palette therefore stay in one place -- edit them there and both
figures follow.

    <results root>/<system>/<machine>/<dataset>/<iteration>/LocalMapping/data/

Draws one figure, localmapping_breakdown.png: a bar per sequence, each segment a
phase's mean time per keyframe averaged over the repeated runs, over the four
sequences named in SEQUENCES.

Usage:
    ./localmapping_breakdown.py
    ./localmapping_breakdown.py --out figures --sequences MH01 room1
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from tracking_breakdown import (MACHINE, RESULTS_ROOTS, SEQUENCES, SURFACE, SYSTEM,
                                aggregate, figure, find_runs, write_csv)

SUBDIR = ('LocalMapping', 'data')

# (label, [series filenames]) in the order the thread runs them, which is also the
# order the palette assigns hues -- see the note on SLOTS in tracking_breakdown.py.
# Phases are leaves of localMapping_time, so "Other" is a meaningful residual:
# searchForTriangulation sits inside MP creation and imuInitFIBA inside IMU init,
# so neither is listed alongside its parent.
CHART = dict(
    out='localmapping_breakdown.png',
    subdir=SUBDIR,
    title='Local Mapping Breakdown',
    total='localMapping_time',
    unit='Mean Keyframe Time (ms)',
    phases=[
        ('Process Keyframe', ['processKF_time']),
        ('MP Culling',       ['MPCulling_time']),
        ('MP Creation',      ['MPCreation_time']),
        ('MP Fusion',        ['searchInNeighbors_time']),
        ('Local BA',         ['LBA_time']),
        ('KF Culling',       ['KFCulling_time']),
        # Scale refinement is the tail of the same IMU-initialisation pipeline.
        ('IMU Init',         ['imuInit_time', 'scaleRefinement_time']),
    ],
    note='',
)


def main():
    sns.set_context("paper")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', nargs='*', default=RESULTS_ROOTS,
                    help='results roots to search, in order')
    ap.add_argument('--sequences', nargs='*', default=SEQUENCES,
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
    if not agg:
        print('  no Local Mapping data')
        return 1
    # Re-runs in flight show up as skipped iterations, not as zeroed bars.
    for dataset, entries in sorted(skipped.items()):
        for iteration, why in entries:
            print(f'  skipped {dataset}/{iteration} -- {why}')
    if absent:
        print(f'  series absent in some runs: {", ".join(absent)}')

    path = os.path.join(args.out, CHART['out'])
    if figure(agg, CHART, CHART['phases'], args.sequences, path, n_expected, None):
        print(f'  wrote {path}')
    csv_path = os.path.join(args.out, 'localmapping_breakdown.csv')
    write_csv(csv_path, {'localmapping': agg})
    print(f'  wrote {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
