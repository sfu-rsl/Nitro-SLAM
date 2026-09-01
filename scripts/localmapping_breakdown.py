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

Both platforms are drawn by default, into one directory with the platform in the
filename -- analysis_out/localmapping_breakdown_{desktop,jetson}.png.

Usage:
    ./localmapping_breakdown.py
    ./localmapping_breakdown.py --platform jetson --sequences MH01 room1
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from tracking_breakdown import (OUT_ROOT, PLATFORMS, SEQUENCES, SURFACE, SYSTEM,
                                aggregate, figure, find_runs, out_path,
                                select_platforms, write_csv)

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
    if not agg:
        print('  no Local Mapping data')
        return 0
    # Re-runs in flight show up as skipped iterations, not as zeroed bars.
    for dataset, entries in sorted(skipped.items()):
        for iteration, why in entries:
            print(f'  skipped {dataset}/{iteration} -- {why}')
    if absent:
        print(f'  series absent in some runs: {", ".join(absent)}')

    written = 0
    path = out_path(args.out, CHART['out'], platform)
    if figure(agg, CHART, CHART['phases'], args.sequences, path, n_expected, None):
        print(f'  wrote {path}')
        written = 1
    if args.csv:
        csv_path = out_path(args.out, 'localmapping_breakdown.csv', platform)
        write_csv(csv_path, {'localmapping': agg})
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
    ap.add_argument('--sequences', nargs='*', default=SEQUENCES,
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
