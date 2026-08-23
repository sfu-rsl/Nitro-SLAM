#!/usr/bin/env python3
"""ORB-SLAM3 vs Nitro-SLAM tracking breakdown, side by side per sequence.

Same data and phases as tracking_breakdown.py, drawn as one group per sequence with
a stacked bar per system, so the phase that moved is visible rather than just the
totals. Loading, aggregation, palette and styling all come from tracking_breakdown,
so the comparison stays in step with the single-system figures.

Nitro-SLAM's results carry an extra kernel-status component,

    <results root>/Nitro-SLAM/<kernel status>/<machine>/<dataset>/<iteration>/

which is discovered automatically; --nitro-kernel picks one when a tree holds more
than one.

Usage:
    ./tracking_comparison.py
    ./tracking_comparison.py --out figures --sequences MH01 room1
    ./tracking_comparison.py --nitro-kernel 11111-1111-001111
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import tracking_breakdown as tb
from tracking_breakdown import (INK, INK_MUTED, LEGEND_NCOL, MACHINE, RESULTS_ROOTS,
                                SEQUENCES, SURFACE, aggregate, annotate_total,
                                colors_for, draw_stack, find_runs, header,
                                style_axes, visible_phases)

BASELINE = 'ORB-SLAM3'
CONTENDER = 'Nitro-SLAM'
# Colour carries the phase, so the system is carried by fill instead: the baseline
# is solid and the contender striped. That frees the axis of per-bar labels, which
# were what forced the groups apart.
HATCH = {BASELINE: None, CONTENDER: '//'}
CHART = dict(tb.CHARTS['tracking'], out='tracking_comparison.png')


def resolve_system(roots, system, machine, kernel=None):
    """Path fragment for `system`, with the kernel-status component if it has one.

    ORB-SLAM3 runs sit directly under <root>/<system>/<machine>; Nitro-SLAM inserts
    the kernel status between the two. Returns the fragment to hand to find_runs.
    """
    found = set()
    for root in roots:
        base = os.path.join(root, system)
        if not os.path.isdir(base):
            continue
        if os.path.isdir(os.path.join(base, machine)):
            found.add(system)
            continue
        for entry in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, entry, machine)):
                found.add(os.path.join(system, entry))
    if kernel:
        want = os.path.join(system, kernel)
        return want if want in found else None
    if len(found) > 1:
        print(f'  warning: {system} has several kernel configurations '
              f'({", ".join(sorted(os.path.basename(f) for f in found))}); '
              f'pick one with --nitro-kernel')
        return None
    return found.pop() if found else None


def figure_comparison(agg, spec, phases, datasets, systems, out_path, n_expected):
    """One group per sequence, one stacked bar per system.

    `agg` is keyed by (system, dataset); a sequence missing on either system is
    dropped rather than drawn as a gap.
    """
    datasets = [d for d in datasets if any((s, d) in agg for s in systems)]
    if not datasets:
        return False
    keys = [(s, d) for d in datasets for s in systems if (s, d) in agg]
    labels = visible_phases({k[1]: agg[k] for k in keys}, phases,
                            [k[1] for k in keys])
    cmap = colors_for(labels)

    # With the system read off the fill, a group need only be as wide as its bars.
    fig, ax = plt.subplots(figsize=(1.05 * len(datasets) + 2.6, 4.0))
    fig.patch.set_facecolor(SURFACE)
    span = max(sum(agg[k]['phases'].values()) for k in keys)

    width, gap = 0.34, 0.04
    for gi, dataset in enumerate(datasets):
        present = [s for s in systems if (s, dataset) in agg]
        extent = len(present) * width + (len(present) - 1) * gap
        for si, system in enumerate(present):
            x = gi + si * (width + gap) - extent / 2 + width / 2
            entry = agg[(system, dataset)]
            # Eight bars of per-segment values is more numbers than the eye can
            # use; the totals carry the comparison and the CSV carries the rest.
            top = draw_stack(ax, x, entry, labels, cmap, width, span,
                             show_values=False, hatch=HATCH.get(system))
            annotate_total(ax, x, top, entry, n_expected,
                           spec.get('show_run_count', True))
        # Speedup callout, only where both systems ran the sequence.
        if len(present) == 2:
            base = sum(agg[(present[0], dataset)]['phases'].values())
            other = sum(agg[(present[1], dataset)]['phases'].values())
            if other > 0:
                ax.text(gi, -0.115, f'{base / other:.2f}×', ha='center', va='top',
                        transform=ax.get_xaxis_transform(), fontsize=10, color=INK,
                        fontweight='semibold')

    ax.set_xticks([])
    for gi, dataset in enumerate(datasets):
        ax.text(gi, -0.03, dataset, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=11, color=INK)
    ax.set_xlim(-0.65, len(datasets) - 0.35)
    style_axes(ax, spec)
    # Neutral swatches for the fill key: a coloured one would read as a phase.
    key = [(plt.Rectangle((0, 0), 1, 1, facecolor='#b8b7b3', hatch=HATCH.get(s),
                          edgecolor=SURFACE), s) for s in systems]
    header(fig, ax, labels, cmap, spec, None, extra=key)

    fig.savefig(out_path, dpi=800, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


def run(chart, default_sequences, doc):
    """Argument parsing, aggregation and drawing for one thread's comparison.

    localmapping_comparison.py and loopclosing_comparison.py call this with their
    own chart spec, so the three comparisons stay identical but for the phases.
    """
    sns.set_context("paper")
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', nargs='*', default=RESULTS_ROOTS,
                    help='results roots to search, in order')
    ap.add_argument('--sequences', nargs='*', default=default_sequences,
                    help='sequences to plot, in this order')
    ap.add_argument('--systems', nargs=2, default=[BASELINE, CONTENDER],
                    metavar=('BASELINE', 'CONTENDER'),
                    help='the two systems to compare; the speedup is baseline/contender')
    ap.add_argument('--nitro-kernel', help='kernel-status directory to read for '
                                           'Nitro-SLAM, when the tree holds several')
    ap.add_argument('--machine', default=MACHINE, help='machine directory to read')
    ap.add_argument('--out', default='.', help='directory for the figure and CSV')
    ap.add_argument('--allow-missing-series', action='store_true',
                    help='keep runs whose stats dump is incomplete, counting the '
                         'absent phases as zero instead of dropping the run')
    args = ap.parse_args()

    phases = chart['phases']
    agg, n_expected = {}, 1
    for system in args.systems:
        path = resolve_system(args.results, system, args.machine, args.nitro_kernel)
        if path is None:
            print(f'  warning: no runs for {system} under '
                  f'{", ".join(args.results)}/*/{args.machine}')
            continue
        runs, missing = find_runs(args.results, args.sequences, path, args.machine)
        for d in missing:
            print(f'  warning: {system} has no "{d}"')
        if not runs:
            continue
        n_expected = max(n_expected,
                         max(len({r['iteration'] for r in runs if r['dataset'] == d})
                             for d in {r['dataset'] for r in runs}))
        one, skipped, absent = aggregate(runs, chart, phases,
                                         args.allow_missing_series)
        print(f'  {system} ({path}): {len(runs)} run directories')
        for dataset, entries in sorted(skipped.items()):
            for iteration, why in entries:
                print(f'    skipped {dataset}/{iteration} -- {why}')
        if absent:
            print(f'    series absent in some runs: {", ".join(absent)}')
        agg.update({(system, d): e for d, e in one.items()})

    if not agg:
        print('no data to compare')
        return 1

    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})
    path = os.path.join(args.out, chart['out'])
    if figure_comparison(agg, chart, phases, args.sequences, args.systems, path,
                         n_expected):
        print(f'  wrote {path}')

    # One CSV row per system/sequence/phase, tagged so both systems sit in one table.
    csv_path = os.path.join(args.out, chart['out'].replace('.png', '.csv'))
    tb.write_csv(csv_path, {s: {d: e for (sys_, d), e in agg.items() if sys_ == s}
                            for s in args.systems})
    print(f'  wrote {csv_path}')

    for dataset in args.sequences:
        a, b = (args.systems[0], dataset), (args.systems[1], dataset)
        if a in agg and b in agg:
            ta, tteb = sum(agg[a]['phases'].values()), sum(agg[b]['phases'].values())
            print(f'  {dataset}: {ta:.2f} ms → {tteb:.2f} ms  ({ta / tteb:.2f}× faster)')
    return 0


if __name__ == '__main__':
    sys.exit(run(CHART, SEQUENCES, __doc__))
