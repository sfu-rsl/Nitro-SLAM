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

Both platforms are drawn by default, into one directory with the platform in the
filename -- analysis_out/tracking_comparison_{desktop,jetson}.png.

Usage:
    ./tracking_comparison.py
    ./tracking_comparison.py --platform jetson --sequences MH01 room1
    ./tracking_comparison.py --nitro-kernel 11111-1111-001111 --csv
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import tracking_breakdown as tb
from tracking_breakdown import (INK, INK_MUTED, LEGEND_NCOL, OUT_ROOT, PLATFORMS,
                                SEQUENCES, SURFACE, aggregate, annotate_total,
                                colors_for, draw_stack, error_bar, find_runs, header,
                                out_path, scale_text, select_platforms, style_axes,
                                visible_phases)

BASELINE = 'ORB-SLAM3'
CONTENDER = 'Nitro-SLAM'
# Colour carries the phase, so the system is carried by fill instead: the baseline
# is solid and the contender striped. That frees the axis of per-bar labels, which
# were what forced the groups apart.
HATCH = {BASELINE: None, CONTENDER: '//'}
CHART = dict(tb.CHARTS['tracking'], out='tracking_comparison.png')

# --group pools sequences of one kind into a single pair of bars. Sequences within a
# family differ in length and route but not in what the thread is doing, so five
# Machine Hall bars mostly restate each other; one bar over all of their runs says
# the same thing with the spread that comes with it. Matched by prefix, longest
# first, and in this order on the axis. EuRoC's two Vicon rooms are kept apart
# because V1 and V2 are different rooms, not two runs of one.
GROUPS = [
    ('MH', 'Machine Hall'),
    ('V1', 'Vicon 1'),
    ('V2', 'Vicon 2'),
    ('room', 'Room'),
    ('corridor', 'Corridor'),
    ('magistrale', 'Magistrale'),
    ('outdoors', 'Outdoors'),
    ('slides', 'Slides'),
]
GROUP_ORDER = [label for _, label in GROUPS]


def group_of(dataset):
    """The family label for a sequence, or None when it belongs to none of them."""
    for prefix, label in GROUPS:
        if dataset.startswith(prefix):
            return label
    return None


def all_sequences(roots, frag, machine):
    """Every sequence under one system, in results-root order.

    Grouping is only meaningful over a whole family, so it discovers the sequences
    rather than using the handful the per-sequence figures name.
    """
    seqs = []
    for root in roots:
        base = os.path.join(root, frag, machine)
        if not os.path.isdir(base):
            continue
        seqs += [d for d in sorted(os.listdir(base))
                 if os.path.isdir(os.path.join(base, d)) and d not in seqs]
    return seqs


def regroup(runs):
    """Relabel each run's dataset with its family, dropping any that has none.

    Rewriting the label is all the pooling needs: aggregate() groups by it, so a
    family's bar is the mean over every run of every sequence in it, and its spread
    is across those runs rather than within one sequence.
    """
    kept, unmatched = [], set()
    for run_ in runs:
        label = group_of(run_['dataset'])
        if label is None:
            unmatched.add(run_['dataset'])
            continue
        kept.append(dict(run_, dataset=label))
    return kept, sorted(unmatched)


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


def abbreviate(label, limit):
    """`label` cut to `limit` characters, keeping whatever index it ends in.

    Sequence names differ only in their trailing digits -- magistrale1 against
    magistrale2 -- so cutting the tail would collapse two bars into one label. The
    digits are kept and the name in front of them is shortened, with an ellipsis
    marking the cut. Only what is drawn changes; the CSV keeps the full names.
    """
    if len(label) <= limit:
        return label
    head, tail = re.match(r'^(.*?)(\d*)$', label).groups()
    return f'{head[:max(1, limit - len(tail) - 1)]}\u2026{tail}'


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
    # Below the floor the fixed size wins, so the four-sequence figures are
    # unchanged; above it the axis grows rather than crushing the groups together.
    # The other comparisons plot four sequences and are sized for the column at
    # 5in; loop closing discovers its set and can run to a dozen, where 1.45in per
    # group is what keeps a "magistrale1" tick and a pair of stacked total
    # annotations clear of their neighbours.
    width_in = 5.0 if len(datasets) <= 4 else 1.45 * len(datasets)
    fig, ax = plt.subplots(figsize=(width_in, 4.0))
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
            if spec.get('error_bars'):
                error_bar(ax, x, top, entry)
            else:
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
    limit = spec.get('label_chars')
    for gi, dataset in enumerate(datasets):
        ax.text(gi, -0.03, abbreviate(dataset, limit) if limit else dataset,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=11, color=INK)
    ax.set_xlim(-0.65, len(datasets) - 0.35)
    style_axes(ax, spec)
    # Neutral swatches for the fill key: a coloured one would read as a phase.
    key = [(plt.Rectangle((0, 0), 1, 1, facecolor='#b8b7b3', hatch=HATCH.get(s),
                          edgecolor=SURFACE), s) for s in systems]
    header(fig, ax, labels, cmap, spec, None, extra=key)
    if spec.get('font_scale', 1.0) != 1.0:
        scale_text(fig, spec['font_scale'])

    fig.savefig(out_path, dpi=800, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


def build(args, platform, roots, machine, chart, default_sequences):
    """Draw one platform's comparison. Returns the number of figures written."""
    phases = chart['phases']
    if args.group or args.nseq:
        # Families are different sizes -- eight Outdoors sequences against three
        # Vicon 1 -- so "25/40 runs" would read as missing data rather than as how
        # many sequences the family has. The counts go to stdout instead. The spread
        # becomes an error bar rather than a printed total: pooling a family widens
        # it enough to be worth drawing, and eight groups leave no room for the
        # labels.
        #
        # --nseq gets exactly the same treatment over a named set of sequences, so
        # it comes out at the size and weight the grouped figures do; with the run
        # counts off the figure they matter more there, since a named sequence can
        # rest on a single closure where a pooled family never does.
        chart = dict(chart, show_run_count=False, error_bars=True, font_scale=1.5)
    if args.nseq:
        # A sequence name is longer than a family name, and at this type size
        # "magistrale1" and "magistrale2" run into each other at the width eight
        # groups get. Nine characters is what fits, and leaves "outdoors5" alone.
        chart = dict(chart, label_chars=9)
    resolved = {}
    for system in args.systems:
        path = resolve_system(roots, system, machine, args.nitro_kernel)
        if path is None:
            print(f'  warning: no runs for {system} under '
                  f'{", ".join(roots)}/*/{machine}')
            continue
        resolved[system] = path

    sequences = args.sequences if args.sequences is not None else default_sequences
    if args.nseq and args.sequences is None:
        sequences = chart.get('nseq_sequences', default_sequences)
    if args.group and args.sequences is None:
        sequences = []
        for frag in resolved.values():
            for name in all_sequences(roots, frag, machine):
                if name not in sequences:
                    sequences.append(name)

    agg, n_expected = {}, 1
    for system, path in resolved.items():
        runs, missing = find_runs(roots, sequences, path, machine)
        for d in missing:
            print(f'  warning: {system} has no "{d}"')
        if args.group:
            runs, unmatched = regroup(runs)
            if unmatched:
                print(f'    warning: no group for {", ".join(unmatched)}; dropped')
        if not runs:
            continue
        n_expected = max(n_expected,
                         max(len([r for r in runs if r['dataset'] == d])
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
        print('  no data to compare')
        return 0

    # Grouped, the axis is the families that turned up rather than the sequences,
    # and the figure gets its own name so it does not overwrite the per-sequence one.
    columns = ([g for g in GROUP_ORDER if any((s, g) in agg for s in args.systems)]
               if args.group else sequences)
    suffix = '_grouped' if args.group else '_nseq' if args.nseq else ''
    name = chart['out'].replace('.png', f'{suffix}.png')

    os.makedirs(args.out, exist_ok=True)
    written = 0
    path = out_path(args.out, name, platform)
    if figure_comparison(agg, chart, phases, columns, args.systems, path,
                         n_expected):
        print(f'  wrote {path}')
        written = 1

    # One CSV row per system/sequence/phase, tagged so both systems sit in one table.
    if args.csv:
        csv_path = out_path(args.out, name.replace('.png', '.csv'), platform)
        tb.write_csv(csv_path, {s: {d: e for (sys_, d), e in agg.items() if sys_ == s}
                                for s in args.systems})
        print(f'  wrote {csv_path}')

    for dataset in columns:
        a, b = (args.systems[0], dataset), (args.systems[1], dataset)
        if a in agg and b in agg:
            ta, tb_ = sum(agg[a]['phases'].values()), sum(agg[b]['phases'].values())
            runs = (f"  over {agg[a]['n_runs']}/{agg[b]['n_runs']} runs"
                    if args.group or args.nseq else '')
            print(f'  {dataset}: {ta:.2f} ms → {tb_:.2f} ms  '
                  f'({ta / tb_:.2f}× faster){runs}')
    return written


def run(chart, default_sequences, doc):
    """Argument parsing, aggregation and drawing for one thread's comparison.

    localmapping_comparison.py and loopclosing_comparison.py call this with their
    own chart spec, so the three comparisons stay identical but for the phases.
    """
    sns.set_context("paper")
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--platform', nargs='*', choices=sorted(PLATFORMS),
                    help='platforms to draw (default: all of them)')
    ap.add_argument('--results', nargs='*',
                    help="results roots to search, in order; overrides --platform's")
    ap.add_argument('--sequences', nargs='*',
                    help='sequences to plot, in this order')
    ap.add_argument('--nseq', action='store_true',
                    help="the chart's named sequence set, drawn the way --group "
                         'draws families: error bars, larger type, no run counts')
    ap.add_argument('--group', action='store_true',
                    help='pool sequences into one pair of bars per family '
                         '(Machine Hall, Vicon 1, Room, ...) over every sequence '
                         'found, instead of one pair per sequence')
    ap.add_argument('--systems', nargs=2, default=[BASELINE, CONTENDER],
                    metavar=('BASELINE', 'CONTENDER'),
                    help='the two systems to compare; the speedup is baseline/contender')
    ap.add_argument('--nitro-kernel', help='kernel-status directory to read for '
                                           'Nitro-SLAM, when the tree holds several')
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
        written += build(args, name, roots, machine, chart, default_sequences)
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(run(CHART, SEQUENCES, __doc__))
