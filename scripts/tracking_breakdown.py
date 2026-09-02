#!/usr/bin/env python3
"""Stacked-bar time breakdowns for the tracking thread, across EuRoC and TUM-VI.

Reads the "<key>: <value>" series written by the Stats/ singletons (same format
breakdown.py and timing_charts.py consume) from a run layout of

    <results root>/<system>/<machine>/<dataset>/<iteration>/Tracking/data/

and draws two figures over the four sequences named in SEQUENCES:

  tracking_breakdown.png   Tracking::Track split into its phases, per frame
  tlm_breakdown.png        Tracking::TrackLocalMap split into its phases, per call

Both are one bar per sequence, each segment the phase's mean time per iteration
averaged over the repeated runs. Sequences are pulled from whichever results root
holds them, so EuRoC and TUM-VI sit on the same axis.

Runs whose series are missing or empty are skipped and reported rather than
counted as zero, so a partially re-run sweep still plots -- the bar is annotated
with how many runs it actually averaged.

The same sweep is run on both platforms, so both are drawn by default, into one
directory with the platform in the filename:

    analysis_out/tracking_breakdown_desktop.png
    analysis_out/tracking_breakdown_jetson.png

Usage:
    ./tracking_breakdown.py
    ./tracking_breakdown.py --platform jetson --deep
    ./tracking_breakdown.py --sequences MH01 V101 room1 corridor1 --csv
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── what to plot ──────────────────────────────────────────────────────────────
RESULTS_ROOTS = ['Results-euroc', 'Results-tumvi']
SEQUENCES = ['MH01', 'V101', 'room1', 'corridor1']
SYSTEM = 'ORB-SLAM3'
MACHINE = 'desktop'
# The desktop and the Jetson run the same sweep into trees that differ only in
# their prefix and in the machine directory at the bottom, so a platform is just
# (roots, machine) and the two are interchangeable everywhere below. Figures are
# written per platform because the filenames are otherwise identical.
PLATFORMS = {
    'desktop': dict(roots=RESULTS_ROOTS, machine=MACHINE, label='Desktop'),
    'jetson': dict(roots=[os.path.join('data', 'jetson', r) for r in RESULTS_ROOTS],
                   machine='jetson', label='Orin Nano'),
}
OUT_ROOT = 'analysis_out'
# Which thread's stats directory a chart reads; a spec overrides it with 'subdir'.
# localmapping_breakdown.py reuses everything below with ('LocalMapping', 'data').
THREAD_SUBDIR = ('Tracking', 'data')

# ── palette ───────────────────────────────────────────────────────────────────
# Categorical slots in fixed order, shared with timing_charts.py so the figures
# read as one set. Verified with the data-viz validator on the light surface:
# every adjacent pair clears CVD dE >= 8 and normal-vision dE >= 15 *in this
# order*, so phases must be assigned in the order listed and never cycled. Three
# of the hues fall under 3:1 against the surface, which makes the direct value
# labels mandatory rather than decorative. "other" is a residual rather than an
# identity, so it takes the neutral secondary ink -- deliberately below the
# chroma floor, and separated from whichever hue it lands beside.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
OTHER_COLOR = '#52514e'
OTHER_LABEL = 'Other'          # the residual's name in the legend, CSV and lookups
LEGEND_NCOL = 3                # legend entries per row
SURFACE = '#ffffff'
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#e2e1dd'

# ── chart definitions ─────────────────────────────────────────────────────────
# (label, [series filenames]). Phases are leaves of the chart's total, so the
# residual "other" is meaningful; a parent series is replaced by its children
# rather than drawn alongside them.
CHARTS = {
    'tracking': dict(
        out='tracking_breakdown.png',
        title='Tracking Breakdown',
        total='tracking_time',
        unit='Mean Frametime (ms)',
        phases=[
            ('ORB extraction',    ['orbExtraction_time']),
            ('Stereo Matching',   ['stereoMatch_time']),
            ('Pose Prediction',   ['trackWithMotionModel_time']),
            ('Track Local map',   ['trackLocalMap_time']),
            ('Keyframe Creation', ['createKF_time']),
            ('Relocalization',    ['relocalization_time']),
        ],
        note=''
        # note='Total is wall time of Tracking::Track, so "other" includes tracking '
        #      'blocking on Local Mapping.',
    ),
    'tlm': dict(
        out='tlm_breakdown.png',
        title='TrackLocalMap Breakdown',
        total='trackLocalMap_time',
        unit='Mean TrackLocalMap Time (ms)',
        phases=[
            ('Update Local Map',  ['updateLocalMap_time']),
            ('Search Local Points', ['searchLocalPoints_time']),
            ('Pose Optimization', ['TLM_poseOptimization_time']),
        ],
        # --deep replaces the two composite phases with their own children, which
        # sum to within ~0.1% of their parents on these sequences.
        deep_phases=[
            ('update local KFs',    ['updateLocalKF_time']),
            ('update local points', ['updateLocalPoints_time']),
            ('local MP iteration',  ['SLP_localMapPointsItr_time']),
            ('search by projection', ['SLP_searchByProjection_time']),
            ('frame MP iteration',  ['SLP_frameMapPointsItr_time']),
            ('pose optimization',   ['TLM_poseOptimization_time']),
        ],
        note='',
        # note='Frames counted are those that reached TrackLocalMap, so the '
        #      'denominator is smaller than the frame count.',
    ),
}


# ── loading ───────────────────────────────────────────────────────────────────
def load(path):
    """{iteration key: summed value} for one series file, or None if absent.

    Series that fire more than once per iteration repeat their key, so values are
    summed rather than overwritten -- same rule as breakdown.py.
    """
    if not os.path.isfile(path):
        return None
    out = defaultdict(float)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            try:
                out[int(k.strip())] += float(v.strip())
            except ValueError:
                continue
    return dict(out)


def find_runs(roots, sequences, system, machine, subdir=THREAD_SUBDIR):
    """[{dataset, iteration, path}] for every run directory that exists.

    A sequence is looked up in each root in turn, so EuRoC and TUM-VI sequences
    can be named on one axis without the caller saying which root holds which.
    """
    runs, missing = [], []
    for dataset in sequences:
        found = False
        for root in roots:
            seq_dir = os.path.join(root, system, machine, dataset)
            if not os.path.isdir(seq_dir):
                continue
            found = True
            for it in sorted(os.listdir(seq_dir)):
                run_dir = os.path.join(seq_dir, it)
                if os.path.isdir(os.path.join(run_dir, *subdir)):
                    runs.append(dict(dataset=dataset, iteration=it, path=run_dir))
        if not found:
            missing.append(dataset)
    return runs, missing


def run_breakdown(run_dir, spec, phases, allow_missing=False):
    """Mean ms per iteration for each phase of one run.

    Returns (per-phase dict including the residual, iteration count, [absent series]),
    or None when the run is unusable for this chart. A phase series that is absent
    rather than empty means the stats dump did not complete -- the phase would be
    silently counted as zero and reappear inside "other" -- so by default such a
    run is dropped; --allow-missing-series keeps it with those phases at zero.
    An empty file is fine: it means the phase genuinely never fired.
    """
    data_dir = os.path.join(run_dir, *spec.get('subdir', THREAD_SUBDIR))
    total = load(os.path.join(data_dir, spec['total'] + '.txt'))
    if not total:
        return None

    # A chart can restrict itself to the iterations where a flag series fired --
    # loop closing is only interesting on the iterations that closed a loop.
    flag = spec.get('only_where')
    if flag:
        fired = load(os.path.join(data_dir, flag + '.txt')) or {}
        total = {k: v for k, v in total.items() if fired.get(k)}
        if not total:
            return None

    keys = set(total)
    n = len(keys)
    out, accounted, absent = {}, 0.0, []
    for label, files in phases:
        tot = 0.0
        for fname in files:
            s = load(os.path.join(data_dir, fname + '.txt'))
            if s is None:
                absent.append(fname)
                if not allow_missing:
                    return None, absent
                continue
            # Restrict to iterations present in the total so warm-up or shutdown
            # samples cannot inflate a phase's share.
            tot += sum(v for k, v in s.items() if k in keys)
        out[label] = tot / n
        accounted += tot
    out[OTHER_LABEL] = (sum(total.values()) - accounted) / n
    return out, n, absent


def aggregate(runs, spec, phases, allow_missing=False):
    """{dataset: {phases, phases_std, total_std, n_runs, n_iters}} over repeated runs."""
    grouped = defaultdict(list)
    skipped, absent = defaultdict(list), set()
    for r in runs:
        res = run_breakdown(r['path'], spec, phases, allow_missing)
        if res is None:
            skipped[r['dataset']].append(
                (r['iteration'], spec.get('empty_reason', f'no {spec["total"]} data')))
            continue
        if len(res) == 2:                  # incomplete dump, dropped
            skipped[r['dataset']].append(
                (r['iteration'], f'missing {", ".join(res[1])}'))
            absent.update(res[1])
            continue
        vals, n, miss = res
        absent.update(miss)
        grouped[r['dataset']].append((vals, n))

    labels = [lbl for lbl, _ in phases] + [OTHER_LABEL]
    agg = {}
    for dataset, results in grouped.items():
        per_run = [p for p, _ in results]
        agg[dataset] = dict(
            phases={l: float(np.mean([p[l] for p in per_run])) for l in labels},
            phases_std={l: float(np.std([p[l] for p in per_run], ddof=1))
                        if len(per_run) > 1 else 0.0 for l in labels},
            total_std=float(np.std([sum(p.values()) for p in per_run], ddof=1))
            if len(per_run) > 1 else 0.0,
            n_runs=len(per_run),
            n_iters=float(np.mean([n for _, n in results])),
        )
    return agg, dict(skipped), sorted(absent)


# ── drawing ───────────────────────────────────────────────────────────────────
def visible_phases(agg, phases, datasets):
    """Phase labels in fixed order, dropping any that is zero on every bar --
    a legend entry for a phase that never ran is noise."""
    labels = [lbl for lbl, _ in phases] + [OTHER_LABEL]
    return [l for l in labels
            if any(agg[d]['phases'].get(l, 0.0) > 1e-9 for d in datasets)]


def colors_for(labels):
    hues = [SLOTS[i % len(SLOTS)]
            for i in range(len(labels) - (1 if OTHER_LABEL in labels else 0))]
    return {l: (OTHER_COLOR if l == OTHER_LABEL else hues[i])
            for i, l in enumerate(labels)}


def _relative_luminance(hex_color):
    def channel(c):
        c = int(hex_color.lstrip('#')[c:c + 2], 16) / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = channel(0), channel(2), channel(4)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def label_ink(fill):
    """Black or white, whichever has more contrast against the segment fill.

    Three of the palette's hues are light enough that white sits below 4.5:1 on
    them, and the value labels are load-bearing -- those same hues are the ones
    under 3:1 against the surface -- so the ink is computed, not fixed.
    """
    lum = _relative_luminance(fill)
    return 'white' if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.05) else '#111111'


def draw_stack(ax, x, entry, labels, cmap, width, span, show_values=True,
               hatch=None):
    """One stacked bar. A surface-coloured edge leaves a 2px gap between segments
    so adjacent fills never touch.

    `span` is the tallest bar on the figure -- what the y axis is scaled to. Value
    labels are placed on the segments that are tall enough *on the axis* to hold
    text, not on the ones that are a large share of their own bar: when one
    sequence is 40x another, every segment of the short bar is a large share of a
    bar that is itself a few pixels tall, and the labels collide. show_values=False
    drops them entirely, for figures with too many bars to label legibly.
    """
    bottom = 0.0
    for lbl in labels:
        h = entry['phases'].get(lbl, 0.0)
        if h <= 1e-9:
            continue
        # No outline: a surface-coloured edge eats a thin segment from both sides
        # and leaves it reading as a gap. The edge colour is still set, because the
        # hatch is drawn in it -- light stripes over the fill, a second encoding for
        # figures that put two variants of the same phase side by side.
        ax.bar(x, h, width, bottom=bottom, color=cmap[lbl], hatch=hatch,
               edgecolor=SURFACE, linewidth=0, zorder=3)
        # Direct value labels on segments with room for them; smaller slices are
        # left to the total annotation and the CSV rather than crowded with text.
        if show_values and h >= 0.04 * span:
            ax.text(x, bottom + h / 2, f'{h:.2f}', ha='center', va='center',
                    fontsize=8.5, color=label_ink(cmap[lbl]), zorder=4,
                    fontweight='medium')
        bottom += h
    return bottom


def scale_text(fig, factor):
    """Multiply every text artist on `fig` by `factor`, after it is laid out.

    Cheaper than threading a size through every fontsize= in the drawing code, and
    it leaves a figure that does not ask for scaling byte-for-byte unchanged. Safe
    to run last: the legend grows upward out of the axes, so nothing it displaces
    lands on the plot.
    """
    for ax in fig.axes:
        items = ([ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
                 + ax.get_xticklabels() + ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            items += legend.get_texts()
        for item in items:
            item.set_fontsize(item.get_fontsize() * factor)
    for text in fig.texts:
        text.set_fontsize(text.get_fontsize() * factor)


def error_bar(ax, x, top, entry):
    """The spread across runs, as a cap on top of the bar.

    Stands in for the printed total where the numbers would crowd: with two bars per
    group the pair of labels is wider than the group itself.
    """
    if entry['total_std'] <= 0:
        return
    ax.errorbar(x, top, yerr=entry['total_std'], fmt='none', ecolor=INK_MUTED,
                elinewidth=1.1, capsize=4.0, capthick=1.1, zorder=5)


def annotate_total(ax, x, top, entry, n_expected, show_run_count=True):
    """The bar's total, and -- unless the chart opts out with show_run_count=False --
    how many runs it averaged when that is fewer than the rest of the figure."""
    # The spread goes on its own line under the mean: side by side it doubles the
    # label's width, which is what collides with the neighbouring bar's label.
    tag = f'{top:.2f}'
    if entry['total_std'] > 0:
        tag += f'\n± {entry["total_std"]:.2f}'
    if show_run_count and entry['n_runs'] < n_expected:
        tag += f'\n({entry["n_runs"]}/{n_expected} runs)'
    ax.text(x, top * 1.015, tag, ha='center', va='bottom',
            fontsize=9, color=INK_MUTED, linespacing=1.3)


def figure(agg, spec, phases, datasets, out_path, n_expected, subtitle):
    datasets = [d for d in datasets if d in agg]
    if not datasets:
        return False
    labels = visible_phases(agg, phases, datasets)
    cmap = colors_for(labels)

    # fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 5.0, 5.4))
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.patch.set_facecolor(SURFACE)
    span = max(sum(agg[d]['phases'].values()) for d in datasets)
    for i, dataset in enumerate(datasets):
        top = draw_stack(ax, i, agg[dataset], labels, cmap, 0.5, span)
        annotate_total(ax, i, top, agg[dataset], n_expected,
                       spec.get('show_run_count', True))

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=11, color=INK)
    ax.set_xlim(-0.7, len(datasets) - 0.3)
    style_axes(ax, spec)
    header(fig, ax, labels, cmap, spec, subtitle)

    fig.savefig(out_path, dpi=800, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


def style_axes(ax, spec):
    """Grid, spines, ticks and headroom -- shared by every figure in the set, so
    tracking_comparison.py stays in step with the single-bar charts."""
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_ylabel(spec['unit'], color=INK_MUTED, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    # ax.grid(axis='x', visible=False)
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#d6d5d0')
    ax.tick_params(colors=INK_MUTED, labelsize=10, length=0)


def header(fig, ax, labels, cmap, spec, subtitle, extra=None):
    """Legend, optional subtitle and title, stacked above the plot.

    `extra` is [(handle, label)] appended after the phase entries, for a figure
    that encodes something else alongside the phases.
    """
    # Lay the axes out first: the legend, subtitle and title are then positioned
    # against the axes' final height, and savefig's tight bbox grows the canvas to
    # include whatever they overflow above it.
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    # Legend centred above the plot, LEGEND_NCOL entries per row. Reversed so it
    # reads top-down in the order the segments stack; with more than one row
    # matplotlib fills column by column, which keeps that order intact down each
    # column.
    ordered = list(reversed(labels))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap[l], edgecolor=SURFACE)
               for l in ordered]
    for handle, label in (extra or []):
        handles.append(handle)
        ordered.append(label)
    # leg = ax.legend(handles, ordered, loc='lower center', bbox_to_anchor=(0.5, 1.015),
    #                 ncol=min(LEGEND_NCOL, len(ordered)),
    #                 frameon=False, fontsize=9.5, labelcolor=INK_MUTED,
    #                 handlelength=1.1, handleheight=1.1, borderpad=0,
    #                 columnspacing=1.4, handletextpad=0.6, labelspacing=0.5)

    leg = ax.legend(handles, ordered, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                    ncol=min(LEGEND_NCOL, len(ordered)), frameon=False)

    # How tall the legend ends up depends on how many rows it wrapped to, so each
    # line above it is stacked on the measured top of the one below.
    def stack_above(y, text, size, weight='normal', color=INK_MUTED):
        t = ax.text(0, y, text, transform=ax.transAxes, fontsize=size, color=color,
                    fontweight=weight, va='bottom')
        fig.canvas.draw()
        return t.get_window_extent().transformed(ax.transAxes.inverted()).y1

    fig.canvas.draw()
    top = leg.get_window_extent().transformed(ax.transAxes.inverted()).y1
    if subtitle:                       # optional -- the title stacks straight on
        top = stack_above(top + 0.025, subtitle, 9.5)
    # stack_above(top + 0.02, spec['title'], 13, weight='semibold', color=INK)

    fig.text(0.011, 0.008, spec['note'], fontsize=8.5, color=INK_MUTED, style='italic')


# ── table view ────────────────────────────────────────────────────────────────
def write_csv(path, all_agg):
    rows = ['chart,dataset,phase,mean_ms,std_ms,n_runs,mean_iterations']
    for chart, agg in all_agg.items():
        for dataset, e in agg.items():
            for lbl, v in e['phases'].items():
                rows.append(f'{chart},{dataset},{lbl},{v:.4f},'
                            f'{e["phases_std"][lbl]:.4f},{e["n_runs"]},{e["n_iters"]:.1f}')
            rows.append(f'{chart},{dataset},TOTAL,{sum(e["phases"].values()):.4f},'
                        f'{e["total_std"]:.4f},{e["n_runs"]},{e["n_iters"]:.1f}')
    with open(path, 'w') as f:
        f.write('\n'.join(rows) + '\n')


def out_path(out_dir, filename, platform):
    """<out_dir>/<stem>_<platform><ext>.

    The platform goes in the name rather than a directory above it, so a figure and
    its counterpart sit side by side wherever the set is collected.
    """
    stem, ext = os.path.splitext(filename)
    return os.path.join(out_dir, f'{stem}_{platform}{ext}')


def select_platforms(args):
    """[(name, roots, machine)] to draw, honouring --results/--machine overrides.

    Both platforms are drawn by default. An explicit root or machine describes one
    tree rather than a family of them, so it is only accepted alongside a single
    --platform -- otherwise it would silently point both platforms at the same runs.
    """
    chosen = args.platform or sorted(PLATFORMS)
    override = args.results is not None or args.machine is not None
    if override and len(chosen) > 1:
        raise SystemExit('--results/--machine describe one tree; name a single '
                         '--platform alongside them')
    return [(name, args.results or PLATFORMS[name]['roots'],
             args.machine or PLATFORMS[name]['machine']) for name in chosen]


def build(args, platform, roots, machine):
    """Draw every chart for one platform. Returns the number of figures written."""
    runs, missing = find_runs(roots, args.sequences, args.system, machine)
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

    written = 0
    all_agg = {}
    for chart, spec in CHARTS.items():
        phases = spec['deep_phases'] if (args.deep and 'deep_phases' in spec) \
            else spec['phases']
        agg, skipped, absent = aggregate(runs, spec, phases, args.allow_missing_series)
        all_agg[chart] = agg
        if not agg:
            print(f'  {chart}: no data')
            continue
        # Re-runs in flight show up as skipped iterations, not as zeroed bars.
        for dataset, entries in sorted(skipped.items()):
            for iteration, why in entries:
                print(f'  {chart}: skipped {dataset}/{iteration} -- {why}')
        if absent:
            print(f'  {chart}: series absent in some runs: {", ".join(absent)}')
        subtitle = None
        path = out_path(args.out, spec['out'], platform)
        if figure(agg, spec, phases, args.sequences, path, n_expected, subtitle):
            print(f'  wrote {path}')
            written += 1

    if args.csv:
        csv_path = out_path(args.out, 'tracking_breakdown.csv', platform)
        write_csv(csv_path, all_agg)
        print(f'  wrote {csv_path}')
    return written


def main():
    sns.set_context("paper")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--platform', nargs='*', choices=sorted(PLATFORMS),
                    help='platforms to draw (default: all of them)')
    ap.add_argument('--results', nargs='*',
                    help='results roots to search, in order; overrides --platform\'s')
    ap.add_argument('--sequences', nargs='*', default=SEQUENCES,
                    help='sequences to plot, in this order')
    ap.add_argument('--system', default=SYSTEM, help='system directory to read')
    ap.add_argument('--machine', help="machine directory to read; overrides --platform's")
    ap.add_argument('--out', default=OUT_ROOT,
                    help='directory to write the figures into')
    ap.add_argument('--csv', action='store_true',
                    help='also write the per-phase table beside the figures')
    ap.add_argument('--allow-missing-series', action='store_true',
                    help='keep runs whose stats dump is incomplete, counting the '
                         'absent phases as zero instead of dropping the run')
    ap.add_argument('--deep', action='store_true',
                    help='split TrackLocalMap into leaf phases instead of its three parts')
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
