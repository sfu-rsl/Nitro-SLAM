#!/usr/bin/env python3
"""Aggregate per-thread timing across repeated runs and draw stacked-bar breakdowns.

Consumes the same "<key>: <value>" series that breakdown.py reads, but over a whole
sweep of runs laid out by run_script.sh as

    Results/<system>/[<kernel_dir>/]<version>/<dataset>/<iteration>/

and reduces them to one stacked bar per (system, dataset): each segment is a phase's
mean time per iteration of that thread, averaged over the repeated runs.

Produces nine figures -- {tracking, local mapping, loop closing} x {Nitro-SLAM,
ORB-SLAM3, both side by side} -- plus a tidy CSV of every number drawn.

Usage:
    ./timing_charts.py --version timing
    ./timing_charts.py --version timing --results Results --out figures
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

# ── palette ───────────────────────────────────────────────────────────────────
# Categorical slots in fixed order from the validated data-viz palette; adjacent
# pairs clear the CVD (dE >= 8) and normal-vision (dE >= 15) gates in this order,
# so phases must be assigned in the order listed and never cycled. "other" is a
# residual rather than an identity, so it takes the neutral secondary ink -- which
# also keeps it separated from whichever hue it ends up beside.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
OTHER_COLOR = '#52514e'
SURFACE = '#fcfcfb'
INK, INK_MUTED = '#0b0b0b', '#52514e'

# ── thread definitions ────────────────────────────────────────────────────────
# (label, series filename). Phases are leaves of the thread's total, so the
# residual "other" is meaningful; parents (e.g. loopCorrection) are deliberately
# replaced by their children.
THREADS = {
    'tracking': dict(
        title='Tracking thread',
        subdir=('Tracking', 'data'),
        total='tracking_time',
        unit='mean time per frame (ms)',
        phases=[
            ('ORB extraction',    ['orbExtraction_time']),
            ('stereo matching',   ['stereoMatch_time']),
            ('pose prediction',   ['trackWithMotionModel_time']),
            ('track local map',   ['trackLocalMap_time']),
            ('keyframe creation', ['createKF_time']),
            ('relocalization',    ['relocalization_time']),
        ],
        note='"other" includes tracking blocking on Local Mapping.',
    ),
    'localmapping': dict(
        title='Local Mapping thread',
        subdir=('LocalMapping', 'data'),
        total='localMapping_time',
        unit='mean time per keyframe (ms)',
        phases=[
            ('local BA',           ['LBA_time']),
            ('map point fusion',   ['searchInNeighbors_time']),
            ('search + triangul.', ['MPCreation_time']),
            ('process keyframe',   ['processKF_time']),
            ('keyframe culling',   ['KFCulling_time']),
            ('map point culling',  ['MPCulling_time']),
            # scale refinement is the tail of the same IMU-initialisation pipeline.
            ('IMU init / VIBA',    ['imuInit_time', 'scaleRefinement_time']),
        ],
        note='local BA and keyframe culling are skipped when a keyframe arrives mid-iteration.',
    ),
    'loopclosing': dict(
        title='Loop Closing thread',
        subdir=('LoopClosing', 'data'),
        total='loopClosing_time',
        unit='mean time per loop closure (ms)',
        phases=[
            ('place recognition',  ['placeRecognition_time']),
            ('loop fusion',        ['loopFusion_time']),
            ('graph optimization', ['graphOptimization_time']),
        ],
        note='iterations with loopClosed == 1 only; global BA runs on its own thread '
             'and is not part of the total.',
        closures_only=True,
    ),
}

SYSTEM_ORDER = ['ORB-SLAM3', 'Nitro-SLAM']


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


def find_runs(results_dir, version):
    """Locate every run directory for `version`, keyed by (system, dataset, iteration).

    The kernel-status component is present only when optimisations are enabled, so
    the depth varies; anchor on the `version` component instead of a fixed depth.
    """
    runs = []
    for root, dirs, _ in os.walk(results_dir):
        if not os.path.isdir(os.path.join(root, 'Tracking', 'data')):
            continue
        dirs[:] = []                       # a run directory has no runs beneath it
        rel = os.path.relpath(root, results_dir).split(os.sep)
        if len(rel) < 4 or version not in rel[1:-2]:
            continue
        runs.append(dict(path=root, system=rel[0], dataset=rel[-2], iteration=rel[-1]))
    return runs


def run_breakdown(run_dir, spec):
    """Mean ms per iteration for each phase of one thread in one run.

    Returns (per_phase_dict, n_iterations) or None when the thread produced no data.
    """
    data_dir = os.path.join(run_dir, *spec['subdir'])
    total = load(os.path.join(data_dir, spec['total'] + '.txt'))
    if not total:
        return None

    if spec.get('closures_only'):
        closed = load(os.path.join(data_dir, 'loopClosed.txt')) or {}
        total = {k: v for k, v in total.items() if closed.get(k)}
        if not total:
            return None                    # this run never closed a loop

    keys = set(total)
    n = len(keys)
    out, accounted = {}, 0.0
    for label, files in spec['phases']:
        tot = 0.0
        for fname in files:
            s = load(os.path.join(data_dir, fname + '.txt')) or {}
            # Restrict to iterations present in the total so warm-up or shutdown
            # samples cannot inflate a phase's share.
            tot += sum(v for k, v in s.items() if k in keys)
        out[label] = tot / n
        accounted += tot
    out['other'] = (sum(total.values()) - accounted) / n
    return out, n


def aggregate(runs, spec):
    """Average each phase's per-iteration mean over the repeated runs.

    Returns {(system, dataset): {'phases': {...}, 'total_std': float, 'n_runs': int,
                                 'n_iters': float}}.
    """
    grouped = defaultdict(list)
    for r in runs:
        res = run_breakdown(r['path'], spec)
        if res is not None:
            grouped[(r['system'], r['dataset'])].append(res)

    agg = {}
    for key, results in grouped.items():
        phases = [p for p, _ in results]
        labels = [lbl for lbl, _ in spec['phases']] + ['other']
        agg[key] = dict(
            phases={lbl: float(np.mean([p[lbl] for p in phases])) for lbl in labels},
            phases_std={lbl: float(np.std([p[lbl] for p in phases], ddof=1))
                        if len(phases) > 1 else 0.0 for lbl in labels},
            total_std=float(np.std([sum(p.values()) for p in phases], ddof=1))
            if len(phases) > 1 else 0.0,
            n_runs=len(results),
            n_iters=float(np.mean([n for _, n in results])),
        )
    return agg


# ── drawing ───────────────────────────────────────────────────────────────────
def visible_phases(agg, spec, keys):
    """Phase labels in fixed order, dropping any that is zero everywhere on this
    figure -- a legend entry for a phase that never ran is noise."""
    labels = [lbl for lbl, _ in spec['phases']] + ['other']
    return [lbl for lbl in labels
            if any(agg[k]['phases'].get(lbl, 0.0) > 1e-9 for k in keys)]


def colors_for(labels):
    hues = [SLOTS[i % len(SLOTS)] for i in range(len(labels) - (1 if 'other' in labels else 0))]
    return {lbl: (OTHER_COLOR if lbl == 'other' else hues[i])
            for i, lbl in enumerate(labels)}


def draw_stack(ax, x, agg_entry, labels, cmap, width):
    """One stacked bar. A surface-coloured edge leaves a hairline gap between
    segments so adjacent fills never touch."""
    bottom = 0.0
    for lbl in labels:
        h = agg_entry['phases'].get(lbl, 0.0)
        if h <= 1e-9:
            continue
        ax.bar(x, h, width, bottom=bottom, color=cmap[lbl],
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
        # Direct value labels on the fat segments: three of the palette's hues sit
        # below 3:1 on a light surface, and the relief rule makes labels mandatory.
        if h / max(sum(agg_entry['phases'].values()), 1e-9) >= 0.07:
            ax.text(x, bottom + h / 2, f'{h:.1f}', ha='center', va='center',
                    fontsize=8.5, color='white', zorder=4,
                    fontweight='medium')
        bottom += h
    return bottom


def annotate_total(ax, x, top, std, n_runs, n_total):
    err = f' ± {std:.1f}' if std > 0 else ''
    tag = f'{top:.1f}{err}'
    if n_runs < n_total:
        tag += f'\n({n_runs}/{n_total} runs)'
    ax.text(x, top * 1.015, tag, ha='center', va='bottom',
            fontsize=9, color=INK_MUTED, linespacing=1.3)


def style(ax, spec, subtitle):
    ax.set_ylabel(spec['unit'], color=INK_MUTED, fontsize=10)
    ax.set_title(f"{spec['title']} breakdown\n", fontsize=13, color=INK,
                 fontweight='semibold', loc='left')
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=INK_MUTED, va='bottom')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e2e1dd', linewidth=0.8)
    ax.grid(axis='x', visible=False)
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#d6d5d0')
    ax.tick_params(colors=INK_MUTED, labelsize=10, length=0)


def legend(fig, ax, labels, cmap, note):
    # Reversed so the legend reads top-down in the same order the segments stack.
    ordered = list(reversed(labels))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap[l], edgecolor=SURFACE)
               for l in ordered]
    leg = ax.legend(handles, ordered, loc='upper left', bbox_to_anchor=(1.015, 1.0),
                    frameon=False, fontsize=9.5, labelcolor=INK_MUTED,
                    handlelength=1.1, handleheight=1.1, borderpad=0)
    leg.set_title('phase', prop=dict(size=9.5, weight='semibold'))
    leg.get_title().set_color(INK_MUTED)
    leg.get_title().set_ha('left')
    fig.text(0.011, 0.008, note, fontsize=8.5, color=INK_MUTED, style='italic')


def figure_single(agg, spec, system, datasets, out_path, n_total):
    # Sequences with no data for this thread are dropped, not drawn empty.
    keys = [(system, d) for d in datasets if (system, d) in agg]
    if not keys:
        return False
    labels = visible_phases(agg, spec, keys)
    cmap = colors_for(labels)

    fig, ax = plt.subplots(figsize=(1.5 * len(keys) + 5.2, 5.4))
    fig.patch.set_facecolor(SURFACE)
    for i, key in enumerate(keys):
        top = draw_stack(ax, i, agg[key], labels, cmap, 0.5)
        annotate_total(ax, i, top, agg[key]['total_std'], agg[key]['n_runs'], n_total)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[1] for k in keys], fontsize=11, color=INK)
    ax.set_xlim(-0.7, len(keys) - 0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    style(ax, spec, f'{system} · mean of {n_total} runs per sequence')
    legend(fig, ax, labels, cmap, spec['note'])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


def figure_comparison(agg, spec, systems, datasets, out_path, n_total):
    # A sequence with no data on either system (e.g. one that never closed a loop)
    # is dropped outright rather than left as an empty slot on the axis.
    datasets = [d for d in datasets if any((s, d) in agg for s in systems)]
    keys = [(s, d) for d in datasets for s in systems if (s, d) in agg]
    if not keys:
        return False
    labels = visible_phases(agg, spec, keys)
    cmap = colors_for(labels)

    fig, ax = plt.subplots(figsize=(2.1 * len(datasets) + 5.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    width, gap = 0.34, 0.04
    ticks, ticklabels = [], []
    for gi, dataset in enumerate(datasets):
        present = [s for s in systems if (s, dataset) in agg]
        span = len(present) * width + (len(present) - 1) * gap
        for si, system in enumerate(present):
            x = gi + (si * (width + gap)) - span / 2 + width / 2
            top = draw_stack(ax, x, agg[(system, dataset)], labels, cmap, width)
            annotate_total(ax, x, top, agg[(system, dataset)]['total_std'],
                           agg[(system, dataset)]['n_runs'], n_total)
            ticks.append(x)
            ticklabels.append(system)
        # Speedup callout when both systems ran the same sequence.
        if len(present) == 2:
            a = sum(agg[(present[0], dataset)]['phases'].values())
            b = sum(agg[(present[1], dataset)]['phases'].values())
            if b > 0:
                ax.text(gi, -0.115, f'{a / b:.2f}× faster', transform=
                        ax.get_xaxis_transform(), ha='center', va='top',
                        fontsize=9.5, color=INK, fontweight='semibold')

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=9.5, color=INK_MUTED)
    for gi, dataset in enumerate(datasets):
        ax.text(gi, -0.065, dataset, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=11.5, color=INK, fontweight='semibold')
    ax.set_xlim(-0.75, len(datasets) - 0.25)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    style(ax, spec, f'ORB-SLAM3 vs Nitro-SLAM · mean of {n_total} runs per sequence')
    legend(fig, ax, labels, cmap, spec['note'])
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


# ── table view ────────────────────────────────────────────────────────────────
def write_csv(path, all_agg):
    rows = ['thread,system,dataset,phase,mean_ms,std_ms,n_runs,mean_iterations']
    for thread, agg in all_agg.items():
        for (system, dataset), e in sorted(agg.items()):
            for lbl, v in e['phases'].items():
                rows.append(f'{thread},{system},{dataset},{lbl},{v:.4f},'
                            f'{e["phases_std"][lbl]:.4f},{e["n_runs"]},{e["n_iters"]:.1f}')
            rows.append(f'{thread},{system},{dataset},TOTAL,'
                        f'{sum(e["phases"].values()):.4f},{e["total_std"]:.4f},'
                        f'{e["n_runs"]},{e["n_iters"]:.1f}')
    with open(path, 'w') as f:
        f.write('\n'.join(rows) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='Results', help='root of the Results tree')
    ap.add_argument('--version', required=True, help='version component used by run_script.sh')
    ap.add_argument('--out', default='figures', help='directory for the figures and CSV')
    ap.add_argument('--datasets', nargs='*', help='restrict to these sequences, in this order')
    args = ap.parse_args()

    runs = find_runs(args.results, args.version)
    if not runs:
        print(f'no runs found under {args.results} for version "{args.version}"')
        return 1

    systems = [s for s in SYSTEM_ORDER if any(r['system'] == s for r in runs)]
    systems += sorted({r['system'] for r in runs} - set(systems))
    datasets = args.datasets or sorted({r['dataset'] for r in runs})
    n_total = max(len({r['iteration'] for r in runs
                       if r['system'] == s and r['dataset'] == d})
                  for s in systems for d in datasets) or 1

    print(f'{len(runs)} runs · systems: {", ".join(systems)} · '
          f'sequences: {", ".join(datasets)}')
    os.makedirs(args.out, exist_ok=True)
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})

    all_agg = {}
    for thread, spec in THREADS.items():
        agg = aggregate(runs, spec)
        all_agg[thread] = agg
        if not agg:
            print(f'  {thread}: no data')
            continue
        for system in systems:
            p = os.path.join(args.out, f'{thread}_{system.lower()}.png')
            if figure_single(agg, spec, system, datasets, p, n_total):
                print(f'  wrote {p}')
        p = os.path.join(args.out, f'{thread}_comparison.png')
        if figure_comparison(agg, spec, systems, datasets, p, n_total):
            print(f'  wrote {p}')

    csv_path = os.path.join(args.out, 'breakdown_summary.csv')
    write_csv(csv_path, all_agg)
    print(f'  wrote {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
