#!/usr/bin/env python3
"""Peak CPU and GPU memory per sequence, for Nitro-SLAM.

Reads the per-run memory.csv and memory_summary.txt that monitor_memory.py writes
beside every run, over every sequence present in the results roots:

    <results root>/<system>/[<kernel status>/]<machine>/<dataset>/<iteration>/

Draws one figure, memory_usage.png: one plot, two bars per sequence, CPU beside
GPU. Both readings are MiB, so they share one axis honestly. The time averages are computed too and written to
memory_usage.csv beside the figure, they are simply not drawn. --systems takes more
than one system when a comparison is wanted, and adds a legend when it does.

CPU is resident set size; its peak comes from the kernel's VmHWM in the summary,
which is exact rather than sampling-limited. GPU is the device's used memory minus
the baseline taken before the run, because NVML does not report per-process usage
on these machines -- gpu_proc_mib is empty in every run.

Thirty-nine bars is a wall rather than a figure, so it plots the four sequences in
SEQUENCES by default, light to heavy; --all-sequences restores the full sweep, and
--group collapses them into one bar per sequence family, holding the largest value
each family reached.

Both platforms are drawn by default, into one directory with the platform in the
filename -- analysis_out/memory_usage_{desktop,jetson}.png. The GPU reading is not
the same measurement on the two: the desktop's comes from NVML as a discrete
device's used memory, the Jetson's from jtop, where the iGPU allocates out of the
same system RAM the CPU bar is counting.

Usage:
    ./memory_usage.py
    ./memory_usage.py --all-sequences
    ./memory_usage.py --group
    ./memory_usage.py --all-sequences --combined
    ./memory_usage.py --ratio
    ./memory_usage.py --platform jetson --sequences MH01 room1 corridor1
    ./memory_usage.py --systems Nitro-SLAM ORB-SLAM3 --csv
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tracking_breakdown import (GRID, INK, INK_MUTED, OUT_ROOT, PLATFORMS, SLOTS,
                                SURFACE, out_path, select_platforms)
from tracking_comparison import (BASELINE, CONTENDER, GROUP_ORDER, group_of,
                                resolve_system)

# Colour carries the metric, since that is what the two bars in a group differ by.
# A second system, if one is asked for, is carried by fill -- the same convention
# the comparison charts use.
METRICS = [
    dict(key='cpu', label='CPU (resident)', color=SLOTS[0]),
    # SLOTS[2] rather than the orange next to it: only two colours are in play
    # here, and blue against green separates further under colour-vision
    # deficiency than blue against orange does.
    dict(key='gpu', label='GPU (over baseline)', color=SLOTS[2]),
]
SYSTEM_HATCH = [None, '//']

# One light, one mid, two heavy -- the span the sweep covers, in ascending order of
# peak memory. Not tracking_breakdown's SEQUENCES: those four are chosen for their
# timing behaviour and sit almost on top of each other on this axis.
SEQUENCES = ['MH01', 'corridor1', 'magistrale1', 'outdoors6']


def keyframe_count(run_dir):
    """Keyframes in the saved map, one per line of the kf_*.txt trajectory.

    This is the map after culling, which is what the memory bars are holding -- not
    the number of keyframes ever created.
    """
    found = glob.glob(os.path.join(run_dir, 'trajectory', 'kf_*.txt'))
    if not found:
        return None
    with open(found[0]) as f:
        return sum(1 for line in f if line.strip())


def read_run(run_dir):
    """{cpu_avg, cpu_peak, gpu_avg, gpu_peak, n_kfs} for one run, or None.

    Trailing samples where RSS has fallen to zero are the process being unmapped at
    exit rather than real occupancy, so they are dropped before averaging -- same
    rule memory_charts.py uses.
    """
    path = os.path.join(run_dir, 'memory.csv')
    if not os.path.isfile(path):
        return None
    cpu, gpu = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                cpu.append(float(row['cpu_rss_mib'] or 0.0))
                gpu.append(float(row['gpu_dev_delta_mib'] or 0.0))
            except (KeyError, ValueError):
                continue
    end = len(cpu)
    while end > 1 and cpu[end - 1] <= 0.0:
        end -= 1
    cpu, gpu = np.asarray(cpu[:end]), np.asarray(gpu[:end])
    if not cpu.size:
        return None

    summary = {}
    for name in ('memory_summary.txt', 'monitor.log'):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            for line in open(p):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        summary[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
            break
    return dict(
        cpu_avg=float(cpu.mean()),
        # VmHWM is the kernel's own high-water mark, so prefer it to the samples.
        cpu_peak=summary.get('cpu_hwm_rss_mib', summary.get('cpu_peak_rss_mib',
                                                            float(cpu.max()))),
        gpu_avg=float(gpu.mean()),
        gpu_peak=summary.get('gpu_peak_delta_mib', float(gpu.max()) if gpu.size else 0.0),
        n_kfs=keyframe_count(run_dir),
    )


def collect(roots, systems, machine, sequences, kernel=None):
    """{(system, dataset): {metric: (mean, std)}, ...}, plus the sequences found.

    Sequences are discovered from the tree when none are named, so a sweep that has
    only run half the datasets still plots what it has.
    """
    agg, found = {}, []
    paths = {s: resolve_system(roots, s, machine, kernel) for s in systems}
    for root in roots:
        for system in systems:
            base = os.path.join(root, paths[system] or system, machine)
            if not os.path.isdir(base):
                continue
            for dataset in sorted(os.listdir(base)):
                if sequences and dataset not in sequences:
                    continue
                seq_dir = os.path.join(base, dataset)
                runs = [read_run(os.path.join(seq_dir, i))
                        for i in sorted(os.listdir(seq_dir))]
                runs = [r for r in runs if r]
                if not runs:
                    continue
                if dataset not in found:
                    found.append(dataset)
                agg[(system, dataset)] = {
                    m: (float(np.mean([r[m] for r in runs])),
                        float(np.std([r[m] for r in runs], ddof=1)) if len(runs) > 1
                        else 0.0)
                    for m in ('cpu_avg', 'cpu_peak', 'gpu_avg', 'gpu_peak')}
                agg[(system, dataset)]['n_runs'] = len(runs)
                kfs = [r['n_kfs'] for r in runs if r['n_kfs'] is not None]
                agg[(system, dataset)]['n_kfs'] = (
                    (float(np.mean(kfs)),
                     float(np.std(kfs, ddof=1)) if len(kfs) > 1 else 0.0)
                    if kfs else None)
    return agg, found


def figure(agg, systems, datasets, path):
    """One plot: a CPU bar and a GPU bar per sequence, per system."""
    # Rotation buys horizontal room when the axis is crowded and costs legibility
    # when it is not, so it follows the sequence count -- and the width follows the
    # rotation: upright labels need room to print their names side by side, tilted
    # ones can pack tighter. The wide many-sequence figure is unchanged.
    tilt = len(datasets) > 8
    fig, ax = plt.subplots(figsize=((0.42 if tilt else 0.95) * len(datasets) + 2.0,
                                    4.4))
    fig.patch.set_facecolor(SURFACE)

    bars = [(system, metric) for system in systems for metric in METRICS]
    width, gap = 0.84 / len(bars), 0.03
    extent = len(bars) * width + (len(bars) - 1) * gap
    for bi, (system, metric) in enumerate(bars):
        hatch = SYSTEM_HATCH[systems.index(system) % len(SYSTEM_HATCH)] \
            if len(systems) > 1 else None
        for di, dataset in enumerate(datasets):
            entry = agg.get((system, dataset))
            if not entry:
                continue
            x = di + bi * (width + gap) - extent / 2 + width / 2
            ax.bar(x, entry[f'{metric["key"]}_peak'][0], width, color=metric['color'],
                   hatch=hatch, edgecolor=SURFACE, linewidth=0, zorder=3)

    ax.set_ylabel('Peak Memory (MiB)', color=INK_MUTED, fontsize=9.5)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.grid(axis='x', visible=False)
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#d6d5d0')
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_ylim(bottom=0)
    # The keyframe count rides under the sequence name: memory is dominated by how
    # much map is resident, so the axis says how big each map got rather than
    # leaving the reader to infer it from the sequence's reputation. Averaged over
    # the systems drawn, which agree to a few percent.
    labels = []
    for dataset in datasets:
        kfs = [agg[(s, dataset)]['n_kfs'][0] for s in systems
               if (s, dataset) in agg and agg[(s, dataset)].get('n_kfs')]
        labels.append(f'{dataset}\n{np.mean(kfs):,.0f} KF' if kfs else dataset)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(labels, fontsize=9, color=INK,
                       rotation=45 if tilt else 0,
                       ha='right' if tilt else 'center',
                       rotation_mode='anchor' if tilt else None)
    ax.set_xlim(-0.7, len(datasets) - 0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=m['color']) for m in METRICS]
    labels = [m['label'] for m in METRICS]
    if len(systems) > 1:                 # neutral swatches: colour means metric here
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor='#b8b7b3',
                                  hatch=SYSTEM_HATCH[i % len(SYSTEM_HATCH)],
                                  edgecolor=SURFACE)
                    for i, _ in enumerate(systems)]
        labels += list(systems)
    elif systems:
        # The legend sits in the strip directly above the axes, so the title has
        # to clear it. On a wide axis a left-aligned title and a centred legend
        # miss each other by luck; at four sequences the axis is narrow enough
        # that they collide, so the title is lifted a full row instead.
        ax.set_title(systems[0], color=INK, fontsize=11, loc='left',
                     fontweight='semibold', pad=26)
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.005),
              ncol=len(labels), frameon=False, fontsize=9.5, labelcolor=INK_MUTED,
              handlelength=1.1, handleheight=1.1, borderpad=0, columnspacing=1.5,
              handletextpad=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def write_csv(path, agg, systems, datasets):
    """One row per system/sequence/metric. Memory rows are MiB; n_kfs is a count."""
    rows = ['system,dataset,metric,mean,std,n_runs']
    for dataset in datasets:
        for system in systems:
            e = agg.get((system, dataset))
            if not e:
                continue
            for metric in ('cpu_avg', 'cpu_peak', 'gpu_avg', 'gpu_peak'):
                mean, std = e[metric]
                rows.append(f'{system},{dataset},{metric},{mean:.2f},{std:.2f},'
                            f'{e["n_runs"]}')
            if e.get('n_kfs'):
                mean, std = e['n_kfs']
                rows.append(f'{system},{dataset},n_kfs,{mean:.1f},{std:.1f},'
                            f'{e["n_runs"]}')
    with open(path, 'w') as f:
        f.write('\n'.join(rows) + '\n')


# The series collect() aggregates, as opposed to METRICS above, which is the pair
# of bars the figure draws.
SERIES = ('cpu_avg', 'cpu_peak', 'gpu_avg', 'gpu_peak')


def _draw_bars(ax, agg, systems, datasets):
    """The thin horizontal bars for one platform, into an existing axes."""
    rows = [(system, metric) for system in systems for metric in METRICS]
    height, gap = 0.80 / len(rows), 0.06
    extent = len(rows) * height + (len(rows) - 1) * gap
    for ri, (system, metric) in enumerate(rows):
        hatch = SYSTEM_HATCH[systems.index(system) % len(SYSTEM_HATCH)] \
            if len(systems) > 1 else None
        for di, dataset in enumerate(datasets):
            entry = agg.get((system, dataset))
            if not entry:
                continue
            y = di + ri * (height + gap) - extent / 2 + height / 2
            mean, std = entry[f'{metric["key"]}_peak']
            # The bar is the mean of the runs' own peaks, so the spread across
            # those runs belongs on it. Caps are small: at this bar height a
            # default cap is taller than the bar it sits on.
            ax.barh(y, mean, height, color=metric['color'], hatch=hatch,
                    edgecolor=SURFACE, linewidth=0, zorder=3,
                    xerr=std if std > 0 else None,
                    error_kw=dict(ecolor=INK_MUTED, elinewidth=0.7, capsize=1.6,
                                  capthick=0.7, zorder=4))


def annotate_keyframes(ax, agg, systems, datasets):
    """The map size for each sequence, set just past the end of its longest bar.

    Reported as mean and spread over the same runs the bars average, so the two
    numbers on a row are talking about the same five runs.

    Memory is dominated by how much map is resident, so the keyframe count is what
    makes the bar lengths mean something. It rides at the end of the row rather
    than under the name: the names are already at 7.5pt and a second line each
    would double the height of a figure that is 39 rows tall. Returns the furthest
    right anything was drawn, so the caller can leave room for it.
    """
    reach = 0.0
    for di, dataset in enumerate(datasets):
        entry = next((agg[(s, dataset)] for s in systems if (s, dataset) in agg),
                     None)
        if not entry or not entry.get('n_kfs'):
            continue
        # Clear the error bar, not just the bar.
        end = max(sum(entry[f'{m["key"]}_peak']) for m in METRICS)
        mean, std = entry['n_kfs']
        text = f'{mean:,.0f} ± {std:,.0f} KF' if std else f'{mean:,.0f} KF'
        ax.text(end * 1.02, di, text, va='center', ha='left', fontsize=6.5,
                color=INK_MUTED)
        reach = max(reach, end)
    return reach


def _style_h(ax, datasets, labels=True):
    """Grid, spines and ticks for a sideways bar axes."""
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.grid(axis='y', visible=False)
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('#d6d5d0')
    ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=0)
    ax.set_xlim(left=0)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets if labels else [], fontsize=7.5, color=INK)
    ax.set_ylim(-0.7, len(datasets) - 0.3)
    ax.invert_yaxis()             # first sequence at the top, reading downward


def _legend_handles(systems):
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=m['color']) for m in METRICS]
    labels = [m['label'] for m in METRICS]
    if len(systems) > 1:
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor='#b8b7b3',
                                  hatch=SYSTEM_HATCH[i % len(SYSTEM_HATCH)],
                                  edgecolor=SURFACE)
                    for i, _ in enumerate(systems)]
        labels += list(systems)
    return handles, labels


def figure_horizontal(agg, systems, datasets, path):
    """Every sequence as its own thin horizontal bar, sequences down the page.

    Thirty-nine sequences will not fit across a page as upright bars -- the names
    alone need more width than the figure has -- so the axes are turned on their
    side: the length of the sheet absorbs the sequence count, the names read
    horizontally at their natural size, and the figure stays narrow enough for a
    column. Bars are thin because there are a lot of them, not for looks: at this
    density a thick bar leaves no gap between one sequence and the next.
    """
    fig, ax = plt.subplots(figsize=(5.4, 0.185 * len(datasets) + 1.05))
    fig.patch.set_facecolor(SURFACE)
    _draw_bars(ax, agg, systems, datasets)
    _style_h(ax, datasets)
    ax.set_xlabel('Peak Memory (MiB)', color=INK_MUTED, fontsize=9.5)

    # Room on the right for the keyframe counts, which sit outside the bars.
    reach = annotate_keyframes(ax, agg, systems, datasets)
    if reach:
        ax.set_xlim(0, reach * 1.30)

    handles, labels = _legend_handles(systems)
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.005),
              ncol=1 if len(labels) > 2 else len(labels), frameon=False,
              fontsize=8.5, labelcolor=INK_MUTED, handlelength=1.1,
              handleheight=1.1, borderpad=0, columnspacing=1.2, handletextpad=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def figure_horizontal_panels(panels, systems, datasets, path):
    """Both platforms in one figure: a panel each, sharing the sequence axis.

    Four bars to a row -- two metrics on two platforms -- would need the rows three
    times as tall before the bars separated, which is most of a page. A panel per
    platform keeps the rows as they are and spends width instead, and the sequence
    names are written once down the left. The two panels share an x limit, so a bar
    in one can be read against a bar in the other; without that the eye would
    compare lengths drawn to different scales.
    """
    fig, axes = plt.subplots(1, len(panels), sharey=True,
                             figsize=(2.7 * len(panels) + 1.2,
                                      0.185 * len(datasets) + 1.30))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)

    span = max(entry[f'{m["key"]}_peak'][0]
               for _, agg in panels for entry in agg.values() for m in METRICS)
    for ax, (label, agg) in zip(axes, panels):
        _draw_bars(ax, agg, systems, datasets)
        _style_h(ax, datasets)
        ax.set_xlim(0, span * 1.05)
        ax.set_title(label, color=INK, fontsize=10, fontweight='semibold')
        # The panels keep their platform titles: with two side by side the reader
        # has to be told which is which.

    handles, labels = _legend_handles(systems)
    fig.supxlabel('Peak Memory (MiB)', color=INK_MUTED, fontsize=9.5)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False, fontsize=8.5, labelcolor=INK_MUTED,
               handlelength=1.1, handleheight=1.1, borderpad=0, columnspacing=1.5,
               handletextpad=0.6)

    fig.tight_layout(rect=(0, 0.015, 1, 0.965))
    fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def group_max(agg, systems):
    """Collapse sequences into families, keeping the largest value in each.

    A peak is already a worst case over a run, so the family's number is the worst
    case over its sequences rather than their mean: the figure then answers "how
    much does a Room sequence need", which an average across a family of six would
    understate. Each metric takes its own maximum, so a family's CPU and GPU bars
    can come from different sequences -- which is what a memory envelope is. The
    sequence behind each one is returned so it can be named on stdout.
    """
    grouped, source = {}, {}
    for (system, dataset), entry in agg.items():
        label = group_of(dataset)
        if label is None:
            continue
        key = (system, label)
        best = grouped.setdefault(key, {'n_runs': 0, 'n_kfs': None})
        for metric in SERIES + ('n_kfs',):
            value = entry.get(metric)
            if value is None:
                continue
            if best.get(metric) is None or value[0] > best[metric][0]:
                best[metric] = value
                source.setdefault(key, {})[metric] = dataset
        best['n_runs'] += entry['n_runs']
    return grouped, source


def ratio_report(gathered, systems, datasets, out_csv=None):
    """How much more memory the desktop holds than the Jetson, per sequence.

    Reported as a geometric mean, since these are ratios: the arithmetic mean of
    a set of ratios is pulled around by whichever sequence happens to be largest,
    and is not the number whose reciprocal is the mean of the reciprocals. The
    per-sequence spread is printed alongside, because a summary of ratios that
    range over a factor of two is worth distrusting.
    """
    platforms = dict(gathered)
    if 'desktop' not in platforms or 'jetson' not in platforms:
        print('need both platforms for a ratio; run without --platform')
        return 1

    rows = []
    for dataset in datasets:
        for system in systems:
            desk = platforms['desktop'].get((system, dataset))
            jet = platforms['jetson'].get((system, dataset))
            if not desk or not jet:
                continue
            ratios = {}
            for metric in METRICS:
                key = f'{metric["key"]}_peak'
                if jet[key][0] > 0:
                    ratios[metric['key']] = desk[key][0] / jet[key][0]
            rows.append((dataset, system, desk, jet, ratios))

    if not rows:
        print('no sequence has both platforms')
        return 1

    print(f'{"sequence":13}{"desk CPU":>10}{"jet CPU":>9}{"ratio":>7}'
          f'{"desk GPU":>11}{"jet GPU":>9}{"ratio":>7}')
    print('-' * 66)
    for dataset, _, desk, jet, ratios in rows:
        print(f'{dataset:13}{desk["cpu_peak"][0]:10.0f}{jet["cpu_peak"][0]:9.0f}'
              f'{ratios.get("cpu", float("nan")):7.2f}'
              f'{desk["gpu_peak"][0]:11.0f}{jet["gpu_peak"][0]:9.0f}'
              f'{ratios.get("gpu", float("nan")):7.2f}')

    print()
    for metric in METRICS:
        values = [r[metric['key']] for _, _, _, _, r in rows if metric['key'] in r]
        if not values:
            continue
        geo = float(np.exp(np.mean(np.log(values))))
        print(f'{metric["label"]:22} desktop / Jetson = {geo:.2f}x  '
              f'(geometric mean of {len(values)} sequences, '
              f'range {min(values):.2f}-{max(values):.2f})')

    # By family too: the sweep is dominated by whichever family has most sequences.
    print()
    print(f'{"family":14}{"CPU":>7}{"GPU":>7}{"n":>4}')
    for label in GROUP_ORDER:
        members = [r for d, _, _, _, r in rows if group_of(d) == label]
        if not members:
            continue
        cells = ''
        for metric in METRICS:
            vals = [m[metric['key']] for m in members if metric['key'] in m]
            cells += f'{float(np.exp(np.mean(np.log(vals)))):7.2f}' if vals else f'{"--":>7}'
        print(f'{label:14}{cells}{len(members):4}')

    if out_csv:
        lines = ['sequence,system,metric,desktop_mib,jetson_mib,ratio']
        for dataset, system, desk, jet, ratios in rows:
            for metric in METRICS:
                key = metric['key']
                if key in ratios:
                    lines.append(f'{dataset},{system},{key}_peak,'
                                 f'{desk[f"{key}_peak"][0]:.2f},'
                                 f'{jet[f"{key}_peak"][0]:.2f},{ratios[key]:.4f}')
        with open(out_csv, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f'\nwrote {out_csv}')
    return 0


def build(args, platform, agg, datasets):
    """Draw the figure for one platform. Returns the number of figures written."""
    source = {}
    if args.group:
        agg, source = group_max(agg, args.systems)
        datasets = GROUP_ORDER
    datasets = [d for d in datasets if any((s, d) in agg for s in args.systems)]
    if not datasets:
        print('  no memory data found')
        return 0

    for dataset in datasets:
        for system in args.systems:
            if (system, dataset) not in agg:
                print(f'  warning: no memory data for {system} on {dataset}')
    print(f'  {len(datasets)} {"groups" if args.group else "sequences"} · '
          f'{", ".join(args.systems)} · '
          f'{sum(e["n_runs"] for e in agg.values())} runs')
    # Which sequence set each bar says nothing about on the figure itself.
    for dataset in datasets:
        for system in args.systems:
            got = source.get((system, dataset))
            if got:
                print(f'    {dataset}: cpu {got["cpu_peak"]}, gpu {got["gpu_peak"]}')

    name = ('memory_usage_grouped.png' if args.group else
            'memory_usage_all.png' if args.all_sequences else 'memory_usage.png')
    os.makedirs(args.out, exist_ok=True)
    path = out_path(args.out, name, platform)
    draw = figure_horizontal if args.all_sequences else figure
    draw(agg, args.systems, datasets, path)
    print(f'  wrote {path}')
    if args.csv:
        csv_path = out_path(args.out, name.replace('.png', '.csv'), platform)
        write_csv(csv_path, agg, args.systems, datasets)
        print(f'  wrote {csv_path}')
    return 1


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
    ap.add_argument('--all-sequences', action='store_true',
                    help='every sequence found, as thin horizontal bars down a '
                         'narrow figure instead of upright bars across a wide one')
    ap.add_argument('--group', action='store_true',
                    help='one bar per sequence family, holding the largest value '
                         'any sequence in it reached')
    ap.add_argument('--ratio', action='store_true',
                    help='report desktop/Jetson peak-memory ratios per sequence '
                         'and their geometric mean, instead of drawing anything')
    ap.add_argument('--combined', action='store_true',
                    help='every platform in one figure, a panel each sharing the '
                         'sequence axis, instead of a figure per platform')
    ap.add_argument('--systems', nargs='*', default=[CONTENDER],
                    help='systems to plot, in this order; more than one adds a legend')
    ap.add_argument('--machine', help="machine directory to read; overrides --platform's")
    ap.add_argument('--nitro-kernel', help='kernel-status directory to read for '
                                           'Nitro-SLAM, when the tree holds several')
    ap.add_argument('--out', default=OUT_ROOT, help='directory to write the figure into')
    ap.add_argument('--csv', action='store_true',
                    help='also write the per-metric table beside the figure')
    args = ap.parse_args()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})

    # Read every platform first: the quartile ranking is shared across them, so it
    # cannot be computed until all of their numbers are in.
    gathered, found = [], []
    for name, roots, machine in select_platforms(args):
        agg, datasets = collect(roots, args.systems, machine,
                                None if (args.all_sequences or args.group
                                         or args.ratio)
                                else args.sequences,
                                args.nitro_kernel)
        if not agg:
            print(f'{name}: no memory data found')
            continue
        gathered.append((name, agg))
        for d in datasets:
            if d not in found:
                found.append(d)
    if not gathered:
        return 1

    datasets = found if (args.all_sequences or args.group or args.ratio) else \
        [d for d in args.sequences if d in found]

    if args.ratio:
        return ratio_report(gathered, args.systems, datasets,
                            os.path.join(args.out, 'memory_ratio.csv')
                            if args.csv else None)

    if args.combined:
        panels = [(PLATFORMS[name]['label'], agg) for name, agg in gathered]
        drawn = [d for d in datasets
                 if any((s, d) in agg for _, agg in panels for s in args.systems)]
        if not drawn:
            print('no memory data found')
            return 1
        os.makedirs(args.out, exist_ok=True)
        stem = ('memory_usage_all' if args.all_sequences else 'memory_usage')
        path = os.path.join(args.out, f'{stem}_combined.png')
        print(f'{" + ".join(label for label, _ in panels)}: {len(drawn)} sequences')
        figure_horizontal_panels(panels, args.systems, drawn, path)
        print(f'  wrote {path}')
        return 0

    written = 0
    for name, agg in gathered:
        print(f'{name}:')
        written += build(args, name, agg, datasets)
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
