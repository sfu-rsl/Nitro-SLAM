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
SEQUENCES by default, light to heavy; --all-sequences restores the full sweep.

Both platforms are drawn by default, into one directory with the platform in the
filename -- analysis_out/memory_usage_{desktop,jetson}.png. The GPU reading is not
the same measurement on the two: the desktop's comes from NVML as a discrete
device's used memory, the Jetson's from jtop, where the iGPU allocates out of the
same system RAM the CPU bar is counting.

Usage:
    ./memory_usage.py
    ./memory_usage.py --all-sequences
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
from tracking_comparison import BASELINE, CONTENDER, resolve_system

# Colour carries the metric, since that is what the two bars in a group differ by.
# A second system, if one is asked for, is carried by fill -- the same convention
# the comparison charts use.
METRICS = [
    dict(key='cpu', label='CPU (resident set size)', color=SLOTS[0]),
    dict(key='gpu', label='GPU (device over baseline)', color=SLOTS[1]),
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

    ax.set_ylabel('peak memory (MiB)', color=INK_MUTED, fontsize=9.5)
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


def build(args, platform, agg, datasets):
    """Draw the figure for one platform. Returns the number of figures written."""
    datasets = [d for d in datasets if any((s, d) in agg for s in args.systems)]
    if not datasets:
        print('  no memory data found')
        return 0

    for dataset in datasets:
        for system in args.systems:
            if (system, dataset) not in agg:
                print(f'  warning: no memory data for {system} on {dataset}')
    print(f'  {len(datasets)} sequences · {", ".join(args.systems)} · '
          f'{sum(e["n_runs"] for e in agg.values())} runs')

    os.makedirs(args.out, exist_ok=True)
    path = out_path(args.out, 'memory_usage.png', platform)
    figure(agg, args.systems, datasets, path)
    print(f'  wrote {path}')
    if args.csv:
        csv_path = out_path(args.out, 'memory_usage.csv', platform)
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
                    help='draw every sequence found instead')
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
                                None if args.all_sequences else args.sequences,
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

    datasets = found if args.all_sequences else \
        [d for d in args.sequences if d in found]

    written = 0
    for name, agg in gathered:
        print(f'{name}:')
        written += build(args, name, agg, datasets)
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
