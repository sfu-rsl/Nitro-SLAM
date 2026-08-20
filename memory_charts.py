#!/usr/bin/env python3
"""Plot GPU and CPU memory across a sweep of runs, from monitor_memory.py output.

Reads the per-run memory.csv written alongside each run by run_timing_batch.sh and
produces, for each sequence:

  * a GPU memory time series with every individual run drawn, ORB-SLAM3 against
    Nitro-SLAM;
  * the same for CPU resident set size;
  * peak GPU and peak CPU bars per sequence and system.

Peaks come from the sampled series for GPU and from the kernel's VmHWM for CPU, so
the CPU peak is exact rather than sampling-limited.

Usage:
    ./memory_charts.py --version bench
    ./memory_charts.py --version bench --out figures --gpu-metric proc
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from timing_charts import (INK, INK_MUTED, SLOTS, SURFACE, SYSTEM_ORDER, find_runs)

SYSTEM_COLOR = {'ORB-SLAM3': SLOTS[0], 'Nitro-SLAM': SLOTS[1]}


def read_series(path):
    """Load one memory.csv. Trailing samples where RSS has fallen to zero are the
    process being unmapped at exit, not real occupancy, so they are dropped."""
    if not os.path.isfile(path):
        return None
    cols = defaultdict(list)
    seen = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                blank = v in ('', None)
                cols[k].append(0.0 if blank else float(v))
                if not blank:
                    seen.add(k)
    if not cols.get('t_s'):
        return None
    rss = cols['cpu_rss_mib']
    end = len(rss)
    while end > 1 and rss[end - 1] <= 0.0:
        end -= 1
    # A column blank in every row is a reading the sampler could not take -- the
    # NVML gpu_* ones on a Jetson, say. Dropping it makes it absent rather than a
    # flat line at zero, so collect() skips the run instead of plotting a fiction.
    return {k: np.asarray(v[:end]) for k, v in cols.items() if k in seen}


def read_summary(path):
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


def collect(runs, gpu_metric):
    """{(system, dataset): [ {t, gpu, cpu, gpu_peak, cpu_peak}, ... ]} in run order."""
    gpu_col = {'proc': 'gpu_proc_mib', 'delta': 'gpu_dev_delta_mib',
               'device': 'gpu_dev_used_mib'}[gpu_metric]
    out = defaultdict(list)
    for r in sorted(runs, key=lambda r: (r['system'], r['dataset'], r['iteration'])):
        s = read_series(os.path.join(r['path'], 'memory.csv'))
        if s is None or gpu_col not in s:
            continue
        summary = read_summary(os.path.join(r['path'], 'memory_summary.txt'))
        out[(r['system'], r['dataset'])].append(dict(
            iteration=r['iteration'],
            t=s['t_s'], gpu=s[gpu_col], cpu=s['cpu_rss_mib'],
            gpu_peak=float(np.max(s[gpu_col])) if len(s[gpu_col]) else 0.0,
            # VmHWM is the kernel's high-water mark; fall back to the sampled max.
            cpu_peak=summary.get('cpu_hwm_rss_mib',
                                 float(np.max(s['cpu_rss_mib'])) if len(s['cpu_rss_mib']) else 0.0),
        ))
    return out


def style(ax, ylabel, xlabel=None):
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e2e1dd', linewidth=0.8)
    ax.grid(axis='x', visible=False)
    ax.set_facecolor(SURFACE)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#d6d5d0')
    ax.tick_params(colors=INK_MUTED, labelsize=9.5, length=0)


def header(fig, title, subtitle):
    """Title and subtitle placed a fixed number of inches from the top edge, and the
    axes rect shrunk to match -- figure-fraction offsets drift with figure height."""
    h = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=13, color=INK, fontweight='semibold',
                 x=0.005, y=1 - 0.24 / h, ha='left', va='top')
    fig.text(0.005, 1 - 0.50 / h, subtitle, fontsize=9.5, color=INK_MUTED,
             ha='left', va='top')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.72 / h))


def figure_timeseries(data, systems, datasets, metric, out_path, title, ylabel):
    """Small multiples: one panel per sequence, every run drawn as its own line."""
    panels = [d for d in datasets if any((s, d) in data for s in systems)]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels) + 1.6, 4.4),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, dataset in zip(axes[0], panels):
        for system in systems:
            runs = data.get((system, dataset), [])
            for i, run in enumerate(runs):
                # Every run is drawn; the darker overlay is the pointwise median,
                # so a single outlying run stays visible instead of being averaged away.
                ax.plot(run['t'], run[metric], color=SYSTEM_COLOR[system],
                        linewidth=0.9, alpha=0.35, zorder=2,
                        label=None)
            if runs:
                grid = np.linspace(0, min(r['t'][-1] for r in runs), 600)
                stack = np.vstack([np.interp(grid, r['t'], r[metric]) for r in runs])
                ax.plot(grid, np.median(stack, axis=0), color=SYSTEM_COLOR[system],
                        linewidth=2.0, zorder=3, label=f'{system}  (n={len(runs)})')
        ax.set_title(dataset, fontsize=11.5, color=INK, fontweight='semibold', loc='left')
        style(ax, ylabel, 'time since process start (s)')
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_MUTED, loc='lower right')
    header(fig, title,
           'thin lines are individual runs; heavy line is the pointwise median')
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


def figure_peaks(data, systems, datasets, out_path, n_total):
    """Peak GPU and peak CPU as two panels -- never one chart with two y-scales."""
    panels = [d for d in datasets if any((s, d) in data for s in systems)]
    if not panels:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(5.5 * 2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    width, gap = 0.34, 0.04
    handles = {}
    for ax, (key, label) in zip(axes, [('gpu_peak', 'peak GPU memory (MiB)'),
                                       ('cpu_peak', 'peak CPU RSS (MiB)')]):
        top = 0.0
        for gi, dataset in enumerate(panels):
            present = [s for s in systems if (s, dataset) in data]
            span = len(present) * width + (len(present) - 1) * gap
            for si, system in enumerate(present):
                vals = [r[key] for r in data[(system, dataset)]]
                x = gi + si * (width + gap) - span / 2 + width / 2
                m = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                b = ax.bar(x, m, width, color=SYSTEM_COLOR[system], edgecolor=SURFACE,
                           linewidth=1.2, zorder=3)
                handles.setdefault(system, b)
                if sd:
                    ax.errorbar(x, m, yerr=sd, color=INK_MUTED, capsize=3,
                                linewidth=1.0, zorder=4)
                # Individual runs over the bar, so spread is visible, not just implied.
                ax.scatter([x] * len(vals), vals, s=9, color=SURFACE,
                           edgecolors=INK_MUTED, linewidths=0.7, zorder=5)
                ax.text(x, m + sd, f'{m:.0f}', ha='center', va='bottom',
                        fontsize=9, color=INK_MUTED)
                top = max(top, m + sd, max(vals))
        # One tick per sequence at the group centre; the legend carries the system.
        ax.set_xticks(range(len(panels)))
        ax.set_xticklabels(panels, fontsize=11, color=INK)
        ax.set_xlim(-0.75, len(panels) - 0.25)
        style(ax, label)
        ax.set_ylim(0, top * 1.16)
    # A single figure-level legend, clear of every bar.
    fig.legend([handles[s] for s in handles], list(handles), loc='upper right',
               bbox_to_anchor=(0.995, 1.0), frameon=False, fontsize=9.5,
               labelcolor=INK_MUTED, ncol=len(handles), handlelength=1.1)
    header(fig, f'Peak memory · mean of {n_total} runs per sequence',
           'dots are individual runs; bars are the mean, whiskers 1 s.d. '
           'CPU peak is the kernel VmHWM, so it is exact.')
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


def write_csv(path, data):
    rows = ['system,dataset,iteration,gpu_peak_mib,cpu_peak_rss_mib,duration_s,samples']
    for (system, dataset), runs in sorted(data.items()):
        for r in runs:
            rows.append(f'{system},{dataset},{r["iteration"]},{r["gpu_peak"]:.1f},'
                        f'{r["cpu_peak"]:.1f},{r["t"][-1]:.1f},{len(r["t"])}')
    with open(path, 'w') as f:
        f.write('\n'.join(rows) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='Results')
    ap.add_argument('--version', required=True)
    ap.add_argument('--out', default='figures')
    ap.add_argument('--datasets', nargs='*')
    ap.add_argument('--gpu-metric', default='proc',
                    choices=['proc', 'delta', 'device'],
                    help='proc: per-process, from NVML or jtop (default). delta: '
                         'device-wide minus the pre-run baseline. device: raw '
                         'device-wide total.')
    args = ap.parse_args()

    runs = find_runs(args.results, args.version)
    if not runs:
        print(f'no runs found under {args.results} for version "{args.version}"')
        return 1
    data = collect(runs, args.gpu_metric)
    if not data:
        available = sorted({m for m in ('proc', 'delta', 'device') if collect(runs, m)})
        if available:
            print(f'no run has data for gpu metric "{args.gpu_metric}"; '
                  f'try --gpu-metric {available[0]} (available: {", ".join(available)}). '
                  f'Empty gpu_* columns mean the sampler found no GPU source -- on a '
                  f'Jetson that is jtop.service being down, or the user not being in '
                  f'the "jtop" group; check gpu_source in memory_summary.txt.')
        else:
            print('no memory.csv found in any run -- was the batch run with the sampler?')
        return 1

    systems = [s for s in SYSTEM_ORDER if any(k[0] == s for k in data)]
    datasets = args.datasets or sorted({k[1] for k in data})
    n_total = max(len(v) for v in data.values())
    print(f'{sum(len(v) for v in data.values())} runs with memory data · '
          f'gpu metric: {args.gpu_metric}')

    os.makedirs(args.out, exist_ok=True)
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})

    figure_timeseries(data, systems, datasets, 'gpu',
                      os.path.join(args.out, 'memory_gpu_timeseries.png'),
                      'GPU memory over time', 'GPU memory (MiB)')
    figure_timeseries(data, systems, datasets, 'cpu',
                      os.path.join(args.out, 'memory_cpu_timeseries.png'),
                      'CPU resident set size over time', 'CPU RSS (MiB)')
    figure_peaks(data, systems, datasets,
                 os.path.join(args.out, 'memory_peaks.png'), n_total)
    write_csv(os.path.join(args.out, 'memory_summary.csv'), data)
    for n in ('memory_gpu_timeseries.png', 'memory_cpu_timeseries.png',
              'memory_peaks.png', 'memory_summary.csv'):
        print(f'  wrote {os.path.join(args.out, n)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
