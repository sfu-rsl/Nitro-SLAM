#!/usr/bin/env python3
"""Per-pose trajectory error distributions, ORB-SLAM3 against Nitro-SLAM.

Two boxes per sequence, one per system, each summarising the per-pose translational
errors of every run of that sequence pooled together. Where the ATE tables reduce a
sequence to one number per system, this draws the distribution those numbers came
from, so "the two systems are equally accurate" is something the reader can see
rather than infer from a pair of medians.

Each run is aligned to ground truth on its own -- Umeyama rotation and translation
with scale left alone, exactly as evaluate3.py does it -- and the residuals are then
concatenated across runs, which is the same pooling pooled_ate.py reduces to a single
RMSE.

The axis is logarithmic because the sequences span four orders of magnitude, from
millimetres in room6 to tens of metres outdoors; a linear axis would collapse every
indoor sequence onto zero. Whiskers are the 5th and 95th percentiles rather than
1.5 IQR, and outliers are not drawn: pooled over five runs a sequence carries tens of
thousands of poses, so a conventional flier cloud would be a solid bar the height of
the axis and would say nothing the whisker does not.

A caution on reading it: per-pose errors within one run are strongly autocorrelated,
being successive samples of one drifting trajectory, so they are nowhere near
independent. The boxes are descriptive -- they show the shape and overlap of the
error distributions. Any significance claim belongs at the run or sequence level,
where the samples actually are independent; a KS or rank test run over tens of
thousands of pooled poses would report a vanishing p-value for a difference of no
consequence.

Usage:
    ./error_strips.py <desktop euroc root> --dataset euroc
    ./error_strips.py <root> ... --out analysis_out/v2 --cache /tmp/errcache
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from median_trajectory import find_sequences, run_key, trajectory_files
from tracking_breakdown import GRID, INK, INK_MUTED, SLOTS, SURFACE
from tracking_comparison import BASELINE, CONTENDER, GROUP_ORDER, group_of

SYSTEM_COLOR = {BASELINE: SLOTS[0], CONTENDER: SLOTS[1]}
SYSTEM_ORDER = [BASELINE, CONTENDER]
# Families in axis order, so the sequences group the way every other figure groups.
FAMILIES = ['MH', 'V1', 'V2', 'room', 'corridor', 'magistrale', 'outdoors', 'slides']
# Errors are positive but can round to zero; the axis is logarithmic, so anything
# below this is drawn at this value rather than dropped.
FLOOR_M = 1e-4


def family_key(sequence):
    for i, prefix in enumerate(FAMILIES):
        if sequence.startswith(prefix):
            return (i, sequence)
    return (len(FAMILIES), sequence)


def system_of(variant):
    """Which system a variant path belongs to, or None if it is neither."""
    head = variant.split('/')[0]
    return head if head in SYSTEM_COLOR else None


def align_and_diff(traj_file, gt):
    """Per-pose translational errors of one run against an already-loaded `gt`.

    The same Umeyama fit evaluate3.py uses -- rotation and translation, scale left
    alone -- kept here rather than borrowed so that ground truth is read once per
    sequence instead of once per run, which is ten times the file reads over a sweep.
    """
    import copy

    from evo.core import lie_algebra, sync

    from trajectory_plots import MAX_DIFF, read_estimate

    est = read_estimate(traj_file)
    ref_s, est_s = sync.associate_trajectories(gt, est, max_diff=MAX_DIFF)
    r, t, _ = copy.deepcopy(est_s).align(ref_s, correct_scale=False)
    matched = copy.deepcopy(est_s)
    matched.transform(lie_algebra.se3(r, t))
    return np.linalg.norm(matched.positions_xyz - ref_s.positions_xyz, axis=1)


def errors_for(run_dir, gt, cache_dir, key):
    """Per-pose errors for one run, from the cache when it is already there.

    Recomputing means re-aligning every run, which is minutes of work for the whole
    sweep, so the vectors are kept and the plotting can be re-run freely.
    """
    path = os.path.join(cache_dir, key + '.npy') if cache_dir else None
    if path and os.path.isfile(path):
        return np.load(path)
    traj = [f for f in trajectory_files(run_dir)
            if f.endswith('.txt') and os.path.basename(f).startswith('f_')]
    if not traj:
        return None
    err = align_and_diff(traj[0], gt)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, err)
    return err


def collect(roots, dataset_root, cache_dir):
    """{sequence: {system: pooled error vector}} over every run under `roots`."""
    from evo.tools import file_interface

    from trajectory_plots import groundtruth_path

    sequences = find_sequences(roots)
    wanted = {}
    for key in sequences:
        variant, _, sequence = key.rpartition('/')
        system = system_of(variant)
        if system:
            wanted.setdefault(sequence, {})[system] = (variant, sequences[key])

    out = {}
    gt_cache = {}
    for sequence in sorted(wanted, key=family_key):
        if sequence not in gt_cache:
            gt_file = groundtruth_path(sequence, dataset_root)
            gt_cache[sequence] = (file_interface.read_euroc_csv_trajectory(gt_file)
                                  if gt_file else None)
        gt = gt_cache[sequence]
        if gt is None:
            print(f'  {sequence}: no ground truth; skipped')
            continue
        for system, (variant, run_dirs) in wanted[sequence].items():
            vectors = []
            for run_dir in sorted(run_dirs, key=run_key):
                key = os.path.join(variant, sequence, os.path.basename(run_dir))
                try:
                    err = errors_for(run_dir, gt, cache_dir, key)
                except Exception as exc:                      # noqa: BLE001
                    print(f'  {sequence}/{system}/{os.path.basename(run_dir)}: {exc}')
                    continue
                if err is not None and len(err):
                    vectors.append(err)
            if vectors:
                out.setdefault(sequence, {})[system] = np.concatenate(vectors)
        if sequence in out:
            got = out[sequence]
            bits = '  '.join(f'{s}: n={len(v)} med={np.median(v):.3f}'
                             for s, v in got.items())
            print(f'  {sequence:12} {bits}')
    return out


def regroup(data):
    """Pool each family's sequences into one entry, keyed by the family label.

    Sequences within a family differ in length and route but not in what is being
    measured, so one pair of boxes per family says what eight pairs of boxes say and
    leaves room for both datasets in a single figure. Pooling is by pose, so a long
    sequence carries more weight in its family than a short one -- the box describes
    the family's poses, not the average of its sequences.
    """
    out = {}
    for sequence, systems in data.items():
        label = group_of(sequence)
        if label is None:
            print(f'  no family for {sequence}; dropped')
            continue
        for system, values in systems.items():
            out.setdefault(label, {}).setdefault(system, []).append(values)
    return {label: {system: np.concatenate(chunks)
                    for system, chunks in systems.items()}
            for label, systems in out.items()}


def figure(data, out_file, title=None, order=None):
    """Two boxes per sequence, systems side by side, on a log error axis."""
    sequences = ([c for c in order if c in data] if order
                 else sorted(data, key=family_key))
    if not sequences:
        return False

    width = max(6.0, 0.62 * len(sequences) + 1.4)
    fig, ax = plt.subplots(figsize=(width, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    half = 0.19          # box centres sit this far either side of the tick
    for si, system in enumerate(SYSTEM_ORDER):
        values, positions = [], []
        for i, sequence in enumerate(sequences):
            v = data[sequence].get(system)
            if v is None or not len(v):
                continue
            values.append(np.maximum(v, FLOOR_M))
            positions.append(i + (si * 2 - 1) * half)
        if not values:
            continue
        colour = SYSTEM_COLOR[system]
        ax.boxplot(values, positions=positions, widths=0.30, whis=(5, 95),
                   showfliers=False, patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor=colour, edgecolor=colour, linewidth=0.8),
                   whiskerprops=dict(color=colour, linewidth=0.9),
                   capprops=dict(color=colour, linewidth=0.9),
                   medianprops=dict(color=INK, linewidth=1.3))

    ax.set_yscale('log')
    ax.set_ylabel('per-pose translational error (m)', color=INK)
    ax.set_xticks(range(len(sequences)))
    ax.set_xticklabels(sequences, rotation=45, ha='right', fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(sequences) - 0.4)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SYSTEM_COLOR[s],
                             edgecolor=SYSTEM_COLOR[s], label=s)
               for s in SYSTEM_ORDER]
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.3, label='median'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1.14),
              ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    if title:
        ax.set_title(title, color=INK, pad=26)

    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    fig.savefig(out_file, dpi=400, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='+', help='results roots to scan')
    ap.add_argument('--out', default='analysis_out', help='directory for the figure')
    ap.add_argument('--name', default='error_strips', help='figure basename')
    ap.add_argument('--platform', default='desktop', help='name used in the filename')
    ap.add_argument('--dataset-root', default='/root/SLAM/Datasets',
                    help='root holding the tumvi/ and EuRoc/ dataset trees')
    ap.add_argument('--cache', help='directory to keep per-run error vectors in')
    ap.add_argument('--title', help='title for the figure')
    ap.add_argument('--group', action='store_true',
                    help='pool each sequence family (Machine Hall, Vicon 1, Room, '
                         '...) into one pair of boxes instead of one pair per '
                         'sequence')
    args = ap.parse_args()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})
    data = collect(args.roots, args.dataset_root, args.cache)
    if not data:
        print('no error vectors computed', file=sys.stderr)
        return 1

    order = None
    if args.group:
        data = regroup(data)
        order = [g for g in GROUP_ORDER if g in data]

    label = 'family' if args.group else 'sequence'
    print(f'\n  {label:14}ORB med    NS med     ratio   ORB n    NS n')
    for sequence in (order or sorted(data, key=family_key)):
        d = data[sequence]
        if BASELINE not in d or CONTENDER not in d:
            continue
        a, b = np.median(d[BASELINE]), np.median(d[CONTENDER])
        print(f'  {sequence:14}{a:9.4f} {b:9.4f} {a / max(b, 1e-12):8.2f}x '
              f'{len(d[BASELINE]):7d} {len(d[CONTENDER]):7d}')

    path = os.path.join(args.out, f'{args.name}_{args.platform}.png')
    if figure(data, path, args.title, order):
        print(f'\nwrote {path}')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
