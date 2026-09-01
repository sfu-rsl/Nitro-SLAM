#!/usr/bin/env python3
"""Ground truth vs ORB-SLAM3 vs Nitro-SLAM, for the median run of each sequence.

One figure per sequence, drawn with evo: ground truth as a dashed black line, the
two systems over it. The run plotted is the one median_trajectory.py picks -- the
median-ATE run of that sequence's five -- so the picture and the ATE table describe
the same run rather than a flattering one.

Estimates are aligned to ground truth the way evaluate3.py aligns them when it
computes the reported ATE: Umeyama rotation and translation with the scale left
alone, since stereo-inertial observes scale. The transform is fitted on the poses
that have ground truth and then applied to the whole trajectory, so the drawn path
is the entire run rather than only its matched part.

Ground truth is drawn in segments, not as one line. TUM-VI only has mocap where the
sequence is inside the motion-capture room: room6 is covered throughout, but
magistrale4 and outdoors7 start and end there and have nothing in between, so a
single line would invent a path across the gap. The covered fraction is printed for
each sequence.

Usage:
    ./trajectory_plots.py
    ./trajectory_plots.py --sequences room6 outdoors7
    ./trajectory_plots.py --platform jetson
"""

import argparse
import copy
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evo.core import lie_algebra, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
# evo.tools.plot applies its own configured backend at import, which is TkAgg by
# default and needs a display this container has no tkinter for. Point evo at the
# same headless backend matplotlib is already on, before importing it.
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'
from evo.tools import plot as evo_plot

import median_trajectory as mt
from tracking_breakdown import GRID, INK, INK_MUTED, OUT_ROOT, PLATFORMS, SLOTS, SURFACE
from tracking_comparison import BASELINE, CONTENDER, resolve_system

SEQUENCES = ['room6', 'magistrale4', 'outdoors7']
DATASET_ROOT = '/root/SLAM/Datasets'
# Poses further apart in time than this are a hole in the ground truth rather than
# a sparse stretch of it, and are not joined up when drawing.
GT_GAP_S = 1.0
# evaluate3.py associates on timestamps with this tolerance (seconds).
MAX_DIFF = 0.02

GT_COLOR = '#111111'
# Loosely dashed, on/off ink in points -- matplotlib's own (5, 10). Ground truth is
# drawn on top of the estimates, so anything denser reads as a solid black line laid
# over them and hides what the figure is for; a long gap marks the reference without
# covering the two paths being compared.
GT_DASHES = (5.0, 10.0)
SYSTEM_COLOR = {BASELINE: SLOTS[0], CONTENDER: SLOTS[1]}
# evo.tools.plot.traj takes only style/colour/label, so weight and stacking order
# are set on the line it just drew.
GT_WEIGHT, SYSTEM_WEIGHT = (0.9, 4), (1.2, 2)


def read_estimate(path):
    """The estimated trajectory, with timestamps in seconds.

    Both examples' mains write the stamp straight through as nanoseconds, and
    evaluate3.py compares against ground truth in the same units, so nothing there
    ever had to convert. evo's TUM reader takes seconds, and its EuRoC reader has
    already divided the ground truth down, so the estimate is converted here to
    meet it -- otherwise nothing associates at all.
    """
    traj = file_interface.read_tum_trajectory_file(path)
    if traj.timestamps.size and traj.timestamps[0] > 1e15:
        traj = PoseTrajectory3D(poses_se3=traj.poses_se3,
                                timestamps=traj.timestamps / 1e9)
    return traj


def groundtruth_path(sequence, dataset_root):
    """The mocap/state-estimate CSV for one sequence, or None if it is not there."""
    candidates = [
        os.path.join(dataset_root, 'tumvi', f'dataset-{sequence}_512_16',
                     'mav0', 'mocap0', 'data.csv'),
        os.path.join(dataset_root, 'EuRoc', sequence,
                     'mav0', 'state_groundtruth_estimate0', 'data.csv'),
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def median_run(roots, sequence, frag, machine, metric='rmse', agg='median'):
    """(trajectory file, ATE) of the sequence's median run, or (None, None)."""
    for root in roots:
        seq_dir = os.path.join(root, frag, machine, sequence)
        if not os.path.isdir(seq_dir):
            continue
        runs = [os.path.join(seq_dir, d) for d in sorted(os.listdir(seq_dir))
                if os.path.isfile(os.path.join(seq_dir, d, 'ostream.txt'))]
        value, run_dir, _, _ = mt.aggregate_runs(runs, metric, agg)
        if run_dir is None:
            continue
        found = [f for f in glob.glob(os.path.join(run_dir, 'trajectory', 'f_*.txt'))]
        if found:
            return found[0], value
    return None, None


def gt_segments(traj, gap_s=GT_GAP_S):
    """Contiguous stretches of ground truth, split at holes, and the covered share.

    Each stretch comes back as its own trajectory so it can be drawn as a separate
    line: TUM-VI has mocap only inside the capture room, and joining the two ends of
    an outdoor sequence would draw several hundred metres of path that was never
    measured. The covered fraction says which case a figure is in.
    """
    stamps = traj.timestamps
    dt = np.diff(stamps)
    bounds = np.concatenate(([0], np.flatnonzero(dt > gap_s) + 1, [len(stamps)]))
    segments = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop - start > 1:
            segment = copy.deepcopy(traj)
            segment.reduce_to_ids(list(range(int(start), int(stop))))
            segments.append(segment)
    span = stamps[-1] - stamps[0]
    covered = span - float(dt[dt > gap_s].sum())
    return segments, (covered / span if span > 0 else 1.0)


def aligned(est, ref):
    """`est` rigidly aligned to `ref`, in full, plus its ATE on the matched poses.

    The fit uses only the poses ground truth covers -- that is all evaluate3.py has
    to work with -- but the transform is then applied to every pose, so a sequence
    with ground truth at its two ends still draws its whole path.
    """
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=MAX_DIFF)
    r, t, s = copy.deepcopy(est_s).align(ref_s, correct_scale=False)

    full = copy.deepcopy(est)
    full.transform(lie_algebra.se3(r, t))

    matched = copy.deepcopy(est_s)
    matched.transform(lie_algebra.se3(r, t))
    err = np.linalg.norm(matched.positions_xyz - ref_s.positions_xyz, axis=1)
    return full, float(np.sqrt(np.mean(err ** 2))), len(err)


AXES_OF = {evo_plot.PlotMode.xy: (0, 1), evo_plot.PlotMode.xz: (0, 2),
           evo_plot.PlotMode.yz: (1, 2)}
# Every figure is this size, whatever shape its trajectory is, so the three sit
# together in a row without one dwarfing the others. The top strip is reserved for
# the legend; the sequence and its ATE belong to the caption and the table, so they
# are not repeated on the figure.
FIGSIZE = (4.0, 4.6)
HEADER_FRACTION = 0.90
# --grid packs the sequences into one figure instead. Each panel is this size, and
# the panels are titled, since a shared caption can no longer say which is which.
PANEL_SIZE = (3.4, 3.4)
GRID_HEADER_IN = 0.45
# Blank margin around the trajectory, as a share of its longer side.
MARGIN = 0.06


def plot_mode_for(xyz):
    """The projection that shows the most of this trajectory.

    Picks the two axes it spreads over furthest, so a corridor sequence is not
    drawn edge-on into a line.
    """
    spread = xyz.max(axis=0) - xyz.min(axis=0)
    keep = tuple(sorted(np.argsort(spread)[-2:]))
    return next(m for m, a in AXES_OF.items() if a == keep)


def set_limits(ax, mode, paths):
    """Frame everything drawn, with a margin, keeping metres square.

    The canvas is a fixed size for every sequence, and the shapes are not: room6 is
    a squarish scribble, magistrale4 more than twice as tall as it is wide. Setting
    the limits to the data and leaving the aspect equal with adjustable='datalim'
    lets matplotlib widen whichever axis has slack, so the scale stays honest --
    a metre is a metre on both axes -- and the box is filled rather than the figure
    being reshaped around the trajectory.
    """
    i, j = AXES_OF[mode]
    pts = np.vstack([p.positions_xyz for p in paths])
    pad = MARGIN * max(np.ptp(pts[:, i]), np.ptp(pts[:, j]))
    ax.set_xlim(pts[:, i].min() - pad, pts[:, i].max() + pad)
    ax.set_ylim(pts[:, j].min() - pad, pts[:, j].max() + pad)
    ax.set_aspect('equal', adjustable='datalim')


def draw_panel(fig, subplot_arg, gt, estimates):
    """One sequence into one axes: ground truth dashed, a line per system over it.

    Returns (axes, share of the sequence ground truth covers).
    """
    mode = plot_mode_for(gt.positions_xyz)
    segments, covered = gt_segments(gt)
    ax = evo_plot.prepare_axis(fig, mode, subplot_arg)

    def draw(traj, style, color, label, weight, dashes=None):
        evo_plot.traj(ax, mode, traj, style=style, color=color, label=label)
        ax.lines[-1].set_linewidth(weight[0])
        ax.lines[-1].set_zorder(weight[1])
        if dashes:
            ax.lines[-1].set_dashes(dashes)

    for i, segment in enumerate(segments):
        draw(segment, '--', GT_COLOR,
             'Ground truth' if i == 0 else '_nolegend_', GT_WEIGHT, GT_DASHES)

    for name, (traj, _, _) in estimates.items():
        draw(traj, '-', SYSTEM_COLOR[name], name, SYSTEM_WEIGHT)

    set_limits(ax, mode, segments + [t for t, _, _ in estimates.values()])

    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#d6d5d0')
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    # evo.tools.plot.traj puts a framed legend on the axes as soon as a label is
    # given. Drop it: the single-panel figure replaces it with its own, and the grid
    # carries one legend for all the panels rather than the same three entries four
    # times over the trajectories.
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return ax, covered


def figure(sequence, gt, estimates, out_file):
    """One sequence, alone on its own canvas."""
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(SURFACE)
    ax, covered = draw_panel(fig, 111, gt, estimates)
    # The legend goes in a strip reserved at the top of the figure, filled from the
    # figure's own top down. Anchoring it to the axes instead would pin it to
    # whatever height the trajectory's shape gave the axes, leaving the reserved
    # strip empty above it -- and 'best' placement inside the axes has no free
    # corner to find on a path that fills its own bounding box.
    fig.tight_layout(rect=(0, 0, 1, HEADER_FRACTION))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED,
              loc='upper left', bbox_to_anchor=(0.05, 0.99),
              bbox_transform=fig.transFigure, ncol=1,
              handlelength=1.6, borderpad=0, labelspacing=0.35)
    fig.savefig(out_file, dpi=400, facecolor=SURFACE)
    plt.close(fig)
    return covered


def grid_figure(panels, cols, out_file):
    """Every sequence in one figure, `cols` per row, each panel titled.

    A single canvas cannot carry one caption per trajectory, so the sequence name
    goes above its panel; the legend is drawn once for the whole figure, since all
    the panels use the same three lines.
    """
    rows = -(-len(panels) // cols)
    fig = plt.figure(figsize=(PANEL_SIZE[0] * cols,
                              PANEL_SIZE[1] * rows + GRID_HEADER_IN))
    fig.patch.set_facecolor(SURFACE)

    handles, labels = [], []
    for i, (sequence, gt, estimates) in enumerate(panels, start=1):
        ax, _ = draw_panel(fig, rows * 100 + cols * 10 + i, gt, estimates)
        ax.set_title(sequence, color=INK, fontsize=10.5, fontweight='semibold',
                     pad=6)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    header = GRID_HEADER_IN / (PANEL_SIZE[1] * rows + GRID_HEADER_IN)
    fig.tight_layout(rect=(0, 0, 1, 1 - header), h_pad=1.6, w_pad=1.6)
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_MUTED,
               loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               handlelength=1.6, borderpad=0, columnspacing=2.0)
    fig.savefig(out_file, dpi=400, facecolor=SURFACE)
    plt.close(fig)


def load_sequence(args, roots, machine, sequence):
    """(ground truth, {system: (aligned trajectory, ATE, matched poses)}).

    Returns (None, {}) when the sequence has no ground truth, so a caller can step
    over it rather than failing the whole run.
    """
    gt_file = groundtruth_path(sequence, args.dataset_root)
    if not gt_file:
        print(f'{sequence}: no ground truth under {args.dataset_root}')
        return None, {}
    gt = file_interface.read_euroc_csv_trajectory(gt_file)

    estimates = {}
    for system in args.systems:
        frag = resolve_system(roots, system, machine, args.nitro_kernel)
        if frag is None:
            print(f'  warning: no runs for {system}')
            continue
        traj_file, reported = median_run(roots, sequence, frag, machine, args.metric)
        if not traj_file:
            print(f'  warning: {system} has no median run for {sequence}')
            continue
        traj, ate, n = aligned(read_estimate(traj_file), gt)
        estimates[system] = (traj, ate, n)
        # Tolerance is relative: evo's association and associate3.py's differ
        # slightly in which pose pairs they choose where ground truth is sparse,
        # which moves a 17 m error by centimetres and a 5 mm error by nothing.
        note = '' if abs(ate - reported) <= max(5e-3, 0.01 * reported) else \
            f'  !! differs from reported {reported:.3f} m'
        print(f'  {sequence:12} {system:11} ATE {ate:.3f} m over {n} poses{note}')
    return gt, estimates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--platform', default='desktop', choices=sorted(PLATFORMS),
                    help='platform whose runs to plot (default: desktop)')
    ap.add_argument('--sequences', nargs='*', default=SEQUENCES,
                    help='sequences to plot, one figure each')
    ap.add_argument('--systems', nargs='*', default=[BASELINE, CONTENDER],
                    help='systems to draw over the ground truth')
    ap.add_argument('--nitro-kernel', help='kernel-status directory to read for '
                                           'Nitro-SLAM, when the tree holds several')
    ap.add_argument('--dataset-root', default=DATASET_ROOT,
                    help='root holding the tumvi/ and EuRoc/ dataset trees')
    ap.add_argument('--metric', default='rmse', choices=sorted(mt.ATE_RE),
                    help='ATE statistic the median run is chosen by')
    ap.add_argument('--out', default=OUT_ROOT, help='directory to write the figures into')
    ap.add_argument('--grid', action='store_true',
                    help='draw every sequence into one figure of subplots, titled, '
                         'instead of one figure each')
    ap.add_argument('--cols', type=int, default=2, metavar='N',
                    help='columns in the --grid layout (default: 2); sequences fill '
                         'the rows in the order given')
    ap.add_argument('--name', default='trajectory_grid',
                    help='basename for the --grid figure')
    args = ap.parse_args()

    cfg = PLATFORMS[args.platform]
    roots, machine = cfg['roots'], cfg['machine']
    # evo sets savefig.bbox='tight' when its plot module is imported, which crops
    # every figure back to its own content and undoes the fixed canvas size. The
    # header strip is reserved explicitly, so nothing needs cropping.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE,
                         'savefig.bbox': None})
    os.makedirs(args.out, exist_ok=True)

    panels, written = [], 0
    for sequence in args.sequences:
        gt, estimates = load_sequence(args, roots, machine, sequence)
        if gt is None or not estimates:
            continue
        if args.grid:
            panels.append((sequence, gt, estimates))
            continue
        out_file = os.path.join(args.out,
                                f'trajectory_{sequence}_{args.platform}.png')
        covered = figure(sequence, gt, estimates, out_file)
        print(f'  wrote {out_file}  (ground truth covers {covered * 100:.0f}% '
              f'of the sequence)')
        written += 1

    if args.grid and panels:
        out_file = os.path.join(args.out, f'{args.name}_{args.platform}.png')
        grid_figure(panels, args.cols, out_file)
        print(f'  wrote {out_file}  ({len(panels)} panels, '
              f'{args.cols} per row)')
        written += 1

    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
