#!/usr/bin/env python3
"""Align and evaluate stored trajectories with evo.

Bridges the two formats used here: ground truth is EuRoC CSV (ns timestamps,
qw first), the estimate is TUM-style but with timestamps in NANOSECONDS rather
than seconds, so neither of evo's readers pairs them as-is.

Alignment is Umeyama SE3 with scale FIXED at 1.0 by default -- the right metric
for a stereo-inertial system, and the quantity evaluate3.py reports as
absolute_translational_error.rmse. --sim3 aligns with scale correction instead
(and plots the rescaled trajectory).

Reads stored trajectories only; writes nothing into the results tree.

  ./outdoors_eval/evo_eval.py outdoors5 --run 0 --compare ORB-SLAM3
  ./outdoors_eval/evo_eval.py outdoors5 --run 0 --compare ORB-SLAM3 --sim3
"""
import argparse, copy, glob, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.expanduser('~/SLAM/Datasets/tumvi/dataset-{seq}_512_16/mav0/mocap0/data.csv')
COMPARE_COLORS = ['tab:blue', 'tab:green', 'tab:orange']

from evo.core import sync, metrics, lie_algebra
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
# evo.tools.plot forces the backend from evo's own settings (TkAgg by default,
# which needs tkinter). Pin it before importing that module.
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'
from evo.tools import plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_est_ns(path):
    """TUM layout (t x y z qx qy qz qw) with nanosecond timestamps."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line[0] in '#%':
            continue
        tok = line.split()
        if len(tok) >= 8:
            rows.append([float(v) for v in tok[:8]])
    a = np.array(rows)
    q = a[:, 4:8]                                       # qx qy qz qw
    return PoseTrajectory3D(
        positions_xyz=a[:, 1:4],
        orientations_quat_wxyz=np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]]),
        timestamps=a[:, 0] / 1e9)                       # ns -> s


def find_runs(results, system, seq):
    """Every stored run for this system/sequence, as {run_index: trajectory path}."""
    pat = os.path.join(REPO, results, system, '**', 'desktop', seq, '*',
                       'trajectory', f'f_dataset-{seq}_stereoi.txt')
    out = {}
    for h in glob.glob(pat, recursive=True):
        try:
            out[int(os.path.basename(os.path.dirname(os.path.dirname(h))))] = h
        except ValueError:
            continue
    return dict(sorted(out.items()))


def find_traj(results, system, seq, run):
    pat = os.path.join(REPO, results, system, '**', 'desktop', seq, str(run),
                       'trajectory', f'f_dataset-{seq}_stereoi.txt')
    hits = glob.glob(pat, recursive=True)
    return hits[0] if hits else None


def segments(stamps, gap):
    """Index ranges of contiguous ground truth, split where sampling stops for `gap` s."""
    brk = np.nonzero(np.diff(stamps) > gap)[0]
    edges = [0] + [i + 1 for i in brk] + [len(stamps)]
    return list(zip(edges[:-1], edges[1:]))


def evaluate(traj_ref, traj_full, max_diff, sim3, align_segment=0, gap=1.0):
    """Align to ground truth and score.

    align_segment=0 fits over every matched pose (what evo/evaluate3 do by
    default). align_segment=N fits using only the Nth contiguous ground-truth
    block, then applies that transform to the whole trajectory -- which is the
    honest fit when the blocks are separated by accumulated drift.
    """
    ref_s, est_s = sync.associate_trajectories(traj_ref, traj_full, max_diff=max_diff)
    segs = segments(ref_s.timestamps, gap)

    if align_segment:
        a, b = segs[align_segment - 1]
        ref_fit = copy.deepcopy(ref_s); ref_fit.reduce_to_ids(range(a, b))
        est_fit = copy.deepcopy(est_s); est_fit.reduce_to_ids(range(a, b))
        r_a, t_a, s = est_fit.align(ref_fit, correct_scale=sim3)
        aligned = copy.deepcopy(est_s)
        if sim3:
            aligned.scale(s)
        aligned.transform(lie_algebra.se3(r_a, t_a))
    else:
        aligned = copy.deepcopy(est_s)
        r_a, t_a, s = aligned.align(copy.deepcopy(ref_s), correct_scale=sim3)

    err = np.linalg.norm(aligned.positions_xyz - ref_s.positions_xyz, axis=1)
    rmse = lambda e: float(np.sqrt((e ** 2).mean()))
    stats = dict(rmse=rmse(err), mean=float(err.mean()), median=float(np.median(err)),
                 std=float(err.std()), min=float(err.min()), max=float(err.max()))
    per_seg = [(i + 1, b - a, rmse(err[a:b])) for i, (a, b) in enumerate(segs)]

    full = copy.deepcopy(traj_full)
    if sim3:
        full.scale(s)
    full.transform(lie_algebra.se3(r_a, t_a))
    return stats, est_s, full, s, ref_s.num_poses, per_seg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sequence')
    ap.add_argument('--run', type=int, default=0)
    ap.add_argument('--median', action='store_true',
                    help='evaluate every stored run per system and plot the one with '
                         'the median ATE, instead of --run')
    ap.add_argument('--system', default='Nitro-SLAM')
    ap.add_argument('--compare', nargs='*', default=None,
                    help='also plot these systems, each aligned on its own')
    ap.add_argument('--results', default='Results-tumvi')
    ap.add_argument('--max-diff', type=float, default=0.02, help='association window, s')
    ap.add_argument('--sim3', action='store_true',
                    help='align with scale correction instead of fixing scale at 1.0')
    ap.add_argument('--align-segment', type=int, default=0, metavar='N',
                    help='fit the alignment using only the Nth contiguous ground-truth '
                         'block (1-based); default 0 fits over all matched poses')
    ap.add_argument('--gap', type=float, default=1.0,
                    help='ground-truth gap, in s, that separates blocks (default 1)')
    ap.add_argument('--out', default='outdoors_eval')
    args = ap.parse_args()

    gt_path = GT.format(seq=args.sequence)
    traj_ref = file_interface.read_euroc_csv_trajectory(gt_path)
    mode = 'Sim3 (scale corrected)' if args.sim3 else 'SE3 (scale fixed 1.0)'
    if args.align_segment:
        mode += f', fitted on GT block {args.align_segment}'
    print(f'ref: {gt_path}  ({traj_ref.num_poses} poses)')
    print(f'alignment: {mode}\n')

    series = []
    for i, sysname in enumerate([args.system] + list(args.compare or [])):
        if args.median:
            runs = find_runs(args.results, sysname, args.sequence)
            if not runs:
                print(f'no stored runs for {sysname} {args.sequence}')
                continue
            scored = []
            for idx, p in runs.items():
                r = evaluate(traj_ref, read_est_ns(p), args.max_diff, args.sim3,
                             args.align_segment, args.gap)
                scored.append((r[0]['rmse'], idx, p, r))
            scored.sort()
            chosen = scored[(len(scored) - 1) // 2]        # lower median for even counts
            print(f'{sysname}: ATE per run  ' + '  '.join(
                f'{("*" if idx == chosen[1] else "")}{idx}={rm:.3f}'
                for rm, idx, _, _ in sorted(scored, key=lambda x: x[1])) +
                f'   -> median run {chosen[1]}')
            run_idx, path, res = chosen[1], chosen[2], chosen[3]
        else:
            run_idx = args.run
            path = find_traj(args.results, sysname, args.sequence, run_idx)
            if not path:
                print(f'no stored trajectory for {sysname} {args.sequence} run {run_idx}')
                continue
            res = evaluate(traj_ref, read_est_ns(path), args.max_diff, args.sim3,
                           args.align_segment, args.gap)
        st, matched, full, s, pairs, per_seg = res
        colour = 'red' if i == 0 else COMPARE_COLORS[(i - 1) % len(COMPARE_COLORS)]
        series.append((f'{sysname} (run {run_idx})', full, st, s, colour))
        print(f'{sysname} run {run_idx}   ({pairs} pairs)')
        for k in ('rmse', 'mean', 'median', 'std', 'min', 'max'):
            print(f'    {k:<7}{st[k]:10.4f} m')
        flag = '   <-- alignment is fitting across a discontinuity' if abs(s - 1) > 0.05 else ''
        print(f'    scale  {s:10.4f}{flag}')
        for n, cnt, r in per_seg:
            mark = '  <- fitted here' if n == args.align_segment else ''
            print(f'      segment {n} ({cnt:5d} poses)  rmse {r:9.4f} m{mark}')
        print()
    if not series:
        return 1

    os.makedirs(args.out, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)
    plot.traj(ax, plot.PlotMode.xy, traj_ref, '-', 'black', 'ground truth')
    for nm, full, st, s, colour in series:
        lbl = f'{nm}  (ATE {st["rmse"]:.2f} m'
        lbl += f', scale {s:.3f})' if args.sim3 else ')'
        plot.traj(ax, plot.PlotMode.xy, full, '--', colour, lbl)
    which = 'median-ATE run per system' if args.median else f'run {args.run}'
    ax.set_title(f'{args.sequence}, {which} -- {mode}', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()

    stem = (f'{args.sequence}_{"median" if args.median else f"run{args.run}"}'
            f'_evo_{"sim3" if args.sim3 else "se3"}')
    if args.align_segment:
        stem += f'_seg{args.align_segment}'
    if args.compare:
        stem += '_' + '_'.join(n.lower() for n in ([args.system] + list(args.compare)))
    out = os.path.join(args.out, stem + '.png')
    fig.savefig(out, dpi=115)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
