#!/usr/bin/env python3
"""Recompute ATE from stored trajectories under different alignment settings.

Reads the trajectories a batch run already produced and re-runs evaluate3.py
against ground truth with a sweep of --max_difference (association search
radius, ns) and/or --offset (modelled sensor delay, ns), then reports the
median ATE per sequence for each setting.

Nothing is written into the results tree: evaluate3.py is called without
--plot/--save, and the only output is the CSV named by --out. The stored
ostream.txt values are left untouched, so this can be re-run freely.

  ./reeval_ate.py                                   # magistrale at 2000 ms
  ./reeval_ate.py --sequences slides1 slides2
  ./reeval_ate.py --offset 0 50000000 --max-difference 20000000
"""
import argparse, csv, os, re, subprocess, statistics, sys
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.abspath(__file__))
GT = os.path.expanduser('~/SLAM/Datasets/tumvi/dataset-{seq}_512_16/mav0/mocap0/data.csv')
EVAL = os.path.join(REPO, 'evaluation', 'evaluate3.py')

RE_RMSE = re.compile(r'absolute_translational_error\.rmse\s+([0-9.eE+-]+)')
RE_PAIRS = re.compile(r'compared_pose_pairs\s+(\d+)')


def stored_rmse(run_dir):
    """The ATE the batch run itself recorded, for comparison."""
    try:
        with open(os.path.join(run_dir, 'ostream.txt')) as fh:
            m = RE_RMSE.search(fh.read())
        return float(m.group(1)) if m else None
    except OSError:
        return None


def find_runs(results, seq):
    """Yield (system, iteration, trajectory_path) for every stored run."""
    for system in ('ORB-SLAM3', 'Nitro-SLAM'):
        base = os.path.join(results, system)
        if not os.path.isdir(base):
            continue
        # ORB-SLAM3/<version>/<seq>/<i>, Nitro-SLAM/<kernels>/<version>/<seq>/<i>
        for root, dirs, files in os.walk(base):
            if os.path.basename(root) != 'trajectory':
                continue
            traj = os.path.join(root, f'f_dataset-{seq}_stereoi.txt')
            if not os.path.isfile(traj):
                continue
            run = os.path.dirname(root)
            if os.path.basename(os.path.dirname(run)) != seq:
                continue
            yield system, os.path.basename(run), traj, run


def evaluate(job):
    seq, system, it, traj, run_dir, max_diff, offset = job
    gt = GT.format(seq=seq)
    if not os.path.isfile(gt):
        return dict(seq=seq, system=system, iteration=it, max_difference=max_diff,
                    offset=offset, rmse=None, pairs=None,
                    stored_rmse=stored_rmse(run_dir), error='no ground truth')
    cmd = [sys.executable, '-W', 'ignore', EVAL, gt, traj,
           '--max_difference', str(max_diff), '--offset', str(offset), '--verbose']
    env = dict(os.environ, MPLBACKEND='Agg')   # evaluate3 imports pyplot
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             env=env, cwd=REPO).stdout
    except subprocess.TimeoutExpired:
        return dict(seq=seq, system=system, iteration=it, max_difference=max_diff,
                    offset=offset, rmse=None, pairs=None,
                    stored_rmse=stored_rmse(run_dir), error='timeout')
    m, p = RE_RMSE.search(out), RE_PAIRS.search(out)
    return dict(seq=seq, system=system, iteration=it, max_difference=max_diff,
                offset=offset,
                rmse=float(m.group(1)) if m else None,
                pairs=int(p.group(1)) if p else None,
                stored_rmse=stored_rmse(run_dir),
                error='' if m else 'no rmse in output')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='Results-tumvi')
    ap.add_argument('--sequences', nargs='*',
                    default=[f'magistrale{i}' for i in range(1, 7)])
    ap.add_argument('--max-difference', nargs='*', type=int,
                    default=[2000000000],
                    help='association search radius in ns (evaluate3 default: 20000000)')
    ap.add_argument('--offset', nargs='*', type=int, default=[0],
                    help='modelled sensor delay in ns added to the estimate (default: 0)')
    ap.add_argument('--out', default='reeval_ate.csv')
    ap.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()

    jobs = [(seq, system, it, traj, run, md, off)
            for seq in args.sequences
            for system, it, traj, run in find_runs(args.results, seq)
            for md in args.max_difference
            for off in args.offset]
    if not jobs:
        print(f'no stored trajectories found under {args.results}')
        return 1
    print(f'{len(jobs)} evaluations over {len(args.sequences)} sequences '
          f'({args.jobs} workers)')

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(evaluate, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f'  {i}/{len(jobs)}', flush=True)

    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {args.out}')

    # Median per (sequence, system, setting), against the ATE the batch recorded.
    def _med(vals):
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    def pick(seq, system, setting, key):
        return _med([r[key] for r in rows if r['seq'] == seq and r['system'] == system
                     and (r['max_difference'], r['offset']) == setting])

    settings = [(md, off) for md in args.max_difference for off in args.offset]
    for system in ('ORB-SLAM3', 'Nitro-SLAM'):
        print(f'\n=== {system}: median ATE (m) ===')
        head = f'{"sequence":<13}{"recorded":>12}'
        head += ''.join(f'{f"md={md//1000000}ms/off={off//1000000}ms":>26}'
                        for md, off in settings)
        print(head)
        for seq in args.sequences:
            base = pick(seq, system, settings[0], 'stored_rmse')
            line = f'{seq:<13}' + (f'{base:>12.4f}' if base else f'{"-":>12}')
            for st in settings:
                m, pr = pick(seq, system, st, 'rmse'), pick(seq, system, st, 'pairs')
                if m is None:
                    line += f'{"-":>26}'
                else:
                    d = f' ({(m - base) / base * 100:+.1f}%)' if base else ''
                    line += f'{f"{m:.4f}{d} [{pr:.0f}]":>26}'
            print(line)

    print('\n[N] = median associated pose pairs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
