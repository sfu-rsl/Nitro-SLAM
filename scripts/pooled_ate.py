#!/usr/bin/env python3
"""ATE over every run of a sequence at once, rather than over one chosen run.

median_trajectory.py reports the ATE of a single representative run, which makes
the table describe an actual trajectory on disk but leaves the number at the mercy
of which of the five runs happened to sit in the middle. This pools instead: the
per-pose errors of all of a sequence's runs are treated as one sample and a single
RMSE is taken over the lot.

The pooled RMSE does not need the error vectors themselves. Writing run i's error
terms as e_ij, its reported RMSE is r_i = sqrt(mean_j e_ij^2) over n_i pose pairs,
so sum_j e_ij^2 = n_i * r_i^2 and

    pooled = sqrt( sum_i n_i r_i^2 / sum_i n_i )

which is exactly the RMSE of the concatenated error vectors. Both r_i and n_i are
already in every run's ostream.txt ("absolute_translational_error.rmse" and
"compared_pose_pairs"), so this reads the same summaries median_trajectory.py does
and never re-aligns a trajectory. --verify recomputes the concatenation the long
way, from the trajectory files through evo, to check that claim on real runs.

What pooling is and is not:

  * it uses all five runs rather than one, so it does not move when a different run
    lands in the middle -- the sampling noise of picking a representative is gone;
  * it is a quadratic mean of the per-run RMSEs, weighted by pose count, and a
    quadratic mean is dominated by its largest term. One catastrophic run pulls the
    pooled number most of the way to itself. It is less robust than the median, not
    more.

So the "dominance" column is reported alongside: the share of the pooled sum of
squares contributed by the single worst run. At five equal runs it sits near 0.2;
anything above ~0.6 means the number is mostly one run, and the pooled value should
not be read as typical of the system.

Runs that lost tracking early contribute fewer pose pairs and are weighted down
accordingly, which understates them -- a caveat inherent to pooling, not to this
implementation.

Usage:
    ./pooled_ate.py Results-euroc
    ./pooled_ate.py Results-euroc data/jetson/Results-euroc --latex
    ./pooled_ate.py Results-tumvi --csv pooled.csv --tex pooled.tex
    ./pooled_ate.py Results-euroc --verify 8
"""
import argparse
import csv
import math
import os
import re
import statistics
import sys

from median_trajectory import (ATE_RE, find_sequences, keep, latex_table,
                               read_ate, run_key, trajectory_files)

# evaluate3.py prints the number of associated pose pairs just above the error
# statistics; it is the n_i the pooling weights each run by.
PAIRS_RE = re.compile(r"compared_pose_pairs\s+(\d+)\s+pairs")


def read_pairs(run_dir):
    """Number of associated pose pairs for one run, or None if it has none."""
    log = os.path.join(run_dir, "ostream.txt")
    if not os.path.exists(log):
        return None
    with open(log, errors="replace") as fh:
        m = PAIRS_RE.search(fh.read())
    return int(m.group(1)) if m else None


def read_run(run_dir, metric="rmse"):
    """(rmse, n_pairs) for one run, or None when either is missing.

    A run without both cannot take part in the pooling: the RMSE alone does not say
    how much of the sample it speaks for.
    """
    ate = read_ate(run_dir, metric)
    pairs = read_pairs(run_dir)
    if ate is None or pairs is None or pairs <= 0:
        return None
    return ate, pairs


def pool(runs):
    """Pooled statistics for one sequence's runs.

    `runs` is [(rmse, n_pairs)]. Returns a dict carrying the pooled RMSE, the
    median-run RMSE for comparison, and the share of the pooled sum of squares that
    the single largest contributor accounts for.
    """
    sq = [n * r * r for r, n in runs]           # each run's sum of squared error
    total_sq, total_n = sum(sq), sum(n for _, n in runs)
    pooled = math.sqrt(total_sq / total_n)
    values = sorted(r for r, _ in runs)
    return {
        "pooled_rmse": pooled,
        "median_rmse": statistics.median(values),
        "mean_rmse": statistics.fmean(values),
        "min_rmse": values[0],
        "max_rmse": values[-1],
        "dominance": max(sq) / total_sq if total_sq > 0 else 0.0,
        "n_runs": len(runs),
        "n_pairs": total_n,
    }


def error_vector(run_dir, sequence, dataset_root):
    """The per-pose error vector of one run, recomputed from its trajectory.

    Only --verify needs this; it is the expensive path the closed form exists to
    avoid, and it pulls in evo, so the import is deferred to here.
    """
    import copy

    import numpy as np
    from evo.core import lie_algebra, sync
    from evo.tools import file_interface

    import trajectory_plots as tp

    gt_file = tp.groundtruth_path(sequence, dataset_root)
    if not gt_file:
        return None
    files = [f for f in trajectory_files(run_dir) if f.endswith(".txt")
             and os.path.basename(f).startswith("f_")]
    if not files:
        return None
    ref = file_interface.read_euroc_csv_trajectory(gt_file)
    est = tp.read_estimate(files[0])
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=tp.MAX_DIFF)
    r, t, _ = copy.deepcopy(est_s).align(ref_s, correct_scale=False)
    matched = copy.deepcopy(est_s)
    matched.transform(lie_algebra.se3(r, t))
    return np.linalg.norm(matched.positions_xyz - ref_s.positions_xyz, axis=1)


def verify(sequences, dataset_root, limit):
    """Check the closed form against a real concatenation on `limit` sequences.

    Rebuilds each sequence's concatenated error vector from its trajectory files,
    takes the RMSE of that directly, and compares it with the value computed from
    the per-run summaries.
    """
    import numpy as np

    print(f"verifying the closed form on up to {limit} sequences\n")
    print(f'  {"sequence":44}{"closed form":>13}{"concatenated":>14}{"rel err":>10}')
    checked = worst = 0
    for name in sorted(sequences):
        if checked >= limit:
            break
        run_dirs = sorted(sequences[name], key=run_key)
        runs = [r for r in (read_run(d) for d in run_dirs) if r]
        if len(runs) != len(run_dirs) or not runs:
            continue
        vectors = [error_vector(d, name.rpartition("/")[2], dataset_root)
                   for d in run_dirs]
        if any(v is None for v in vectors):
            continue
        concat = np.concatenate(vectors)
        direct = float(np.sqrt(np.mean(concat ** 2)))
        closed = pool(runs)["pooled_rmse"]
        rel = abs(direct - closed) / direct if direct else 0.0
        worst = max(worst, rel)
        checked += 1
        print(f"  {name:44}{closed:13.4f}{direct:14.4f}{rel:9.2%}")
    if not checked:
        print("  no sequence had both summaries and trajectories to check")
        return 1
    print(f"\n  {checked} sequence(s) checked, worst relative difference {worst:.2%}")
    # The two disagree only where evo's association picks different pose pairs than
    # evaluate3.py did, which moves n_i and r_i slightly; the identity itself is
    # exact.
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["Results-euroc"],
                    help="results roots to scan (default: Results-euroc)")
    ap.add_argument("--metric", default="rmse", choices=sorted(ATE_RE),
                    help="per-run statistic to pool (default: rmse)")
    ap.add_argument("--only", metavar="PATTERN", action="append", default=[],
                    help="keep only sequences matching this glob; repeatable")
    ap.add_argument("--exclude", metavar="PATTERN", action="append", default=[],
                    help="drop sequences matching this glob; repeatable")
    ap.add_argument("--csv", metavar="FILE", help="write the full table to FILE")
    ap.add_argument("--latex", action="store_true",
                    help="print a booktabs table of the pooled values")
    ap.add_argument("--tex", metavar="FILE",
                    help="write the LaTeX table to FILE as well (implies --latex)")
    ap.add_argument("--decimals", type=int, default=3, metavar="N",
                    help="decimal places in the LaTeX table (default: 3)")
    ap.add_argument("--no-bold", action="store_true",
                    help="with --latex, do not bold the best value in each row")
    ap.add_argument("--dominance-warn", type=float, default=0.6, metavar="F",
                    help="flag sequences where one run contributes more than this "
                         "share of the pooled sum of squares (default: 0.6)")
    ap.add_argument("--verify", type=int, nargs="?", const=8, metavar="N",
                    help="recompute N sequences the long way, from trajectories, "
                         "to check the closed form (slow; needs evo)")
    ap.add_argument("--dataset-root", default="/root/SLAM/Datasets",
                    help="root holding the ground truth trees, for --verify")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-sequence report")
    args = ap.parse_args()
    args.latex = args.latex or bool(args.tex)

    sequences = find_sequences(args.roots)
    sequences = {k: v for k, v in sequences.items()
                 if keep(k.rpartition("/")[2], args.only, args.exclude)}
    if not sequences:
        print(f"no runs (directories with an ostream.txt) under "
              f"{', '.join(args.roots)}", file=sys.stderr)
        return 1

    if args.verify is not None:
        return verify(sequences, args.dataset_root, args.verify)

    rows, flagged = [], []
    for name in sorted(sequences):
        run_dirs = sorted(sequences[name], key=run_key)
        runs = [r for r in (read_run(d, args.metric) for d in run_dirs) if r]
        if not runs:
            continue
        stats = pool(runs)
        stats["sequence"] = name
        stats["runs_failed"] = len(run_dirs) - len(runs)
        rows.append(stats)
        if stats["dominance"] > args.dominance_warn:
            flagged.append(stats)

    if not rows:
        print("no run had both an ATE and a pose-pair count", file=sys.stderr)
        return 1

    if not args.quiet and not args.latex:
        print(f'{"sequence":48}{"pooled":>9}{"median":>9}{"mean":>9}'
              f'{"min":>9}{"max":>9}{"dom":>7}{"runs":>6}')
        for r in rows:
            print(f'{r["sequence"]:48}{r["pooled_rmse"]:9.3f}{r["median_rmse"]:9.3f}'
                  f'{r["mean_rmse"]:9.3f}{r["min_rmse"]:9.3f}{r["max_rmse"]:9.3f}'
                  f'{r["dominance"]:7.2f}{r["n_runs"]:6d}')

    if flagged:
        print(f'\n{len(flagged)} sequence(s) where one run contributes more than '
              f'{args.dominance_warn:.0%} of the pooled sum of squares:')
        for r in flagged:
            print(f'  {r["sequence"]:48} dominance {r["dominance"]:.2f}  '
                  f'pooled {r["pooled_rmse"]:.3f} vs median {r["median_rmse"]:.3f}')

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        fields = ["sequence", "pooled_rmse", "median_rmse", "mean_rmse", "min_rmse",
                  "max_rmse", "dominance", "n_runs", "n_pairs", "runs_failed"]
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in fields})
        print(f"wrote {args.csv} ({len(rows)} sequence(s))")

    if args.latex:
        body = latex_table(rows, "pooled_rmse", not args.no_bold, args.decimals)
        print(body)
        if args.tex:
            os.makedirs(os.path.dirname(args.tex) or ".", exist_ok=True)
            with open(args.tex, "w") as fh:
                fh.write(body + "\n")
            print(f"\nwrote {args.tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
