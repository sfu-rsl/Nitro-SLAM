#!/usr/bin/env python3
"""Tabulate the high-keyframe-insertion stress test (run_hikf.sh) against TurboMap Table II.

For every arm and sequence, reads each run's ostream.txt for
  absolute_translational_error.rmse   (ATE, m)
  [LBA skip] ... skipped_queue=N      (LBA + KF culling skipped, queue non-empty)
and reports the mean over runs, next to the numbers published in the TurboMap paper
(arXiv 2511.02036, Table II: avg. 5 runs, desktop setting).

skipped_queue is the paper's metric. skipped_stop (skips caused by a Loop Closing stop
request), aborted (LBAs cut short because Tracking requested a keyframe), kfs_processed,
and Tracking's keyframe decisions under a busy Local Mapping ([KF refuse]: frames where
a keyframe was deferred, and requests refused with 3+ queued) are shown with --detail.

Usage:
  scripts/hikf_table.py <label> [--mf 4] [--detail] [--csv out.csv]
e.g.
  scripts/hikf_table.py desktop
  scripts/hikf_table.py desktop --mf off          # the unpaced-only pilot
"""
import argparse
import glob
import os
import re
import statistics as st

ROOTS = {"MH01": "Results-euroc-hikf", "V101": "Results-euroc-hikf",
         "room1": "Results-tumvi-hikf", "corridor1": "Results-tumvi-hikf"}
SEQS = list(ROOTS)
# Arm name -> top-level directory written by run_script.sh.
ARMS = [("ORB-SLAM3", "ORB-SLAM3"),
        ("FastTrack", "FastTrack"),
        ("TurboMap", "TurboMap"),
        ("FastTrack+TurboMap", "FastTrack&TurboMap"),
        ("TurboMap+FastLoop", "TurboMap&FastLoop"),
        ("Nitro-SLAM", "Nitro-SLAM")]
# TurboMap paper, Table II: (ATE m, #skipped LBA & KF culling).
PAPER = {
    "TurboMap":  {"MH01": (0.040, 3.4), "V101": (0.039, 18.8),
                  "room1": (0.011, 5.8), "corridor1": (0.045, 9.2)},
    "ORB-SLAM3": {"MH01": (1.105, 58), "V101": (0.091, 102.6),
                  "room1": (1.005, 140.4), "corridor1": (10.136, 343.8)},
}

RE_ATE = re.compile(r"absolute_translational_error\.rmse\s+([\d.eE+-]+)")
# Frames lost during tracking never reach the trajectory file, so a run that loses
# tracking is scored on less of the sequence -- and can post a better ATE for it.
# compared_pose_pairs says how much was actually evaluated.
RE_PAIRS = re.compile(r"compared_pose_pairs\s+(\d+)")
RE_SKIP = re.compile(r"\[LBA skip\] kfs_processed=(\d+) skipped_queue=(\d+) skipped_stop=(\d+)(?: aborted=(\d+))?")
RE_KF = re.compile(r"\[KF refuse\] deferred_busy=(\d+) requested_busy=(\d+) refused_queue=(\d+)")


def parse_run(path):
    ate = skip = kf = pairs = None
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_ATE.search(line)
            if m:
                ate = float(m.group(1))
            m = RE_SKIP.search(line)
            if m:
                # (kfs_processed, skipped_queue, skipped_stop, aborted); aborted is None
                # in runs from before the counter existed.
                skip = tuple(int(x) if x is not None else None for x in m.groups())
            m = RE_PAIRS.search(line)
            if m:
                pairs = int(m.group(1))
            m = RE_KF.search(line)
            if m:
                # (deferred_busy, requested_busy, refused_queue); None before the counter.
                kf = tuple(int(x) for x in m.groups())
    return ate, skip, kf, pairs


def collect(arm_dir, seq, version):
    # The kernel-status level is present for GPU arms and absent for ORB-SLAM3.
    pats = [os.path.join(ROOTS[seq], arm_dir, version, seq, "*", "ostream.txt"),
            os.path.join(ROOTS[seq], arm_dir, "*", version, seq, "*", "ostream.txt")]
    runs = []
    for p in sorted(set(sum((glob.glob(x) for x in pats), []))):
        if not os.path.exists(os.path.join(os.path.dirname(p), ".batch_done")):
            continue
        ate, skip, kf, pairs = parse_run(p)
        runs.append({"ate": ate, "skip": skip, "kf": kf, "pairs": pairs, "path": p})
    return runs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("--mf", default="4", help="KF_MAX_FRAMES used by run_hikf.sh (default 4)")
    ap.add_argument("--detail", action="store_true", help="per-run values and stop-caused skips")
    ap.add_argument("--csv", help="also write the table as CSV")
    a = ap.parse_args()
    version = f"hikf-{a.label}-mf{a.mf}"

    # Denominator per sequence: the largest pose-pair count any configuration achieved,
    # i.e. a run that tracked the whole sequence.
    best_pairs = {}
    for _, d in ARMS:
        for seq in SEQS:
            for r in collect(d, seq, version):
                if r["pairs"]:
                    best_pairs[seq] = max(best_pairs.get(seq, 0), r["pairs"])

    rows = []
    for name, d in ARMS:
        for seq in SEQS:
            runs = collect(d, seq, version)
            if not runs:
                continue
            ates = [r["ate"] for r in runs]
            rows.append({
                "arm": name, "seq": seq, "n": len(runs),
                "n_ate": sum(x is not None for x in ates),
                "ate": mean(ates),
                "ate_med": st.median([x for x in ates if x is not None]) if any(x is not None for x in ates) else None,
                "ate_sd": st.stdev([x for x in ates if x is not None]) if sum(x is not None for x in ates) > 1 else None,
                "cov": mean([100.0 * r["pairs"] / best_pairs[seq]
                             if r["pairs"] and best_pairs.get(seq) else None for r in runs]),
                "skip_q": mean([r["skip"][1] if r["skip"] else None for r in runs]),
                "skip_s": mean([r["skip"][2] if r["skip"] else None for r in runs]),
                "kfs": mean([r["skip"][0] if r["skip"] else None for r in runs]),
                "abort": mean([r["skip"][3] if r["skip"] else None for r in runs]),
                "deferred": mean([r["kf"][0] if r["kf"] else None for r in runs]),
                "refused": mean([r["kf"][2] if r["kf"] else None for r in runs]),
                "runs": runs,
            })

    print(f"High keyframe insertion ({version}): ATE RMSE (m) mean/sd/median and #skipped LBA & KF culling (mean) over runs")
    hdr = f"{'arm':<18} {'seq':<10} {'n':>3} {'ATE':>8} {'±sd':>7} {'median':>8} {'cov%':>6} {'skipped':>8}"
    if a.detail:
        hdr += f" {'stop':>6} {'abort':>6} {'KFs':>7} {'defer':>7} {'refuse':>6}"
    hdr += f" | {'paper ATE':>9} {'paper skip':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        p = PAPER.get(r["arm"], {}).get(r["seq"])
        line = (f"{r['arm']:<18} {r['seq']:<10} {r['n']:>3} {fmt(r['ate']):>8} {fmt(r['ate_sd']):>7} {fmt(r['ate_med']):>8} {fmt(r['cov'], 1):>6} "
                f"{fmt(r['skip_q'], 1):>8}")
        if a.detail:
            line += (f" {fmt(r['skip_s'], 1):>6} {fmt(r['abort'], 1):>6} {fmt(r['kfs'], 0):>7}"
                     f" {fmt(r['deferred'], 0):>7} {fmt(r['refused'], 1):>6}")
        line += f" | {fmt(p[0]) if p else '':>9} {fmt(p[1], 1) if p else '':>10}"
        if r["n_ate"] < r["n"]:
            line += f"  ({r['n'] - r['n_ate']} run(s) without ATE)"
        print(line)
        if a.detail:
            for run in r["runs"]:
                s = run["skip"]
                ab = s[3] if s and s[3] is not None else '-'
                k = run["kf"]
                print(f"{'':<29}   {fmt(run['ate']):>8} {'':>7} {'':>8} {s[1] if s else '-':>8} {s[2] if s else '-':>6} "
                      f"{ab:>6} {s[0] if s else '-':>7} {k[0] if k else '-':>7} {k[2] if k else '-':>6}"
                      f"   {os.path.dirname(run['path'])}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "seq", "runs", "ate_mean", "ate_sd", "ate_median", "trajectory_coverage_pct", "skipped_queue_mean",
                        "skipped_stop_mean", "aborted_mean", "kfs_processed_mean", "kf_deferred_busy_mean",
                        "kf_refused_queue_mean", "paper_ate", "paper_skipped"])
            for r in rows:
                p = PAPER.get(r["arm"], {}).get(r["seq"], (None, None))
                w.writerow([r["arm"], r["seq"], r["n"], r["ate"], r["ate_sd"], r["ate_med"], r["cov"], r["skip_q"],
                            r["skip_s"], r["abort"], r["kfs"], r["deferred"], r["refused"], p[0], p[1]])
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
