#!/usr/bin/env python3
"""LaTeX table for the high-keyframe-insertion stress test (run_hikf.sh).

CPU tracking throughout: ORB-SLAM3 against ORB-SLAM3 + TurboMap, isolating TurboMap.

Each cell gives ATE RMSE (m), mean or median over completed runs, and three means:
LBA + KF culling skipped because another keyframe was queued (the TurboMap paper's
metric); full LBA (%), the share of processed keyframes whose LBA was neither skipped
nor aborted by a keyframe request from Tracking (skips alone hide the aborts); and
rejected keyframes -- frames where Tracking wanted a keyframe but did not create one
because Local Mapping was busy. Rejections for a full queue (3+ keyframes waiting)
are counted too but are 0 throughout: Local Mapping drains the queue by skipping.
Tracking failures and map resets were considered and rejected: failures are erratic
run to run, and on TUM-VI every configuration resets 1-2 times during IMU
initialisation regardless of load.
Parsing is shared with hikf_table.py.

Writes both statistics:
  <out_dir>/hikf_<label>_table_mean.tex
  <out_dir>/hikf_<label>_table_median.tex

Usage:
  scripts/hikf_latex.py <label> [--mf 4] [--out-dir analysis_out]
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hikf_table import SEQS, collect  # noqa: E402

# (baseline result dir, + TurboMap result dir)
BASELINE, OURS = "ORB-SLAM3", "TurboMap"

CAPTION = (r"High keyframe insertion (unpaced, reduced keyframe interval): ATE RMSE "
           r"(m, {stat} of 5 runs), skipped LBA \& KF culling, keyframes with a complete "
           r"LBA (\%), and keyframes rejected because Local Mapping was busy (means).")


def full_lba_pct(skip):
    kfs, queue, stop, aborted = skip
    return 100.0 * (kfs - queue - stop - (aborted or 0)) / kfs if kfs else None


def cell(arm_dir, seq, version, stat):
    runs = collect(arm_dir, seq, version)
    ates = [r["ate"] for r in runs if r["ate"] is not None]
    skips = [r["skip"][1] for r in runs if r["skip"]]
    if len(ates) < len(runs) or len(skips) < len(runs):
        print(f"warning: {arm_dir} {seq}: {len(runs)} runs, {len(ates)} with ATE, "
              f"{len(skips)} with skip count", file=sys.stderr)
    full = [full_lba_pct(r["skip"]) for r in runs if r["skip"]]
    full = [x for x in full if x is not None]
    # kf = (deferred_busy, requested_busy, refused_queue); deferred_busy is the count of
    # keyframes Tracking wanted but did not create because Local Mapping was busy.
    rej = [r["kf"][0] + r["kf"][2] for r in runs if r["kf"]]
    agg = st.median if stat == "median" else st.mean
    return (agg(ates) if ates else None,
            st.mean(skips) if skips else None,
            st.mean(full) if full else None,
            st.mean(rej) if rej else None,
            len(runs))


def fmt(x, nd):
    return "--" if x is None else f"{x:.{nd}f}"


def table(version, stat):
    lines = [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{CAPTION.format(stat=stat)}}}",
        r"\label{tab:hikf}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l rrrr rrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{ORB-SLAM3} & \multicolumn{4}{c}{+ TurboMap} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Sequence & ATE & Skip & LBA\% & Rej. & ATE & Skip & LBA\% & Rej. \\",
        r"\midrule",
    ]
    for seq in SEQS:
        b_ate, b_skip, b_full, b_rej, b_n = cell(BASELINE, seq, version, stat)
        o_ate, o_skip, o_full, o_rej, o_n = cell(OURS, seq, version, stat)
        for arm, n in ((BASELINE, b_n), (OURS, o_n)):
            if n != 5:
                print(f"warning: {arm} {seq}: {n} completed runs", file=sys.stderr)
        lines.append(f"{seq} & {fmt(b_ate, 3)} & {fmt(b_skip, 1)} & {fmt(b_full, 1)} & {fmt(b_rej, 0)} & "
                     f"{fmt(o_ate, 3)} & {fmt(o_skip, 1)} & {fmt(o_full, 1)} & {fmt(o_rej, 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("--mf", default="4", help="KF_MAX_FRAMES used by run_hikf.sh (default 4)")
    ap.add_argument("--out-dir", default="analysis_out")
    a = ap.parse_args()
    version = f"hikf-{a.label}-mf{a.mf}"

    os.makedirs(a.out_dir, exist_ok=True)
    for stat in ("mean", "median"):
        path = os.path.join(a.out_dir, f"hikf_{a.label}_table_{stat}.tex")
        with open(path, "w") as f:
            f.write(table(version, stat))
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
