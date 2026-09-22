#!/usr/bin/env python3
"""LaTeX table for the high-keyframe-insertion stress test (run_hikf.sh).

Rows are grouped by tracking backend; within each group the right-hand columns are
the left-hand configuration with TurboMap added, isolating TurboMap's effect:

  CPU tracking: ORB-SLAM3  -> TurboMap
  GPU tracking: FastTrack  -> FastTrack&TurboMap

Each cell gives ATE RMSE (m), mean or median over completed runs; LBA + KF culling
skipped because another keyframe was queued (the TurboMap paper's metric), mean; and
full LBA (%), mean -- the share of processed keyframes whose LBA was neither skipped
nor aborted by a keyframe request from Tracking. It combines skips with aborts, which
is how Local Mapping falls behind without the skip count showing it.
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

# (row label, baseline result dir, + TurboMap result dir)
GROUPS = [("CPU", "ORB-SLAM3", "TurboMap"),
          ("GPU", "FastTrack", "FastTrack&TurboMap")]

CAPTION = (r"High keyframe insertion: ATE RMSE (m, {stat} of up to 5 runs; a few cells "
           r"have fewer, where a run crashed), skipped LBA \& "
           r"KF culling and keyframes with a complete LBA (\%) (means). Baseline is ORB-SLAM3 "
           r"(CPU tracking) or FastTrack (GPU tracking, $\approx$2$\times$ faster, "
           r"doubling the keyframe rate).")


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
    agg = st.median if stat == "median" else st.mean
    return (agg(ates) if ates else None,
            st.mean(skips) if skips else None,
            st.mean(full) if full else None,
            len(runs))


def fmt(x, nd):
    return "--" if x is None else f"{x:.{nd}f}"


def table(version, stat):
    lines = [
        r"% requires \usepackage{booktabs,multirow}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{CAPTION.format(stat=stat)}}}",
        r"\label{tab:hikf}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ll rrr rrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Baseline} & \multicolumn{3}{c}{+ TurboMap} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Tracking & Sequence & ATE & Skip & LBA\% & ATE & Skip & LBA\% \\",
    ]
    for label, base, ours in GROUPS:
        lines.append(r"\midrule")
        for i, seq in enumerate(SEQS):
            b_ate, b_skip, b_full, b_n = cell(base, seq, version, stat)
            o_ate, o_skip, o_full, o_n = cell(ours, seq, version, stat)
            for arm, n in ((base, b_n), (ours, o_n)):
                if n != 5:
                    print(f"warning: {arm} {seq}: {n} completed runs", file=sys.stderr)
            first = rf"\multirow{{{len(SEQS)}}}{{*}}{{{label}}}" if i == 0 else ""
            lines.append(f"{first} & {seq} & {fmt(b_ate, 3)} & {fmt(b_skip, 1)} & {fmt(b_full, 1)} & "
                         f"{fmt(o_ate, 3)} & {fmt(o_skip, 1)} & {fmt(o_full, 1)} \\\\")
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
