#!/usr/bin/env python3
"""LaTeX tables for the FastLoop loop-correction impact study (run_fastloop_impact.sh).

Writes the two tables of the FastLoop paper (arXiv:2603.17201v2):

  Table II  -- Local Mapping stop time and keyframes refused by Tracking while Local
               Mapping is stopped, per system.
  Table III -- for each sequence's largest loop closure event: its duration, the
               tracking frames it overlapped, their mean latency, and the mean latency
               outside any loop closure. The paper reports Magistrale 1 only; every
               Table II sequence is included here.

Columns/rows follow whichever arms have runs, so the same command covers the two-arm
comparison and the three-arm one that adds Nitro-SLAM (FastLoop plus accelerated
tracking and mapping). Parsing and aggregation are shared with fastloop_impact.py, so
the .tex and the readable table can never disagree.

Two things differ from the paper's presentation, both because the underlying runs
turned out noisier than a bare mean admits:

  * Keyframes refused are reported PER CLOSURE EVENT, not per run. A run that closed
    three loops refuses about three times as many keyframes as one that closed a
    single loop, and on magistrale1 the baseline does both -- 11 refusals in a
    one-closure run, 34 in a three-closure run. Per run that averages to 22.5+-16.3,
    which reads as noise; per event it is 11.2+-0.2.
  * A cell carries the number of runs that actually corrected a loop, since closing
    one is not guaranteed. On magistrale1 the gravity gate rejects every detection in
    three of five baseline runs, so that cell is a mean of two. --pm additionally
    prints the spread across runs, which the paper's Tables II and III omit.

Writes:
  <out_dir>/fastloop_<label>_<kernels>_table2.tex
  <out_dir>/fastloop_<label>_<kernels>_table3.tex

Usage:
  scripts/fastloop_latex.py <label> [--fl-kernels 001110] [--window requested|stopped]
                            [--pm] [--out-dir analysis_out]
e.g.
  scripts/fastloop_latex.py desktop
  scripts/fastloop_latex.py desktop --fl-kernels 001111 --pm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastloop_impact import SEQS, ARMS, collect  # noqa: E402

# Paper spelling of the sequence names.
DISPLAY = {"magistrale1": "Magistrale 1", "magistrale2": "Magistrale 2",
           "outdoors5": "Outdoors 5", "V203": "V203"}
# How each arm is named in the paper. The result-tree key stays as ARMS defines it;
# only the printed name changes, so renaming here never moves which runs a column
# reads. Arms appear in ARMS order, baseline first.
ARM_NAME = {"Original": "Original", "FastLoop": "GPU Loop Closing",
            "Nitro-SLAM": "Nitro-SLAM"}

CAPTION2 = (r"Impact of loop correction. Local Mapping stop time (ms) and number of "
            r"keyframes refused by Tracking per loop correction while Local Mapping is "
            r"stopped (desktop setting, mean of up to {n} runs).")
CAPTION3 = (r"Tracking latency during each sequence's largest loop closure event. "
            r"Frames is the number of tracking frames the event overlapped, During "
            r"their mean latency, and Normal the mean latency outside any loop closure "
            r"(desktop setting, mean of up to {n} runs).")


def fmt(mv, nd, pm):
    """A (mean, sd) pair as LaTeX; '--' when the cell has no runs behind it."""
    m, sd = mv
    if m is None:
        return "--"
    if pm and sd is not None:
        return rf"{m:.{nd}f} $\pm$ {sd:.{nd}f}"
    return f"{m:.{nd}f}"


def warn(seq, key, v):
    if v["runs_closed"] < v["runs_total"]:
        print(f"warning: {DISPLAY[seq]} {ARM_NAME[key]}: {v['runs_closed']} of "
              f"{v['runs_total']} runs closed a loop; cell is a mean of "
              f"{v['runs_closed']}", file=sys.stderr)


def runs_note(summ, seqs, keys, nruns):
    """LaTeX comment naming cells backed by fewer runs than were launched.

    "mean of up to N runs" in the caption is only honest if the exceptions travel with
    the file -- a reader pasting this into a paper cannot see the stderr warnings.
    """
    short = [(seq, k, summ[(seq, k)]) for seq in seqs for k in keys
             if 0 < summ[(seq, k)]["runs_closed"] < nruns]
    if not short:
        return [f"% every cell is a mean of {nruns} runs that closed a loop"]
    out = [f"% runs launched per cell: {nruns}. Cells below closed a loop in fewer,",
           "% and are means of that many; closing a loop is not guaranteed."]
    out += [f"%   {DISPLAY[seq]} {ARM_NAME[k]}: {v['runs_closed']}/{v['runs_total']} "
            f"runs, {v['events']} closure event(s)" for seq, k, v in short]
    return out


def table2(summ, seqs, keys, nruns, pm):
    n = len(keys)
    names = " & ".join(ARM_NAME[k] for k in keys)
    lines = runs_note(summ, seqs, keys, nruns) + [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{CAPTION2.format(n=nruns)}}}",
        r"\label{tab:loop-correction-impact}",
        # \scriptsize sits after \caption so only the table body shrinks.
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l {'r' * n} {'r' * n}}}",
        r"\toprule",
        rf" & \multicolumn{{{n}}}{{c}}{{LM Stop Time (ms)}} & "
        rf"\multicolumn{{{n}}}{{c}}{{KFs Refused}} \\",
        rf"\cmidrule(lr){{2-{n + 1}}}\cmidrule(lr){{{n + 2}-{2 * n + 1}}}",
        rf"Sequence & {names} & {names} \\",
        r"\midrule",
    ]
    for seq in seqs:
        stop = " & ".join(fmt(summ[(seq, k)]["lm_stop"], 1, pm) for k in keys)
        kfs = " & ".join(fmt(summ[(seq, k)]["kfs_per_event"], 1, pm) for k in keys)
        lines.append(f"{DISPLAY[seq]} & {stop} & {kfs}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table3(summ, seqs, keys, nruns, pm):
    lines = runs_note(summ, seqs, keys, nruns) + [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{CAPTION3.format(n=nruns)}}}",
        r"\label{tab:loop-closure-latency}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l rr rr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Loop Closure} & "
        r"\multicolumn{2}{c}{Tracking Latency (ms)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Sequence & System & Duration (ms) & Frames & During & Normal \\",
        r"\midrule",
    ]
    for seq in seqs:
        first = True
        for k in keys:
            v = summ[(seq, k)]
            if v["duration"][0] is None:
                continue
            label = DISPLAY[seq] if first else ""
            first = False
            lines.append(
                f"{label} & {ARM_NAME[k]} & {fmt(v['duration'], 0, pm)} & "
                f"{fmt(v['frames'], 1, pm)} & {fmt(v['during'], 1, pm)} & "
                f"{fmt(v['normal'], 1, pm)}" + r" \\")
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", help="run label, e.g. desktop (version dir is fl-<label>)")
    ap.add_argument("--fl-kernels", default="001110",
                    help="FastLoop kernel bitmask (default 001110)")
    ap.add_argument("--window", choices=["requested", "stopped"], default="requested",
                    help="which Local Mapping stop window to report")
    ap.add_argument("--pm", action="store_true",
                    help="include the spread across runs, which the paper omits")
    ap.add_argument("--out-dir", default="analysis_out")
    a = ap.parse_args()
    version = f"fl-{a.label}"

    summ = {}
    for seq in SEQS:
        for arm in ARMS:
            summ[(seq, arm[0])] = collect(seq, arm, version, a.fl_kernels, a.window)

    # Only emit arms and sequences that have runs, so a two-arm tree yields the
    # two-arm table rather than a column of dashes.
    keys = [arm[0] for arm in ARMS
            if any(summ[(s, arm[0])]["runs_total"] for s in SEQS)]
    seqs = [s for s in SEQS if any(summ[(s, k)]["runs_closed"] for k in keys)]
    if not seqs:
        sys.exit(f"no completed runs for {version} / {a.fl_kernels} -- "
                 f"check the label and bitmask")
    for seq in seqs:
        for k in keys:
            warn(seq, k, summ[(seq, k)])
    nruns = max(summ[(s, k)]["runs_total"] for s in seqs for k in keys)

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"fastloop_{a.label}_{a.fl_kernels}"
    for name, body in (("table2", table2(summ, seqs, keys, nruns, a.pm)),
                       ("table3", table3(summ, seqs, keys, nruns, a.pm))):
        path = os.path.join(a.out_dir, f"{stem}_{name}.tex")
        with open(path, "w") as f:
            f.write(body)
        print(f"wrote {path}", file=sys.stderr)
        print(body)


if __name__ == "__main__":
    main()
