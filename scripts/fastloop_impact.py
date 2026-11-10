#!/usr/bin/env python3
"""Tabulate run_fastloop_impact.sh against FastLoop Tables II and III.

FastLoop, arXiv:2603.17201v2:
  Table II  -- Local Mapping stop time (ms) and keyframes refused by Tracking while
               Local Mapping is stopped, per sequence, FastLoop vs. ORB-SLAM3.
  Table III -- for each sequence's largest loop closure event: its duration, the
               number of tracking frames it overlapped, their mean latency, and the
               mean latency outside loop closure.

Everything is derived from absolute steady_clock windows written by a REGISTER_STATS
build, since both tables are about one thread's work overlapping another's and
durations alone cannot express that:

  LoopClosing/data/loopCorrection_window.txt   kfId: startNs endNs   CorrectLoop()
  LoopClosing/data/lmStopRequested_window.txt  kfId: startNs endNs   RequestStop -> Release
  LoopClosing/data/lmStopped_window.txt        kfId: startNs endNs   stopped -> Release
  Tracking/data/frame_window.txt               frameId: startNs endNs
  Tracking/data/kfRefusedLMStop.txt            frameId: ns
  Tracking/data/tracking_time.txt              frameId: ms

Two readings of "Local Mapping stop time" are possible and the paper does not
disambiguate them, so both are reported. `requested` spans RequestStop() to Release():
it is the window over which Tracking actually refuses keyframes, because the gate in
NeedNewKeyFrame() is isStopped() || stopRequested(). `stopped` spans the point Local
Mapping came to rest to Release(), excluding the EmptyQueue() and stop-handshake tail.
`requested` is the default, as it is the one the refusal counts belong to.

Values are meaned over runs and reported as mean +- sample stdev, matching how the
paper reports its 5-run averages. The two FastLoop variants live side by side under
one label, distinguished by their kernel bitmask directory, and share the single
ORB-SLAM3 baseline; --fl-kernels picks which one to compare.

Usage:
  scripts/fastloop_impact.py <label> [--fl-kernels 001110] [--window requested|stopped]
                             [--detail] [--csv out.csv] [--latex]
e.g.
  scripts/fastloop_impact.py desktop
  scripts/fastloop_impact.py desktop --fl-kernels 001111   # GPU global BA enabled
"""
import argparse
import csv
import glob
import os
import statistics as st

SEQS = ["magistrale1", "magistrale2", "outdoors5", "V203"]
# Sequence -> results tree written by run_script.sh via RESULTS_ROOT.
ROOT = {s: ("Results-euroc-fastloop" if s.startswith(("MH", "V1", "V2"))
            else "Results-tumvi-fastloop") for s in SEQS}
# (key, results directory, kernel-directory template; {fl} is --fl-kernels).
# The kernel directory is what distinguishes arms that share a results directory --
# FastLoop/001110 from FastLoop/001111 -- so it is built exactly rather than globbed.
# ORB-SLAM3 has no kernel directory at all; Nitro-SLAM's carries all three modules'
# bitmasks, because it enables FastTrack and TurboMap alongside FastLoop.
ARMS = [("Original",   "ORB-SLAM3",  ""),
        ("FastLoop",   "FastLoop",   "{fl}"),
        ("Nitro-SLAM", "Nitro-SLAM", "111111-1111-{fl}")]

# The published numbers, for side-by-side comparison (desktop, avg. of 5 runs).
PAPER_T2 = {  # sequence -> {arm: (LM stop ms, KFs refused)}
    "magistrale1": {"FastLoop": (74.8, 2), "Original": (252.2, 8)},
    "magistrale2": {"FastLoop": (107.1, 3), "Original": (441.0, 5)},
    "outdoors5":   {"FastLoop": (493.1, 7), "Original": (863.8, 15)},
    "V203":        {"FastLoop": (42.9, 1),  "Original": (58.1, 1)},
}
PAPER_T3 = {  # sequence -> {arm: (duration ms, frames, during ms, normal ms)}
    "magistrale1": {"FastLoop": (168, 3, 29.0, 18.3), "Original": (875, 16, 27.0, 19.3)},
}


def read_windows(path):
    """'<id>: <startNs> <endNs>' -> [(id, startNs, endNs)], in file order."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if len(parts) != 2:
                continue
            out.append((int(key), int(parts[0]), int(parts[1])))
    return out


def read_pairs(path, cast=float):
    """'<id>: <value>' -> {id: value}."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            key, _, rest = line.partition(":")
            rest = rest.strip()
            if not rest:
                continue
            out[int(key)] = cast(rest.split()[0])
    return out


def overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def load_run(run_dir):
    lc = os.path.join(run_dir, "LoopClosing", "data")
    tr = os.path.join(run_dir, "Tracking", "data")
    return {
        "correction": read_windows(os.path.join(lc, "loopCorrection_window.txt")),
        "requested": read_windows(os.path.join(lc, "lmStopRequested_window.txt")),
        "stopped": read_windows(os.path.join(lc, "lmStopped_window.txt")),
        "frames": read_windows(os.path.join(tr, "frame_window.txt")),
        "refused": read_pairs(os.path.join(tr, "kfRefusedLMStop.txt"), int),
        "latency": read_pairs(os.path.join(tr, "tracking_time.txt"), float),
    }


def table2_row(run, window):
    """(mean LM stop ms, #events, #refusals inside those windows, #refusals total)."""
    wins = run[window]
    durs = [(e - s) / 1e6 for _, s, e in wins]
    # Refusals are timestamped so they can be attributed to the correction that caused
    # them. MergeLocal() also stops Local Mapping and is not recorded here, so an
    # unattributed refusal is a merge (or a shutdown stop), not a miscount.
    stop_wins = run["requested"]
    inside = sum(1 for ts in run["refused"].values()
                 if any(s <= ts <= e for _, s, e in stop_wins))
    return (st.mean(durs) if durs else None, len(durs), inside, len(run["refused"]))


def table3_row(run):
    """Largest loop closure event: (duration ms, #frames, during ms, normal ms)."""
    if not run["correction"]:
        return None
    _, s, e = max(run["correction"], key=lambda w: w[2] - w[1])
    during, normal = [], []
    for fid, fs, fe in run["frames"]:
        ms = run["latency"].get(fid)
        if ms is None:
            continue
        if overlaps(fs, fe, s, e):
            during.append(ms)
        # "Normal" is outside *any* loop closure, not just the largest one.
        elif not any(overlaps(fs, fe, cs, ce) for _, cs, ce in run["correction"]):
            normal.append(ms)
    return ((e - s) / 1e6, len(during),
            st.mean(during) if during else None,
            st.mean(normal) if normal else None)


def find_runs(seq, arm, version, fl_kernels):
    """Completed run directories for one sequence and one ARMS entry."""
    _, arm_dir, kdir_tmpl = arm
    kdir = kdir_tmpl.format(fl=fl_kernels)
    parts = [ROOT[seq], arm_dir] + ([kdir] if kdir else []) + [version, seq, "*"]
    runs = [p for p in glob.glob(os.path.join(*parts)) if os.path.isdir(p)]
    # .batch_done is written by the driver only after a run finished and evaluated.
    # Without this a batch still in flight contributes a directory whose stats are
    # absent or half-written, which silently drags every mean toward zero.
    return sorted(r for r in runs if os.path.exists(os.path.join(r, ".batch_done")))


def agg(vals):
    """(mean, sample stdev or None) over runs; None stdev when fewer than two."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else None)


def pm(mv, prec=1):
    """'mean +- sd', or just the mean on a single run."""
    m, sd = mv
    if m is None:
        return "-"
    return f"{m:.{prec}f}" if sd is None else f"{m:.{prec}f}+-{sd:.{prec}f}"


def fmt(v, prec=1):
    return "-" if v is None else f"{v:.{prec}f}"


def aggregate(runs, window="requested"):
    """Per-sequence, per-arm summary over a list of loaded runs.

    Two-stage: each run collapses to one number first (its mean stop time, its
    refusals per closure), then the spread is taken across runs. Runs that closed no
    loop contribute nothing but are still counted in runs_total, because "4 of 5 runs
    closed" is part of the result -- on magistrale1 the gravity gate rejects every
    detection in most runs, and a mean over the survivors alone would hide that.
    """
    per = [table2_row(r, window) for r in runs]
    per = [x for x in per if x[0] is not None]
    t3 = [t for t in (table3_row(r) for r in runs) if t]
    return {
        "runs_total": len(runs),
        "runs_closed": len(per),
        "events": sum(x[1] for x in per),
        "lm_stop": agg([x[0] for x in per]),
        # Per closure event, not per run: a run that closed three loops refuses about
        # three times as many keyframes as one that closed a single loop.
        "kfs_per_event": agg([x[2] / x[1] for x in per]),
        "kfs_per_run": agg([float(x[2]) for x in per]),
        "refused_all": agg([float(x[3]) for x in per]),
        "duration": agg([t[0] for t in t3]),
        "frames": agg([float(t[1]) for t in t3]),
        "during": agg([t[2] for t in t3]),
        "normal": agg([t[3] for t in t3]),
    }


def collect(seq, arm, version, fl_kernels, window="requested"):
    """aggregate() over whatever completed runs exist for one sequence and ARMS entry."""
    return aggregate([load_run(r) for r in
                      find_runs(seq, arm, version, fl_kernels)], window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="run label, e.g. desktop (version dir is fl-<label>)")
    ap.add_argument("--fl-kernels", default="001110",
                    help="FastLoop kernel bitmask to compare (default 001110; "
                         "001111 is the same arm with GPU global BA enabled)")
    ap.add_argument("--window", choices=["requested", "stopped"], default="requested",
                    help="which Local Mapping stop window to report (default: requested)")
    ap.add_argument("--detail", action="store_true",
                    help="per-run rows, both stop windows, and refusal attribution")
    ap.add_argument("--csv")
    ap.add_argument("--latex", action="store_true",
                    help="bare LaTeX rows; scripts/fastloop_latex.py writes full tables")
    args = ap.parse_args()
    version = f"fl-{args.label}"

    data, summ = {}, {}
    for seq in SEQS:
        for arm in ARMS:
            runs = [load_run(r) for r in
                    find_runs(seq, arm, version, args.fl_kernels)]
            data[(seq, arm[0])] = runs
            summ[(seq, arm[0])] = aggregate(runs, args.window)

    # Only show arms and sequences that actually have runs, so a two-arm results tree
    # prints exactly as before rather than growing a column of dashes.
    arms = [a for a in ARMS if any(summ[(s, a[0])]["runs_total"] for s in SEQS)]
    seqs = [s for s in SEQS if any(summ[(s, a[0])]["runs_closed"] for a in arms)]

    print(f"\nTABLE II  Impact of loop correction")
    print(f"  version {version}, FastLoop bitmask {args.fl_kernels}, "
          f"'{args.window}' stop window")
    print(f"{'Sequence':<13} {'System':<12} {'LM Stop (ms)':>16} {'KFs/event':>12} "
          f"{'closed':>8}   {'paper LM / KFs':>16}")
    for seq in seqs:
        for arm in arms:
            v = summ[(seq, arm[0])]
            if not v["runs_closed"]:
                continue
            p2 = PAPER_T2.get(seq, {}).get(arm[0])
            ps = f"{p2[0]:.1f} / {p2[1]}" if p2 else "-"
            cl = f"{v['runs_closed']}/{v['runs_total']}"
            print(f"{seq:<13} {arm[0]:<12} {pm(v['lm_stop']):>16} "
                  f"{pm(v['kfs_per_event']):>12} {cl:>8}   {ps:>16}")

    if args.detail:
        print("\n  refused/event is the headline; refused/run additionally varies with "
              "how many\n  loops a run closed. refused_all includes refusals from a map "
              "merge or shutdown.")
        for seq in seqs:
            for arm in arms:
                v = summ[(seq, arm[0])]
                if not v["runs_closed"]:
                    continue
                print(f"  {seq:<13} {arm[0]:<12} runs_closed={v['runs_closed']}/"
                      f"{v['runs_total']}  events={v['events']:<4} "
                      f"refused/event={pm(v['kfs_per_event'])}  "
                      f"refused/run={pm(v['kfs_per_run'])}  "
                      f"refused_all/run={pm(v['refused_all'])}")
        print("\n  both stop windows, per run:")
        for seq in SEQS:
            for arm in arms:
                for i, run in enumerate(data[(seq, arm[0])]):
                    a = table2_row(run, "requested")
                    b = table2_row(run, "stopped")
                    if a[0] is None:
                        continue
                    print(f"  {seq:<13} {arm[0]:<12} run{i}  requested={a[0]:8.1f}ms  "
                          f"stopped={b[0]:8.1f}ms  events={a[1]}  refused={a[2]}")

    print(f"\nTABLE III  Tracking latency during the largest loop closure event")
    print(f"{'Sequence':<13} {'System':<12} {'Duration (ms)':>15} {'Frames':>11} "
          f"{'During (ms)':>14} {'Normal (ms)':>14}")
    for seq in seqs:
        for arm in arms:
            v = summ[(seq, arm[0])]
            if v["duration"][0] is None:
                continue
            print(f"{seq:<13} {arm[0]:<12} {pm(v['duration'], 0):>15} "
                  f"{pm(v['frames'], 1):>11} {pm(v['during']):>14} "
                  f"{pm(v['normal']):>14}")
            p3 = PAPER_T3.get(seq, {}).get(arm[0])
            if p3:
                print(f"{'':<13} {'  paper':<12} {p3[0]:>15} {p3[1]:>11} "
                      f"{p3[2]:>14} {p3[3]:>14}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "system", "fl_kernels", "runs_closed", "runs_total",
                        "events", "lm_stop_ms", "lm_stop_sd", "kfs_per_event",
                        "kfs_per_event_sd", "kfs_per_run", "event_duration_ms",
                        "frames_overlapped", "latency_during_ms", "latency_normal_ms"])
            for seq in seqs:
                for arm in arms:
                    v = summ[(seq, arm[0])]
                    if not v["runs_closed"]:
                        continue
                    w.writerow([seq, arm[0], args.fl_kernels, v["runs_closed"],
                                v["runs_total"], v["events"],
                                fmt(v["lm_stop"][0]), fmt(v["lm_stop"][1]),
                                fmt(v["kfs_per_event"][0]), fmt(v["kfs_per_event"][1]),
                                fmt(v["kfs_per_run"][0]), fmt(v["duration"][0], 0),
                                fmt(v["frames"][0], 1), fmt(v["during"][0]),
                                fmt(v["normal"][0])])
        print(f"\nwrote {args.csv}")

    if args.latex:
        print("\n% rows: sequence & system & LM stop & KFs/event & duration & frames "
              "& during & normal")
        for seq in seqs:
            for arm in arms:
                v = summ[(seq, arm[0])]
                if not v["runs_closed"]:
                    continue
                print(f"{seq} & {arm[0]} & {pm(v['lm_stop'])} & "
                      f"{pm(v['kfs_per_event'], 1)} & {pm(v['duration'], 0)} & "
                      f"{pm(v['frames'], 1)} & {pm(v['during'])} & "
                      f"{pm(v['normal'])} \\\\")
    print()


if __name__ == "__main__":
    main()
