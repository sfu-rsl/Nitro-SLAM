#!/usr/bin/env python3
"""Summarise the outdoors5 loop-closure ablation into a table.

Walks Results/ for the ablation's run directories and reports, per config, how
many runs detected a loop, how many actually closed one, and the resulting ATE.
"""
import os
import time
import re
import sys
from collections import defaultdict

# Scan every Results root -- the sweep and the maximal run live in separate
# directories. Override by passing paths on the command line.
RESULTS_ROOTS = [d for d in ("Results", "Results-ablation") if os.path.isdir(d)]
ABL_VERSION = "abl"
VERSION_RE = re.compile(r"^(.+)\.(\d+)$")
ATE_RE = re.compile(r"absolute_translational_error\.rmse\s+([0-9.]+)")

# kernel_status bit -> human name, per stereo_inertial_tum_vi.cc
BITS = {
    "FastTrack": ["orbExtraction", "stereoMatch", "searchLocalPoints",
                  "poseEstimation", "poseOptimization(on)"],
    "TurboMap": ["searchForTriangulation", "fuse", "keyframeCulling", "LBA",
                 "newTriangulation", "gpu2"],
    "FastLoop": ["mergedSBP", "merged3SBP", "searchAndFuse", "singleSBP",
                 "graphOpt", "globalBA"],
}


def scan(path):
    """Extract loop-closure outcome from one run's ostream.txt."""
    log = os.path.join(path, "ostream.txt")
    if not os.path.exists(log):
        return None
    with open(log, errors="replace") as fh:
        text = fh.read()
    ate = ATE_RE.search(text)
    finished = "End of saving trajectory" in text
    # An unfinished log is either a crash or a run still in progress. The eval
    # script echoes "Plotting data" once the binary exits, however it exited, so
    # that marker without a saved trajectory means the binary died. A run killed
    # outright never reaches even that, so fall back to a stale mtime.
    if finished:
        live = False
    elif "Plotting data" in text:
        live = False
    else:
        live = (time.time() - os.path.getmtime(log)) < 180
    return {
        "detected": text.count("*Loop detected"),
        "good": text.count("Good loop found!"),
        "bad": text.count("BAD LOOP!!!"),
        "finished": finished,
        "live": live,
        "ate": float(ate.group(1)) if ate else None,
        "kfs": (lambda m: int(m.group(1)) if m else None)(
            re.search(r"Map \d+ has (\d+) KFs", text)),
    }


def label(system, kernel_dir):
    """Name a config by the single kernel it turns on, where there is one."""
    if system == "ORB-SLAM3":
        return "baseline (no optimizations)"
    if not kernel_dir:
        return system
    # Multi-subsystem runs: "FastTrack&TurboMap&FastLoop" / "01101-1111-111111".
    # run_script.sh names the all-three case Nitro-SLAM rather than joining with
    # '&'. Too many kernels to spell out either way, so report the count instead.
    if "&" in system or system == "Nitro-SLAM":
        parts = kernel_dir.split("-")
        on = sum(c == "1" for c in kernel_dir)
        # FastTrack's 5th bit is poseOptimization, on-by-default, not a kernel.
        # Its status leads the directory name whenever FastTrack is enabled,
        # which for Nitro-SLAM it always is.
        if (system.startswith("FastTrack") or system == "Nitro-SLAM") \
                and parts and len(parts[0]) > 4:
            on -= int(parts[0][4] == "1")
        return f"COMBINED ({on} kernels)"
    names = BITS.get(system, [])
    on = [names[i] if i < len(names) else f"bit{i}"
          for i, c in enumerate(kernel_dir) if c == "1"]
    if system == "FastTrack":
        # poseOptimization is on-by-default, not a GPU kernel; report inversely.
        on = [n for n in on if n != "poseOptimization(on)"]
        if len(kernel_dir) > 4 and kernel_dir[4] == "0":
            on.append("poseOptimization OFF")
    if not on:
        return f"{system}, no kernels"
    return f"{system}: " + " + ".join(on)


def parse(rel):
    """Locate a run in the Results tree: (system, kernel_dir, dataset, version, pass).

    Two layouts exist on disk. run_script.sh used to fuse the pass number into
    the version and hang it off the dataset; it now passes the iteration
    separately and puts the version above the dataset, since a version spans
    every dataset in a sweep while an iteration is a repeat of one dataset:

        old   <system>[/<kernels>]/<dataset>/<version>          version "abl.2"
        new   <system>[/<kernels>]/<version>/<dataset>/<iter>   version "abl", iter "2"

    The dataset is second-from-last either way; a numeric leaf is the giveaway
    that the pass moved out of the version string.
    """
    if len(rel) < 3:
        return None
    dataset = rel[-2]
    if rel[-1].isdigit():
        version, pass_no, prefix = rel[-3], int(rel[-1]), rel[:-3]
    else:
        m = VERSION_RE.match(rel[-1])
        if not m:
            return None
        version, pass_no, prefix = m.group(1), int(m.group(2)), rel[:-2]
    # Everything above the version is the config: system, then kernel bits.
    if not prefix:
        return None
    return prefix[0], (prefix[1] if len(prefix) > 1 else ""), dataset, version, pass_no


def collect():
    """Walk Results/ and group every abl.<pass> run by config."""
    runs = defaultdict(dict)  # (system, kernel_dir) -> {pass: result}
    for results in RESULTS_ROOTS:
        for root, dirs, files in os.walk(results):
            if "ostream.txt" not in files:
                continue
            parsed = parse(os.path.relpath(root, results).split(os.sep))
            if not parsed:
                continue
            system, kernel_dir, dataset, version, pass_no = parsed
            if version != ABL_VERSION or dataset != "outdoors5":
                continue
            res = scan(root)
            if res:
                runs[(system, kernel_dir)][pass_no] = res
    return runs


def main():
    runs = collect()

    if not runs:
        print("no abl.* runs found yet")
        return

    order = {"ORB-SLAM3": 0, "FastTrack": 1, "TurboMap": 2, "FastLoop": 3}  # combined runs sort last
    keys = sorted(runs, key=lambda k: (order.get(k[0], 9), k[1]))

    hdr = f"{'config':<46} {'status':<9} {'closed':<7} {'det':<10} {'ATE (m)':<22} {'KFs'}"
    print(hdr)
    print("-" * len(hdr))
    for key in keys:
        system, kernel_dir = key
        by_pass = runs[key]
        passes = sorted(by_pass)
        n = len(passes)
        closed = sum(1 for p in passes if by_pass[p]["good"] > 0)
        live = sum(1 for p in passes if by_pass[p]["live"])
        crashed = sum(1 for p in passes
                      if not by_pass[p]["finished"] and not by_pass[p]["live"])
        det = "/".join(str(by_pass[p]["detected"]) for p in passes)
        ates = ["-" if by_pass[p]["ate"] is None else f"{by_pass[p]['ate']:.2f}"
                for p in passes]
        kfs = "/".join("-" if by_pass[p]["kfs"] is None else str(by_pass[p]["kfs"])
                       for p in passes)
        # A crash after the loop already closed is a shutdown-path failure, not a
        # loop-closure regression -- worth separating, since the two point at
        # completely different code.
        late = sum(1 for p in passes
                   if not by_pass[p]["finished"] and not by_pass[p]["live"]
                   and by_pass[p]["good"] > 0)
        if live:
            status = "RUNNING"
        elif crashed:
            status = f"{crashed} CRASH" + ("*" if late else "")
        else:
            status = "OK"
        tag = f"{label(system, kernel_dir)}"
        if kernel_dir:
            tag += f" [{kernel_dir}]"
        print(f"{tag:<46} {status:<9} {closed}/{n:<5} {det:<10} "
              f"{','.join(ates):<22} {kfs}")


def summary():
    """Per-config counts: runs, how many detected, closed, crashed."""
    runs = collect()
    order = {"ORB-SLAM3": 0, "FastTrack": 1, "TurboMap": 2, "FastLoop": 3}  # combined runs sort last
    keys = sorted(runs, key=lambda k: (order.get(k[0], 9), k[1]))

    hdr = f"{'config':<46} {'runs':>4} {'detected':>9} {'closed':>7} {'crashed':>8}"
    print(hdr)
    print("-" * len(hdr))
    tot = [0, 0, 0, 0]
    for system, kernel_dir in keys:
        by_pass = runs[(system, kernel_dir)]
        # Exclude in-flight runs: their counts are not final.
        done = [r for r in by_pass.values() if not r["live"]]
        if not done:
            continue
        n = len(done)
        det = sum(1 for r in done if r["detected"] > 0)
        cl = sum(1 for r in done if r["good"] > 0)
        cr = sum(1 for r in done if not r["finished"])
        tag = label(system, kernel_dir) + (f" [{kernel_dir}]" if kernel_dir else "")
        print(f"{tag:<46} {n:>4} {det:>9} {cl:>7} {cr:>8}")
        for i, v in enumerate((n, det, cl, cr)):
            tot[i] += v
    print("-" * len(hdr))
    print(f"{'TOTAL':<46} {tot[0]:>4} {tot[1]:>9} {tot[2]:>7} {tot[3]:>8}")


if __name__ == "__main__":
    _paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if _paths:
        RESULTS_ROOTS = _paths
    if "--summary" in sys.argv:
        sys.exit(summary())
    sys.exit(main())
