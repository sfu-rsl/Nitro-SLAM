#!/usr/bin/env python3
"""Summarise a Results-poseopt tree: PO timing per mode plus trajectory ATE.

Reads the [PO_TIMING] block the binary prints at exit (see PoseOptTiming.h)
out of each run's ostream.txt, and the ATE that evaluate3.py prints on the
same stream, then tabulates them per sequence and mode.
"""
import os
import re
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Results-poseopt"

TIMING = re.compile(
    r"\[PO_TIMING\]\s+(\S+)\s+calls=(\d+)\s+mean_ms=([\d.]+)\s+max_ms=([\d.]+)"
    r"\s+total_ms=([\d.]+)\s+mean_inliers=([\d.]+)")
ALL = re.compile(r"\[PO_TIMING\] ALL calls=(\d+) mean_ms=([\d.]+) total_ms=([\d.]+)")
# evaluate3.py prints "absolute_translational_error.rmse <value> m"
RMSE = re.compile(r"absolute_translational_error\.rmse\s+([\d.eE+-]+)\s*m")


def walk(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "ostream.txt" not in filenames:
            continue
        parts = dirpath.split(os.sep)
        # .../<mode>/<sequence>/<iteration>
        if len(parts) < 3:
            continue
        mode, seq = parts[-3], parts[-2]
        yield mode, seq, os.path.join(dirpath, "ostream.txt")


rows = defaultdict(list)
for mode, seq, path in walk(ROOT):
    with open(path, errors="replace") as fh:
        text = fh.read()
    entry = {"variants": {}, "ate": None, "all": None}
    for m in TIMING.finditer(text):
        entry["variants"][m.group(1)] = {
            "calls": int(m.group(2)), "mean_ms": float(m.group(3)),
            "max_ms": float(m.group(4)), "total_ms": float(m.group(5)),
            "inliers": float(m.group(6)),
        }
    m = ALL.search(text)
    if m:
        entry["all"] = {"calls": int(m.group(1)), "mean_ms": float(m.group(2)),
                        "total_ms": float(m.group(3))}
    m = RMSE.search(text)
    if m:
        entry["ate"] = float(m.group(1))
    rows[(seq, mode)].append(entry)

seqs = sorted({s for s, _ in rows})
modes = sorted({m for _, m in rows})

variant_names = sorted({v for e in rows.values() for r in e for v in r["variants"]})

for variant in variant_names + ["ALL"]:
    print(f"\n=== {variant}: mean ms per call ===")
    print(f"{'sequence':<10}" + "".join(f"{m:>22}" for m in modes))
    for seq in seqs:
        line = f"{seq:<10}"
        for mode in modes:
            runs = rows.get((seq, mode), [])
            vals = []
            calls = []
            for r in runs:
                d = r["all"] if variant == "ALL" else r["variants"].get(variant)
                if d:
                    vals.append(d["mean_ms"])
                    calls.append(d["calls"])
            if vals:
                line += f"{sum(vals)/len(vals):>15.3f} (n={sum(calls)//len(calls):<4})"
            else:
                line += f"{'-':>22}"
        print(line)

print("\n=== ATE RMSE (m) ===")
print(f"{'sequence':<10}" + "".join(f"{m:>14}" for m in modes))
for seq in seqs:
    line = f"{seq:<10}"
    for mode in modes:
        vals = [r["ate"] for r in rows.get((seq, mode), []) if r["ate"] is not None]
        line += f"{sum(vals)/len(vals):>14.4f}" if vals else f"{'-':>14}"
    print(line)
