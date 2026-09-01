#!/usr/bin/env python3
"""Pick the median-ATE run of each sequence and report its trajectory.

Each sequence is run N times (Results-<dataset>/<system>/[<config>/]<machine>/
<sequence>/<run>/), so a single "representative" trajectory has to be chosen
before plotting or tabulating. This walks a results root, reads the ATE that
evaluate3.py appends to every run's ostream.txt, and picks the run sitting at
the median of that sequence's runs. With an even number of runs there is no
middle element, so the lower of the two central runs is taken -- an actual run
on disk, unlike an interpolated value, which is the point of the exercise.

Several roots can be scanned at once, which is how one table covers both
platforms: the machine directory at the bottom of each run path (desktop, jetson)
already tells them apart, so the columns come out grouped by platform.

  ./median_trajectory.py Results-euroc
  ./median_trajectory.py Results-euroc data/jetson/Results-euroc --latex
  ./median_trajectory.py Results-tumvi --metric median --csv medians.csv
  ./median_trajectory.py Results-euroc --copy median-trajectories
"""
import argparse
import csv
import fnmatch
import os
import re
import shutil
import statistics
import sys

# evaluate3.py --verbose prints one "absolute_translational_error.<stat> <v> m"
# line per statistic; any of them can be the ranking metric.
ATE_RE = {
    stat: re.compile(r"absolute_translational_error\.%s\s+([0-9.eE+-]+)" % stat)
    for stat in ("rmse", "mean", "median", "std", "min", "max")
}


def read_ate(run_dir, metric):
    """Return the requested ATE statistic for one run, or None if it has none.

    A run without the statistic either crashed or lost tracking before the
    evaluation step, and cannot take part in the ranking.
    """
    log = os.path.join(run_dir, "ostream.txt")
    if not os.path.exists(log):
        return None
    with open(log, errors="replace") as fh:
        text = fh.read()
    m = ATE_RE[metric].search(text)
    return float(m.group(1)) if m else None


def find_sequences(roots):
    """Group run directories by sequence, across every results root given.

    A run directory is any directory holding an ostream.txt; its parent is the
    sequence. Keying on the path from the root down to the sequence keeps the
    systems apart (and the Nitro-SLAM kernel-config level, which ORB-SLAM3 does
    not have) without hardcoding the layout, so the same walk works for
    Results-euroc and Results-tumvi alike.

    The key is taken relative to its own root, so a desktop root and a Jetson
    root merge into one set of variants rather than colliding: the machine
    directory at the end of the variant path is what separates them.
    """
    sequences = {}
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if "ostream.txt" not in filenames:
                continue
            seq_dir = os.path.dirname(dirpath)
            sequences.setdefault(os.path.relpath(seq_dir, root), []).append(dirpath)
    return sequences


def run_key(run_dir):
    """Sort runs numerically where the run id is a number, else by name."""
    name = os.path.basename(run_dir)
    return (0, int(name), "") if name.isdigit() else (1, 0, name)


def trajectory_files(run_dir):
    traj = os.path.join(run_dir, "trajectory")
    if not os.path.isdir(traj):
        return []
    return sorted(os.path.join(traj, f) for f in os.listdir(traj))


def aggregate(values, agg):
    """Reduce a sequence's per-run ATEs to the single number being reported."""
    if agg == "median":
        return statistics.median(values)
    if agg == "best":
        return min(values)
    if agg == "worst":
        return max(values)
    return statistics.fmean(values)


def representative_index(count, agg):
    """Index into the ATE-sorted runs of the one that stands for the sequence.

    Copying a trajectory needs an actual run on disk, and only "best"/"worst"
    name one unambiguously. The median of an even count sits between two runs,
    and a mean usually matches none, so those take the lower central run -- its
    ATE is then close to, but not equal to, the reported number.
    """
    if agg == "best":
        return 0
    if agg == "worst":
        return count - 1
    return (count - 1) // 2


def aggregate_runs(runs, metric, agg):
    """Rank a sequence's runs by ATE and reduce them.

    Returns (reported value, representative run, ranked runs, runs with no ATE).
    The value is computed over the runs' ATEs directly rather than read off the
    representative run, so an even run count gives a true median (the mean of
    the two central values) instead of one run's number wearing that label.
    With an odd count (5, as EuRoC and TUM-VI are run here) they coincide.
    """
    scored, failed = [], []
    for run in sorted(runs, key=run_key):
        ate = read_ate(run, metric)
        (failed if ate is None else scored).append(run if ate is None else (ate, run))
    if not scored:
        return None, None, [], failed
    scored.sort(key=lambda pair: (pair[0], run_key(pair[1])))
    value = aggregate([ate for ate, _ in scored], agg)
    return value, scored[representative_index(len(scored), agg)][1], scored, failed


def escape(text):
    """Escape the LaTeX specials that show up in system/config names."""
    return re.sub(r"([&%$#_{}])", r"\\\1", text)


# The machine directory at the end of a variant path, as it should read in a
# table header. Anything not listed keeps its directory name.
PLATFORM_LABELS = {"desktop": "Desktop", "jetson": "Orin Nano"}
PLATFORM_ORDER = ["desktop", "jetson"]
SYSTEM_ORDER = ["ORB-SLAM3", "Nitro-SLAM"]


def split_variant(variant):
    """(system, machine) for a variant path.

    A variant is everything above the sequence -- "ORB-SLAM3/desktop" but
    "Nitro-SLAM/<kernel config>/desktop" -- so the system is the first component
    and the platform the last, with any kernel-config level in between.
    """
    parts = variant.split("/")
    return parts[0], parts[-1]


def order_variants(variants):
    """Variants grouped by platform, baseline before contender within each.

    Sorting the raw paths would group by system instead, which puts a sequence's
    two desktop numbers at opposite ends of the row -- the comparison a reader
    makes first is within a platform, so that is the grouping.
    """
    def key(variant):
        system, machine = split_variant(variant)
        return (PLATFORM_ORDER.index(machine) if machine in PLATFORM_ORDER
                else len(PLATFORM_ORDER),
                SYSTEM_ORDER.index(system) if system in SYSTEM_ORDER
                else len(SYSTEM_ORDER),
                variant)
    return sorted(variants, key=key)


def column_labels(variants):
    """Shortest unambiguous column header per variant.

    The system name alone is the useful header whenever it is unique, which it
    is for a baseline-vs-ours table on one platform. Across platforms the same
    system appears twice, and the platform is carried by a spanning header
    above rather than repeated in every column, so the system name still
    suffices. Only a genuine ambiguity within one platform -- two kernel
    configs of Nitro-SLAM -- falls back to the full path.
    """
    labels = {}
    for variant in variants:
        system, machine = split_variant(variant)
        clash = [v for v in variants
                 if split_variant(v) == (system, machine) and v != variant]
        labels[variant] = variant.replace("/", " / ") if clash else system
    return labels


def latex_table(rows, key, bold_best, decimals=3):
    """Render the reported ATEs as a booktabs table: sequences down, systems across.

    With more than one platform present the header gains a spanning row, so the
    columns read as two blocks rather than four unrelated ones, and the best value
    is bolded within each platform -- ORB-SLAM3's desktop number against
    Nitro-SLAM's desktop number, not against a Jetson one.
    """
    table = {}
    for row in rows:
        variant, _, sequence = row["sequence"].rpartition("/")
        table.setdefault(sequence, {})[variant] = float(row[key])
    variants = order_variants({v for cols in table.values() for v in cols})
    labels = column_labels(variants)

    # Contiguous runs of one platform, in column order: (machine, [variants]).
    groups = []
    for variant in variants:
        machine = split_variant(variant)[1]
        if groups and groups[-1][0] == machine:
            groups[-1][1].append(variant)
        else:
            groups.append((machine, [variant]))

    out = ["\\begin{tabular}{l%s}" % ("r" * len(variants)), "\\toprule"]
    if len(groups) > 1:
        spans, rules, col = [""], [], 2
        for machine, members in groups:
            spans.append("\\multicolumn{%d}{c}{%s}"
                         % (len(members), escape(PLATFORM_LABELS.get(machine, machine))))
            rules.append("\\cmidrule(lr){%d-%d}" % (col, col + len(members) - 1))
            col += len(members)
        out.append(" & ".join(spans) + " \\\\")
        out.append("".join(rules))
    out.append("Sequence & %s \\\\"
               % " & ".join(escape(labels[v]) for v in variants))
    out.append("\\midrule")

    for sequence in sorted(table):
        cols = table[sequence]
        # Rank on the rounded value, not on the printed string: "17.682" sorts
        # below "9.956" as text, which would bold the worse run of any pair
        # straddling ten. Rounding first still makes two runs that both print
        # 0.038 a tie, so neither is singled out for a difference the reader
        # cannot see.
        shown = {v: "%.*f" % (decimals, a) for v, a in cols.items()}
        ranked = {v: round(a, decimals) for v, a in cols.items()}
        best = {}
        for machine, members in groups:
            present = [ranked[v] for v in members if v in ranked]
            if bold_best and len(present) > 1:
                best[machine] = min(present)
        cells = []
        for variant in variants:
            value = shown.get(variant)
            machine = split_variant(variant)[1]
            if value is None:
                cells.append("--")
            elif machine in best and ranked[variant] == best[machine]:
                cells.append("\\textbf{%s}" % value)
            else:
                cells.append(value)
        out.append("%s & %s \\\\" % (escape(sequence), " & ".join(cells)))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def keep(name, only, exclude):
    """Apply the --only/--exclude globs to one sequence name."""
    if only and not any(fnmatch.fnmatch(name, pat) for pat in only):
        return False
    return not any(fnmatch.fnmatch(name, pat) for pat in exclude)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["Results-euroc"],
                    help="results roots to scan; give a desktop and a Jetson root "
                         "to get both platforms in one table "
                         "(default: Results-euroc)")
    ap.add_argument("--metric", default="rmse", choices=sorted(ATE_RE),
                    help="ATE statistic to rank runs by (default: rmse)")
    ap.add_argument("--csv", metavar="FILE",
                    help="also write the table to FILE as CSV")
    ap.add_argument("--copy", metavar="OUTDIR",
                    help="copy each median run's trajectory/ into "
                         "OUTDIR/<sequence path>/")
    ap.add_argument("--only", metavar="PATTERN", action="append", default=[],
                    help="keep only sequences whose name matches this glob "
                         "(repeatable, e.g. --only 'room*')")
    ap.add_argument("--exclude", metavar="PATTERN", action="append", default=[],
                    help="drop sequences whose name matches this glob "
                         "(repeatable, e.g. --exclude 'outdoors*')")
    ap.add_argument("--agg", default="median",
                    choices=("median", "best", "worst", "mean"),
                    help="how to reduce a sequence's runs to one number: "
                         "median (default), best/worst (min/max ATE), or mean")
    ap.add_argument("--latex", action="store_true",
                    help="print a LaTeX table of the median ATEs (implies --quiet)")
    ap.add_argument("--tex", metavar="FILE",
                    help="write the LaTeX table to FILE as well (implies --latex)")
    ap.add_argument("--decimals", type=int, default=3, metavar="N",
                    help="decimal places in the LaTeX table (default: 3); "
                         "use 4 when sub-centimetre sequences would otherwise "
                         "round into false ties")
    ap.add_argument("--no-bold", action="store_true",
                    help="with --latex, do not bold the best value in each row")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the median rows, not the per-run ranking")
    args = ap.parse_args()
    args.latex = args.latex or bool(args.tex)
    args.quiet = args.quiet or args.latex

    for root in args.roots:
        if not os.path.isdir(root):
            sys.exit("no such results root: %s" % root)

    sequences = find_sequences(args.roots)
    if not sequences:
        sys.exit("no runs (directories with an ostream.txt) under %s"
                 % ", ".join(args.roots))

    # Filter on the sequence name alone, not the whole path, so one --exclude
    # drops that sequence across every system in the root -- half a comparison
    # row is worse than no row.
    sequences = {seq: runs for seq, runs in sequences.items()
                 if keep(os.path.basename(seq), args.only, args.exclude)}
    if not sequences:
        sys.exit("no sequences left after --only/--exclude")

    key = "%s_ate_%s" % (args.agg, args.metric)
    rows = []
    for seq in sorted(sequences):
        value, run_dir, scored, failed = aggregate_runs(sequences[seq],
                                                        args.metric, args.agg)
        if value is None:
            print("%-60s no run produced an ATE (%d run(s) skipped)"
                  % (seq, len(failed)))
            continue
        rows.append({
            "sequence": seq,
            "run": os.path.basename(run_dir),
            key: "%.6f" % value,
            "runs_ranked": len(scored),
            "runs_failed": len(failed),
            "run_dir": run_dir,
            "trajectory": next((f for f in trajectory_files(run_dir)
                                if os.path.basename(f).startswith("f_")
                                and f.endswith(".txt")), ""),
        })

        if not args.latex:
            print("%-58s %s %s=%.6f m  run %-4s (%d run(s)%s)"
                  % (seq, args.agg, args.metric, value,
                     os.path.basename(run_dir),
                     len(scored),
                     ", %d without an ATE" % len(failed) if failed else ""))
        if not args.quiet:
            for rank, (value, run) in enumerate(scored):
                print("      %s %-4s %.6f" % ("*" if run == run_dir else " ",
                                              os.path.basename(run), value))
        if (len(scored) % 2 == 0 and args.agg in ("median", "mean")
                and not args.latex):
            print("      note: %d runs -- the %s falls between runs; run %s "
                  "is the representative trajectory"
                  % (len(scored), args.agg, os.path.basename(run_dir)))

        if args.copy:
            dest = os.path.join(args.copy, seq)
            os.makedirs(dest, exist_ok=True)
            for path in trajectory_files(run_dir):
                shutil.copy2(path, os.path.join(dest, os.path.basename(path)))

    if args.latex and rows:
        body = latex_table(rows, key, not args.no_bold, args.decimals)
        print(body)
        if args.tex:
            os.makedirs(os.path.dirname(args.tex) or ".", exist_ok=True)
            with open(args.tex, "w") as fh:
                fh.write(body + "\n")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("\nwrote %s (%d sequence(s))" % (args.csv, len(rows)))
    if args.copy:
        print("copied median trajectories into %s/" % args.copy)


if __name__ == "__main__":
    main()
