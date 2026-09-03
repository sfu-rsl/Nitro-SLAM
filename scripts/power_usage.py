#!/usr/bin/env python3
"""CPU load, GPU load and power draw on the Jetson, per sequence and per system.

Reads the 10 Hz power.csv the power monitor writes beside each run of the
jetson-power sweep:

    <results root>/<system>/[<kernel status>/]jetson-power/<dataset>/<iteration>/

Each sequence was run once here, unlike the timing sweep's five, so there is no
spread across repeats to report: min, mean and max are taken over the samples
within a run, which is what the question "how hard does it push the board" is
actually asking. The summary across sequences takes the lowest minimum, the
highest maximum, and the mean of the per-sequence means -- sequences weighted
equally, so a 900-second outdoors run does not outvote a 100-second room one.

Two readings are given for CPU and for power, because neither has one honest
number:

  CPU (all cores) the six cores summed and normalised, so 100% is the whole board
                  busy -- everything running, not just the run
  CPU (process)   the SLAM process alone, on that same scale: /proc reports it in
                  cores, where 100% is one core and 600% is six, and it is divided
                  by the core count here so the two rows share a denominator
  Power (board)   total input power, which includes everything the board does
                  when idle
  Power (net)     the same minus that run's own measured baseline, which is what
                  running SLAM actually costs

The baseline is read per run rather than assumed: across the 78 runs it varies
from 8135 to 8923 mW, enough to matter against a net draw of about 1 W.

Usage:
    ./power_usage.py
    ./power_usage.py --per-sequence
    ./power_usage.py --csv
    ./power_usage.py --latex
"""

import argparse
import csv
import os
import sys

import numpy as np

from tracking_breakdown import OUT_ROOT, PLATFORMS
from tracking_comparison import BASELINE, CONTENDER, GROUP_ORDER, group_of, resolve_system

MACHINE = 'jetson-power'
CORES = 6                     # cpu0_pct .. cpu5_pct
# (column, label, unit, scale). 'power_net_mw' is derived, not a column.
METRICS = [
    ('cpu_total_pct', 'CPU (all cores)', '%', 1.0),
    # /proc counts the process in cores -- 100% is one core busy, 600% is six -- so
    # it is divided by the core count here. Both CPU rows are then the same thing
    # measured against the same denominator: the share of the whole board's CPU.
    ('proc_cpu_pct', 'CPU (process)', '%', 1.0 / CORES),
    ('gpu_load_pct', 'GPU (load)', '%', 1.0),
    ('power_in_mw', 'Power (board)', 'W', 1e-3),
    ('power_net_mw', 'Power (net)', 'W', 1e-3),
]


def read_summary(run_dir):
    """{key: float} from power_summary.txt, for the run's measured baseline."""
    path = os.path.join(run_dir, 'power_summary.txt')
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def read_run(run_dir):
    """{metric: (min, mean, max)} over one run's samples, or None if unreadable.

    Samples before the process appears and after it exits are dropped: the monitor
    is started first and stopped last, and those samples describe an idle board
    rather than the run. They are few -- usually one -- but they would drag every
    minimum to the idle value and make the four systems look identical there.
    """
    path = os.path.join(run_dir, 'power.csv')
    if not os.path.isfile(path):
        return None
    columns = {name: [] for name, _, _, _ in METRICS if name != 'power_net_mw'}
    proc = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                for name in columns:
                    columns[name].append(float(row[name]))
                proc.append(float(row['proc_cpu_pct']))
            except (KeyError, TypeError, ValueError):
                continue
    if not proc:
        return None

    # The first sample is a rate with nothing behind it: proc_cpu_pct there is CPU
    # time accumulated since the process started, divided by an interval that has
    # not elapsed yet. It comes out as 0 on most runs and as 3938% on one of them,
    # well past the 600% six cores allow, so it is dropped rather than trimmed on
    # its value.
    proc = np.asarray(proc)[1:]
    columns = {name: values[1:] for name, values in columns.items()}
    if not proc.size:
        return None

    active = np.flatnonzero(proc > 0)
    if not active.size:
        return None
    lo, hi = int(active[0]), int(active[-1]) + 1

    baseline = read_summary(run_dir).get('baseline_power_mw')
    series = {name: np.asarray(values[lo:hi]) for name, values in columns.items()}
    if baseline is not None:
        # Clipped at zero: a sample below the idle baseline is measurement noise,
        # not the process giving power back.
        series['power_net_mw'] = np.maximum(series['power_in_mw'] - baseline, 0.0)

    return {name: (float(v.min()), float(v.mean()), float(v.max()))
            for name, v in series.items()}, hi - lo


def collect(roots, systems, kernel=None):
    """{(system, dataset): stats}, and the sequences found, in results-root order."""
    agg, found = {}, []
    paths = {s: resolve_system(roots, s, MACHINE, kernel) for s in systems}
    for root in roots:
        for system in systems:
            base = os.path.join(root, paths[system] or system, MACHINE)
            if not os.path.isdir(base):
                continue
            for dataset in sorted(os.listdir(base)):
                seq_dir = os.path.join(base, dataset)
                if not os.path.isdir(seq_dir):
                    continue
                runs = [read_run(os.path.join(seq_dir, i))
                        for i in sorted(os.listdir(seq_dir))]
                runs = [r for r in runs if r]
                if not runs:
                    continue
                if len(runs) > 1:
                    print(f'  note: {system}/{dataset} has {len(runs)} runs; '
                          f'using the first', file=sys.stderr)
                if dataset not in found:
                    found.append(dataset)
                stats, samples = runs[0]
                agg[(system, dataset)] = dict(stats, n_samples=samples)
    return agg, found


def summarise(agg, system, datasets):
    """{metric: (min, mean, max, sd)} over the sequences a system ran.

    The extremes are the extremes anywhere -- the lowest sample of any sequence and
    the highest -- while the middle is the mean of the per-sequence means, so every
    sequence counts once however long it ran.

    The spread is across sequences, not across samples or repeats. Each sequence
    was run once here, so there is no run-to-run spread to quote; what varies is
    how hard one sequence works the board compared with another, and that is what
    a reader of a single summary number needs to know about it.
    """
    out = {}
    for name, _, _, _ in METRICS:
        rows = [agg[(system, d)][name] for d in datasets
                if (system, d) in agg and name in agg[(system, d)]]
        if rows:
            means = [r[1] for r in rows]
            out[name] = (min(r[0] for r in rows),
                         float(np.mean(means)),
                         max(r[2] for r in rows),
                         float(np.std(means, ddof=1)) if len(means) > 1 else 0.0)
    return out


def print_summary(agg, systems, datasets, title):
    width = max(len(s) for s in systems) + 2
    print(f'\n{title}')
    print(f'{"metric":22}{"":{width}}{"min":>9}{"mean":>9}{"sd":>8}{"max":>9}')
    print('-' * (22 + width + 35))
    summaries = {s: summarise(agg, s, datasets) for s in systems}
    for name, label, unit, scale in METRICS:
        for i, system in enumerate(systems):
            stats = summaries[system].get(name)
            if not stats:
                continue
            lo, mid, hi, sd = (v * scale for v in stats)
            print(f'{label if i == 0 else "":22}{system:{width}}'
                  f'{lo:9.1f}{mid:9.1f}{sd:8.1f}{hi:9.1f}')
        # The ratio is what the comparison is for, and only the means support one.
        both = [summaries[s].get(name) for s in systems]
        if len(systems) == 2 and all(both) and both[0][1] > 0:
            print(f'{"":22}{"ratio":{width}}{"":9}'
                  f'{both[1][1] / both[0][1]:9.2f}{"":8}{"":9}')
        print()


def print_per_sequence(agg, systems, datasets):
    for name, label, unit, scale in METRICS:
        print(f'\n{label} ({unit})')
        head = f'{"sequence":13}'
        for system in systems:
            head += f'{system[:11] + " min":>16}{"mean":>8}{"max":>8}'
        print(head)
        print('-' * len(head))
        for dataset in datasets:
            line = f'{dataset:13}'
            for system in systems:
                stats = agg.get((system, dataset), {}).get(name)
                if stats:
                    lo, mid, hi = (v * scale for v in stats)
                    line += f'{lo:16.1f}{mid:8.1f}{hi:8.1f}'
                else:
                    line += f'{"--":>16}{"--":>8}{"--":>8}'
            print(line)


# The one-line-per-system summary: what each system costs, meaned over sequences.
# ORB-SLAM3 has no GPU column of its own -- see the note in latex_table.
LATEX_COLUMNS = [
    # Board load, not the process's own share: the GPU column can only be board
    # load -- nothing reports GPU occupancy per process -- so measuring CPU the
    # same way keeps the two utilisation columns on one footing.
    ('cpu_total_pct', 'CPU (\\%)', 1.0),
    ('gpu_load_pct', 'GPU (\\%)', 1.0),
    ('power_in_mw', 'Power (W)', 1e-3),
    ('power_net_mw', 'Net Power (W)', 1e-3),
]


def latex_table(agg, systems, datasets, gpu_systems=(CONTENDER,), decimals=1):
    """A booktabs row per system: mean CPU, GPU and power over the sequences.

    The spread is across sequences -- see summarise() -- since each was run once.

    Both utilisation columns are board load, measured the same way: nothing reports
    GPU occupancy per process, so the GPU column cannot be attributed to a system,
    and CPU is read board-wide to match rather than mixing scopes in one row. The
    per-process figures are still in --per-sequence and the CSV.

    The baseline's GPU cell is left empty rather than filled with the 0.1% the
    board registers while ORB-SLAM3 runs, which is not ORB-SLAM3 using the GPU;
    `gpu_systems` names the systems the column applies to.
    """
    out = ['\\begin{tabular}{l' + 'r' * len(LATEX_COLUMNS) + '}', '\\toprule',
           'System & ' + ' & '.join(h for _, h, _ in LATEX_COLUMNS) + ' \\\\',
           '\\midrule']
    for system in systems:
        stats = summarise(agg, system, datasets)
        cells = []
        for name, _, scale in LATEX_COLUMNS:
            if name == 'gpu_load_pct' and system not in gpu_systems:
                cells.append('--')
            elif name in stats:
                mean, sd = stats[name][1] * scale, stats[name][3] * scale
                cells.append(f'{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}')
            else:
                cells.append('--')
        out.append(f'{system} & ' + ' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(out)


def write_csv(path, agg, systems, datasets):
    rows = ['system,dataset,metric,unit,min,mean,max,n_samples']
    for dataset in datasets:
        for system in systems:
            entry = agg.get((system, dataset))
            if not entry:
                continue
            for name, label, unit, scale in METRICS:
                if name not in entry:
                    continue
                lo, mid, hi = (v * scale for v in entry[name])
                rows.append(f'{system},{dataset},{label},{unit},'
                            f'{lo:.3f},{mid:.3f},{hi:.3f},{entry["n_samples"]}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(rows) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', nargs='*', default=PLATFORMS['jetson']['roots'],
                    help='results roots to search, in order')
    ap.add_argument('--systems', nargs='*', default=[BASELINE, CONTENDER],
                    help='systems to report, in this order')
    ap.add_argument('--sequences', nargs='*',
                    help='restrict to these sequences; default is every one found')
    ap.add_argument('--nitro-kernel', help='kernel-status directory to read for '
                                           'Nitro-SLAM, when the tree holds several')
    ap.add_argument('--per-sequence', action='store_true',
                    help='also print every sequence, not just the summary')
    ap.add_argument('--by-family', action='store_true',
                    help='also summarise each sequence family separately')
    ap.add_argument('--out', default=OUT_ROOT, help='directory for the CSV')
    ap.add_argument('--latex', action='store_true',
                    help='write a booktabs summary table, one row per system')
    ap.add_argument('--csv', action='store_true',
                    help='write the per-sequence table to power_usage.csv')
    args = ap.parse_args()

    agg, found = collect(args.results, args.systems, args.nitro_kernel)
    if not agg:
        print(f'no power data under {", ".join(args.results)}/*/{MACHINE}')
        return 1
    datasets = [d for d in (args.sequences or found) if d in found]

    for system in args.systems:
        missing = [d for d in datasets if (system, d) not in agg]
        if missing:
            print(f'  warning: {system} has no run for {", ".join(missing)}')
    runs = sum(1 for d in datasets for s in args.systems if (s, d) in agg)
    print(f'{len(datasets)} sequences · {", ".join(args.systems)} · {runs} runs · '
          f'{sum(agg[(s, d)]["n_samples"] for d in datasets for s in args.systems if (s, d) in agg):,} samples')

    if args.per_sequence:
        print_per_sequence(agg, args.systems, datasets)
    print_summary(agg, args.systems, datasets, 'over all sequences')
    if args.by_family:
        for label in GROUP_ORDER:
            members = [d for d in datasets if group_of(d) == label]
            if members:
                print_summary(agg, args.systems, members, f'{label} ({len(members)})')

    if args.latex:
        body = latex_table(agg, args.systems, datasets)
        print()
        print(body)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, 'power_usage.tex')
        with open(path, 'w') as fh:
            fh.write(body + '\n')
        print(f'\nwrote {path}')

    if args.csv:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, 'power_usage.csv')
        write_csv(path, agg, args.systems, datasets)
        print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
