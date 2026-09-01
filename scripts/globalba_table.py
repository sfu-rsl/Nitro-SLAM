#!/usr/bin/env python3
"""ORB-SLAM3 vs Nitro-SLAM global bundle adjustment, as a LaTeX table.

The global BA the Loop Closing thread launches after a closure, timed per call in

    <results root>/<system>/<machine>/<dataset>/<iteration>/LoopClosing/data/
        globalBA_time.txt

This is the loop-closing global BA, not the full inertial BA of IMU initialisation
(LocalMapping/imuInitFIBA_time), which is a different optimisation on a different
thread.

Rows default to the sequences the loop-closing figures use, so the table and those
figures describe the same runs. Closing a loop is necessary but not sufficient for
global BA: ORB-SLAM3 launches it only for a map whose IMU is not yet initialised or
that holds few keyframes, so on the larger sequences it closes the loop and skips
the optimisation, leaving an empty series -- which is a "--" here, not a zero.

A cell carries the number of runs behind it, since the two systems do not run
global BA equally often. It is the mean cost of one call, averaged first within a
run and then across runs, with the spread across runs -- the same two-stage
aggregation the breakdown figures use. The speedup is ORB-SLAM3 / Nitro-SLAM.

Both platforms are columns of one table by default. Writes LaTeX (booktabs) to
--out, and a readable version to stdout.

Usage:
    ./globalba_table.py
    ./globalba_table.py --all-sequences
    ./globalba_table.py --platform desktop --out -
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

from loopclosing_breakdown import CLOSING_SEQUENCES
from tracking_breakdown import OUT_ROOT, PLATFORMS, load, select_platforms
from tracking_comparison import BASELINE, CONTENDER, resolve_system

SERIES = 'globalBA_time'
SUBDIR = ('LoopClosing', 'data')
OUT_NAME = 'globalba_table.tex'


def run_mean(run_dir):
    """(mean ms per global BA call, call count) for one run, or None if it never ran.

    A run that closed no loop has no series at all; one that closed has a value per
    call. The absence is not a zero -- there was no optimisation to time -- so the
    caller counts runs rather than averaging a missing run in.
    """
    s = load(os.path.join(run_dir, *SUBDIR, SERIES + '.txt'))
    if not s:
        return None
    return float(np.mean(list(s.values()))), len(s)


def collect(roots, machine, frag, sequences):
    """{sequence: (mean ms, std ms, runs, mean calls per run)} for one system."""
    per_seq = defaultdict(list)
    for dataset in sequences:
        for root in roots:
            seq_dir = os.path.join(root, frag, machine, dataset)
            if not os.path.isdir(seq_dir):
                continue
            for it in sorted(os.listdir(seq_dir)):
                got = run_mean(os.path.join(seq_dir, it))
                if got:
                    per_seq[dataset].append(got)

    out = {}
    for dataset, results in per_seq.items():
        means = [m for m, _ in results]
        out[dataset] = (float(np.mean(means)),
                        float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                        len(means),
                        float(np.mean([c for _, c in results])))
    return out


def all_sequences(roots, machine, frag):
    """Sequences that ran global BA at least once, in results-root order."""
    seqs = []
    for root in roots:
        base = os.path.join(root, frag, machine)
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            seq_dir = os.path.join(base, d)
            if d in seqs or not os.path.isdir(seq_dir):
                continue
            if any(run_mean(os.path.join(seq_dir, it))
                   for it in sorted(os.listdir(seq_dir))):
                seqs.append(d)
    return seqs


def gather(args):
    """[(platform label, {system: {sequence: stats}})], and the sequence order."""
    columns, order = [], [] if args.all_sequences else list(args.sequences)
    for name, roots, machine in select_platforms(args):
        resolved = {}
        for system in args.systems:
            frag = resolve_system(roots, system, machine, args.nitro_kernel)
            if frag is None:
                print(f'  warning: no runs for {system} on {name}', file=sys.stderr)
                continue
            resolved[system] = frag
        if not resolved:
            continue
        if args.all_sequences:
            for frag in resolved.values():
                for seq in all_sequences(roots, machine, frag):
                    if seq not in order:
                        order.append(seq)
        columns.append((PLATFORMS[name]['label'],
                        {s: collect(roots, machine, f, order)
                         for s, f in resolved.items()}))
    return columns, order


def cell(stats):
    """"mean ± std (runs)", or an em dash where the system never ran global BA.

    The run count is in every cell rather than a note under the table: with global
    BA the two systems rarely close the same number of loops, and a mean over one
    run should not read like a mean over five.
    """
    if stats is None:
        return '--'
    mean, std, runs, _ = stats
    body = f'{mean:.1f} $\\pm$ {std:.1f}' if std else f'{mean:.1f}'
    return f'{body} ({runs})'


def speedup(a, b, tex=True):
    if a is None or b is None or b[0] <= 0:
        return '--'
    return f'{a[0] / b[0]:.2f}' + (r'$\times$' if tex else 'x')


def paired(per_system, order, systems):
    """Sequences both systems ran global BA on -- the only ones a summary can use."""
    base, cont = systems
    return [d for d in order
            if d in per_system.get(base, {}) and d in per_system.get(cont, {})
            and per_system[cont][d][0] > 0]


def summary_cells(columns, order, systems, tex=True):
    """The summary row, over the paired sequences only.

    Averaging each system over whatever it happened to run would compare an
    ORB-SLAM3 mean over six sequences against a Nitro-SLAM mean over eleven, which
    is not a comparison at all -- global BA fires on different sequences for the
    two. So the row is restricted to the sequences both ran, and says so. The
    speedup is a geometric mean: these are ratios, and an arithmetic mean of ratios
    would let the largest sequence set the summary.
    """
    bold = (lambda s: rf'\textbf{{{s}}}') if tex else (lambda s: s)
    cells = [bold('Mean (paired)')]
    for _, per_system in columns:
        both = paired(per_system, order, systems)
        for label in systems:
            cells.append(f'{np.mean([per_system[label][d][0] for d in both]):.1f}'
                         if both else '--')
        if both:
            ratios = [per_system[systems[0]][d][0] / per_system[systems[1]][d][0]
                      for d in both]
            geo = float(np.exp(np.mean(np.log(ratios))))
            cells.append(bold(f'{geo:.2f}' + (r'$\times$' if tex else 'x')))
        else:
            cells.append('--')
    return cells


def latex(columns, order, systems):
    """The whole table: one row per sequence, three columns per platform."""
    base, cont = systems
    spec = 'l' + 'rrr' * len(columns)
    rows = [r'\begin{tabular}{' + spec + '}', r'\toprule']

    head = ['']
    rule = []
    for i, (label, _) in enumerate(columns):
        head.append(r'\multicolumn{3}{c}{' + label + '}')
        col = 2 + 3 * i
        rule.append(rf'\cmidrule(lr){{{col}-{col + 2}}}')
    rows.append(' & '.join(head) + r' \\')
    rows.append(''.join(rule))
    rows.append(' & '.join(['Sequence'] +
                           [c for _ in columns
                            for c in (base, cont, 'Speedup')]) + r' \\')
    rows.append(r'\midrule')

    for dataset in order:
        cells = [dataset.replace('_', r'\_')]
        for _, per_system in columns:
            a = per_system.get(base, {}).get(dataset)
            b = per_system.get(cont, {}).get(dataset)
            cells += [cell(a), cell(b), speedup(a, b)]
        rows.append(' & '.join(cells) + r' \\')

    rows.append(r'\midrule')
    rows.append(' & '.join(summary_cells(columns, order, systems)) + r' \\')

    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def plain(columns, order, systems):
    """The same numbers as aligned text, for reading in the terminal."""
    base, cont = systems
    head = f'{"sequence":13}' + ''.join(
        f'{label + " " + base:>26}{label + " " + cont:>26}{"speedup":>10}'
        for label, _ in columns)
    out = [head, '-' * len(head)]
    for dataset in order:
        line = f'{dataset:13}'
        for _, per_system in columns:
            a = per_system.get(base, {}).get(dataset)
            b = per_system.get(cont, {}).get(dataset)
            for stats in (a, b):
                txt = '--' if stats is None else \
                    f'{stats[0]:.1f} ± {stats[1]:.1f} ({stats[2]})'
                line += f'{txt:>26}'
            line += f'{speedup(a, b, tex=False):>10}'
        out.append(line)

    cells = summary_cells(columns, order, systems, tex=False)
    line = f'{cells[0]:13}'
    for i in range(len(columns)):
        a, b, s = cells[1 + 3 * i:4 + 3 * i]
        line += f'{a:>26}{b:>26}{s:>10}'
    out += ['-' * len(head), line]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--platform', nargs='*', choices=sorted(PLATFORMS),
                    help='platforms to tabulate (default: all of them)')
    ap.add_argument('--results', nargs='*',
                    help="results roots to search, in order; overrides --platform's")
    ap.add_argument('--sequences', nargs='*', default=CLOSING_SEQUENCES,
                    help='sequences to tabulate, in this order')
    ap.add_argument('--all-sequences', action='store_true',
                    help='tabulate every sequence that ran global BA instead')
    ap.add_argument('--systems', nargs=2, default=[BASELINE, CONTENDER],
                    metavar=('BASELINE', 'CONTENDER'),
                    help='the two systems to compare; the speedup is baseline/contender')
    ap.add_argument('--nitro-kernel', help='kernel-status directory to read for '
                                           'Nitro-SLAM, when the tree holds several')
    ap.add_argument('--machine', help="machine directory to read; overrides --platform's")
    ap.add_argument('--out', default=os.path.join(OUT_ROOT, OUT_NAME),
                    help='path for the LaTeX table, or - for stdout only')
    args = ap.parse_args()

    columns, order = gather(args)
    if not columns or not order:
        print('no FIBA data found', file=sys.stderr)
        return 1

    print(plain(columns, order, args.systems))
    body = latex(columns, order, args.systems)
    if args.out == '-':
        print()
        print(body)
    else:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(body + '\n')
        print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
