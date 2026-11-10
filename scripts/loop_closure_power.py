#!/usr/bin/env python3
"""Board power over a run, ORB-SLAM3 against Nitro-SLAM, on the Jetson.

One line per system over the whole run, on a shared run-time axis starting at 0,
with each system's loop closure marked by a dashed rule in its own colour. The two
runs close the loop at different points (707 s and 739 s into outdoors5), so there
are two markers rather than one; the sequences are frame-paced, so a given run time
is the same point in the dataset for both.

Reads the 10 Hz power.csv and the events.csv that monitor_power.py writes beside
each run of the jetson-power sweep:

    <root>/<system>/[<kernel status>/]jetson-power*/<sequence>/<iteration>/

The sampled series is noisy at 10 Hz -- the INA3221 integrates over its own
conversion window and the governor moves underneath it -- so the raw trace is drawn
faintly and a rolling mean over --smooth seconds on top of it. Both are the same
data; the mean is there to make the shape legible, not to hide the spread.

The figures the axis is not wide enough to show are printed instead. The baseline
each excursion is measured against is the run's own mean over [--pre, -5) s before
its closure -- what that run was already drawing while tracking the same sequence,
not the idle board. It differs between the systems (~9.3 W for ORB-SLAM3, ~8.7 W
for Nitro-SLAM), so the excess energies are each relative to their own run.

The LoopClosing timings are printed alongside because the power trace alone invites
the wrong reading -- see the note above GBA_KF_LIMIT.

The machine directory differs between generations -- `jetson-power` in the v1 trees
and `jetson-power-v2` in the new ones -- and the comparison pairs the v1 ORB-SLAM3
baseline with the new Nitro-SLAM sweep, so each system is discovered separately
across --results rather than under one shared machine name. --run overrides the
search outright.

Usage:
    ./loop_closure_power.py
    ./loop_closure_power.py --sequence outdoors5 --window 600 800   # zoom in
    ./loop_closure_power.py --run ORB-SLAM3=<dir> --run Nitro-SLAM=<dir>
    ./loop_closure_power.py --large-font      # loop_closure_power_<seq>_lf.png
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tracking_breakdown import (GRID, INK, INK_MUTED, OUT_ROOT, SLOTS, SURFACE,
                                scale_text)
from tracking_comparison import BASELINE, CONTENDER, GROUP_ORDER

# Both systems have a power sweep in both generations, but the comparison pairs the
# v1 ORB-SLAM3 baseline with the new Nitro-SLAM one (the sweep with GPU pose
# optimisation). Which generation a system is taken from is therefore a per-system
# preference, not a shared root order: PREFERRED roots are searched first and the
# rest are a fallback, so the pairing needs no symlink tree. A trailing separator
# keeps "data/jetson/" from also matching "data/jetson-v1/".
RESULTS_ROOTS = [os.path.join('data', 'jetson-v1', 'Results-tumvi'),
                 os.path.join('data', 'jetson-v1', 'Results-euroc'),
                 os.path.join('data', 'jetson', 'Results-tumvi'),
                 os.path.join('data', 'jetson', 'Results-euroc')]
PREFERRED = {BASELINE: os.path.join('data', 'jetson-v1') + os.sep,
             CONTENDER: os.path.join('data', 'jetson') + os.sep}
MACHINE_GLOB = 'jetson-power*'
EVENT = 'loop_found'              # monitor_power.py's marker for "Good loop found!"
# Slots 0 and 1 of the shared categorical order, so this figure reads as one set
# with the timing and breakdown charts. Assigned in order, never cycled.
COLORS = {BASELINE: SLOTS[0], CONTENDER: SLOTS[1]}
# Sized to match the wide grouped comparisons rather than the 5in single-column
# figures: it runs beside them and a time axis has more to resolve than eight bar
# groups. The width is their rule -- 1.45in per group, from GROUP_ORDER, so it
# follows tracking_comparison.py if the families change. The height is not their
# 4.0: those stack a legend above the axes, which the tight bbox then includes, so
# they save at about 4.67in while a bare 4.0 axes here saves at 4.12. The 0.55
# makes the two render to the same height.
WIDE_FIGSIZE = (1.45 * len(GROUP_ORDER), 4.0 + 0.55)
# The sizes in the drawing code below are the ones the 5in single-column figures
# use, which come out small on a canvas this wide. The grouped comparisons scale
# theirs by 1.5 for exactly that reason, so this matches them.
FONT_SCALE = 1.5
# --large-font multiplies the scale above by this, for a copy that stays legible
# when the figure is shrunk to a column or projected on a slide. The traces are
# thickened by the same factor so the lines do not look thin against the text.
LARGE_FONT = 1.5


def find_run(roots, system, sequence):
    """The first <root>/<system>/**/jetson-power*/<sequence>/<iter> that exists.

    Nitro-SLAM inserts a kernel-status directory between the system and the machine
    and ORB-SLAM3 does not, so the middle is globbed rather than named. Roots the
    system prefers (see PREFERRED) are tried first, in their given order.
    """
    prefix = PREFERRED.get(system)
    if prefix:
        roots = ([r for r in roots if r.startswith(prefix)]
                 + [r for r in roots if not r.startswith(prefix)])
    for root in roots:
        for depth in ('', '*/'):
            pattern = os.path.join(root, system, depth + MACHINE_GLOB, sequence, '*')
            for path in sorted(glob.glob(pattern)):
                if os.path.isfile(os.path.join(path, 'power.csv')):
                    return path
    return None


def read_power(run_dir):
    """(t_s, power_W, gpu_pct) for one run, dropping rows the monitor left blank."""
    t, watts, gpu = [], [], []
    with open(os.path.join(run_dir, 'power.csv')) as fh:
        for row in csv.DictReader(fh):
            try:
                t.append(float(row['t_s']))
                watts.append(float(row['power_in_mw']) / 1000.0)
            except (KeyError, TypeError, ValueError):
                continue
            try:
                gpu.append(float(row['gpu_load_pct']))
            except (KeyError, TypeError, ValueError):
                gpu.append(np.nan)
    return np.asarray(t), np.asarray(watts), np.asarray(gpu)


def read_event(run_dir, marker, index=0):
    """t_s of the `index`th occurrence of `marker` in events.csv, or None."""
    path = os.path.join(run_dir, 'events.csv')
    if not os.path.isfile(path):
        return None
    hits = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get('marker') == marker:
                try:
                    hits.append(float(row['t_s']))
                except (TypeError, ValueError):
                    continue
    if not hits:
        return None
    return hits[index] if -len(hits) <= index < len(hits) else None


# What the LoopClosing thread recorded for the closure, printed beside the power so
# the trace is not read as "this system closes loops cheaply". On a large inertial
# map the two systems do different amounts of work, and that is the whole point:
# LoopClosing.cc:1694 only launches global BA when the map is not IMU-initialised,
# or holds fewer than GBA_KF_LIMIT keyframes, or the GPU global BA kernel is on. So
# stock ORB-SLAM3 declines global BA outright on outdoors5 (1501 keyframes at the
# closure) and pays only for the pose-graph correction, while Nitro-SLAM runs the
# global BA the baseline skipped -- the flat trace is a smaller job, not a cheaper
# one.
GBA_KF_LIMIT = 200            # LoopClosing.cc: KeyFramesInMap() < 200
LOOP_STATS = [('loopCorrection_time', 'loop correction'),
              ('graphOptimization_time', 'graph optimisation'),
              ('globalBA_time', 'global BA')]


def read_series(path):
    """[(key, value)] from a "<key>: <value>" stats dump, or [] if absent."""
    if not os.path.isfile(path):
        return []
    out = []
    for line in open(path):
        key, _, value = line.partition(':')
        try:
            out.append((int(key), float(value)))
        except ValueError:
            continue
    return out


def read_loop_stats(run_dir):
    """({label: ms}, keyframes at the closure) from the LoopClosing stats dump.

    A phase absent from the dict recorded nothing. For global BA that means it was
    never launched -- see GBA_KF_LIMIT -- not that it was launched and lost.
    """
    data = os.path.join(run_dir, 'LoopClosing', 'data')
    timings = {}
    for name, label in LOOP_STATS:
        values = [v for _, v in read_series(os.path.join(data, f'{name}.txt'))]
        if values:
            timings[label] = sum(values)

    # numKFs is per LoopClosing iteration, so the map size that the GBA guard saw
    # is the sample at the iteration where loopClosed first flipped to 1.
    closed = [k for k, v in read_series(os.path.join(data, 'loopClosed.txt')) if v]
    kfs = dict(read_series(os.path.join(data, 'numKFs.txt')))
    at_closure = kfs.get(closed[0]) if closed else None
    return timings, at_closure


def rolling_mean(y, t, seconds):
    """Centred rolling mean over a window of `seconds`, in samples."""
    if seconds <= 0 or y.size < 3:
        return y
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.1
    n = max(1, int(round(seconds / dt)))
    if n <= 1:
        return y
    kernel = np.ones(n) / n
    # 'same' would taper the ends toward zero; dividing by the same convolution of
    # ones normalises each end by however many samples it actually covered.
    return np.convolve(y, kernel, 'same') / np.convolve(np.ones_like(y), kernel, 'same')


def excursion(rel, watts, pre, settle):
    """(baseline_W, peak_W, excess_J) for one run around its event.

    The baseline is the run's own level over [pre, -5) s. Excess energy integrates
    power above that baseline over [0, settle] s, clipped at zero -- a sample below
    the pre-loop level is noise, not the loop closure giving power back.
    """
    before = (rel >= pre) & (rel < -5.0)
    after = (rel >= 0.0) & (rel <= settle)
    if not before.any() or not after.any():
        return None
    base = float(watts[before].mean())
    excess = float(np.trapezoid(np.maximum(watts[after] - base, 0.0), rel[after]))
    return base, float(watts[after].max()), excess


def plot(series, events, args, out_path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': SURFACE})
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.patch.set_facecolor(SURFACE)

    # One marker per system, in that system's colour, because the two runs close the
    # loop at different points in the sequence (707 s and 739 s on outdoors5). They
    # sit behind the traces: a marker is not a series.
    lw = LARGE_FONT if args.large_font else 1.0
    for system, event_t in events.items():
        ax.axvline(event_t, color=COLORS.get(system, INK_MUTED), linewidth=0.9 * lw,
                   linestyle=(0, (4, 3)), alpha=0.7, zorder=1)
    if events:
        # 30 s apart on a 1150 s axis is close enough that two labels would collide,
        # so one label is centred over the pair and the colours say which is whose.
        ax.text(sum(events.values()) / len(events), 1.005, 'loop closure',
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=9, color=INK_MUTED)

    for system, (t, watts, _) in series.items():
        color = COLORS.get(system, SLOTS[0])
        ax.plot(t, watts, color=color, linewidth=0.5 * lw, alpha=0.25, zorder=2)
        ax.plot(t, rolling_mean(watts, t, args.smooth), color=color,
                linewidth=2.0 * lw, zorder=3, label=system,
                solid_capstyle='round', solid_joinstyle='round')

    lo = args.window[0] if args.window else 0.0
    hi = args.window[1] if args.window else max(t.max() for t, _, _ in series.values())
    ax.set_xlim(lo, hi)
    ax.set_xlabel('Run time (s)', color=INK_MUTED, fontsize=10)
    ax.set_ylabel('Board Power (W)', color=INK_MUTED, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_facecolor(SURFACE)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#d6d5d0')
    ax.tick_params(colors=INK_MUTED, labelsize=10, length=0)
    ax.legend(loc='upper right', frameon=False, fontsize=9.5, labelcolor=INK_MUTED,
              handlelength=1.6, borderpad=0, labelspacing=0.4)
    if args.note:
        fig.text(0.011, 0.008, args.note, fontsize=8.5, color=INK_MUTED, style='italic')

    font_scale = args.font_scale * (LARGE_FONT if args.large_font else 1.0)
    if font_scale != 1.0:
        # Last, so it scales what was drawn rather than what the sizes imply, and
        # before tight_layout, so the layout is made around the grown text.
        scale_text(fig, font_scale)
    # The grouped comparisons fill their canvas, so their tight bbox comes out at
    # the figure size plus padding. A lone axes does not -- tight would trim the
    # default margins and hand back a narrower image than the figsize asked for --
    # so the axes are expanded to fill first and the two then render to match.
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=800, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sequence', default='outdoors5', help='sequence to plot')
    ap.add_argument('--results', nargs='*', default=RESULTS_ROOTS,
                    help='results roots to search, in order')
    ap.add_argument('--systems', nargs='*', default=[BASELINE, CONTENDER],
                    help='systems to plot, in this order')
    ap.add_argument('--run', action='append', default=[], metavar='SYSTEM=DIR',
                    help='use this run directory for a system instead of searching')
    ap.add_argument('--event', default=EVENT,
                    help='events.csv marker to align on (default: %(default)s)')
    ap.add_argument('--event-index', type=int, default=0,
                    help='which occurrence of the marker, when there are several')
    ap.add_argument('--window', nargs=2, type=float, default=None,
                    metavar=('FROM', 'TO'),
                    help='restrict the x axis to this span of run time; '
                         'the default plots the whole run')
    ap.add_argument('--smooth', type=float, default=1.0,
                    help='rolling-mean window in seconds; 0 plots the raw trace only')
    ap.add_argument('--settle', type=float, default=30.0,
                    help='seconds after the event to integrate excess energy over')
    ap.add_argument('--pre', type=float, default=-60.0,
                    help='start of the pre-event window the baseline is taken from')
    ap.add_argument('--figsize', nargs=2, type=float, default=list(WIDE_FIGSIZE),
                    metavar=('W', 'H'),
                    help='figure size in inches (default: the wide comparison size)')
    ap.add_argument('--font-scale', type=float, default=FONT_SCALE,
                    help='multiply every text size by this (default: %(default)s, '
                         'what the grouped comparisons use)')
    ap.add_argument('--large-font', action='store_true',
                    help=f'scale text and lines a further {LARGE_FONT}x and write '
                         f'the figure with an _lf suffix')
    ap.add_argument('--note', default='', help='italic note under the figure')
    # This script's default pairing is the v2 comparison (v1 ORB-SLAM3 baseline
    # against the new Nitro-SLAM sweep), so its figures land beside the rest of
    # that set rather than in the v1 directory.
    ap.add_argument('--out', default=os.path.join(OUT_ROOT, 'v2'),
                    help='directory for the figure')
    args = ap.parse_args()

    overrides = {}
    for item in args.run:
        system, _, path = item.partition('=')
        if not path:
            print(f'--run wants SYSTEM=DIR, got "{item}"', file=sys.stderr)
            return 1
        overrides[system] = path

    series, events, stats = {}, {}, {}
    for system in args.systems:
        run_dir = overrides.get(system) or find_run(args.results, system, args.sequence)
        if not run_dir:
            print(f'  warning: no {system} power run for {args.sequence}',
                  file=sys.stderr)
            continue
        t, watts, gpu = read_power(run_dir)
        event_t = read_event(run_dir, args.event, args.event_index)
        if event_t is None:
            print(f'  warning: {system} logged no "{args.event}" in {run_dir}',
                  file=sys.stderr)
            continue
        rel = t - event_t
        keep = (np.ones_like(t, dtype=bool) if not args.window else
                (t >= args.window[0]) & (t <= args.window[1]))
        if not keep.any():
            print(f'  warning: {system} has no samples in the window', file=sys.stderr)
            continue
        series[system] = (t[keep], watts[keep], gpu[keep])
        events[system] = event_t
        # GPU is quoted over the excursion, not the whole run: the question is what
        # the closure put on the GPU, and a run-long mean would bury it.
        burst = (rel >= 0.0) & (rel <= args.settle)
        stats[system] = (run_dir, event_t, gpu[burst],
                         excursion(rel, watts, args.pre, args.settle),
                         read_loop_stats(run_dir))

    if not series:
        print(f'no power runs found for {args.sequence} under '
              f'{", ".join(args.results)}', file=sys.stderr)
        return 1

    span = (f'run time [{args.window[0]:g}, {args.window[1]:g}] s' if args.window
            else 'whole run')
    print(f'{args.sequence} · closure marked from "{args.event}" · {span}')
    for system, (run_dir, event_t, gpu, exc, (timings, kfs)) in stats.items():
        print(f'\n{system}')
        print(f'  run           {run_dir}')
        print(f'  event at      {event_t:.1f} s into the run')
        if exc:
            base, peak, excess = exc
            print(f'  baseline      {base:.2f} W   (mean over [{args.pre:g}, -5) s)')
            print(f'  peak          {peak:.2f} W   (+{peak - base:.2f} W over baseline)')
            print(f'  excess energy {excess:.1f} J   (over [0, {args.settle:g}] s)')
        finite = gpu[np.isfinite(gpu)]
        if finite.size:
            print(f'  GPU load      {finite.mean():.1f}% mean, {finite.max():.1f}% peak'
                  f'   (over [0, {args.settle:g}] s after the closure)')
        if kfs is not None:
            print(f'  map at loop   {kfs:.0f} keyframes')
        for _, label in LOOP_STATS:
            if label in timings:
                print(f'  {label:<20}{timings[label] / 1000.0:.2f} s')
            elif label == 'global BA':
                # Not a missing probe: the guard at LoopClosing.cc:1694 skipped it.
                over = ('' if kfs is None else
                        f', map has {kfs:.0f} >= {GBA_KF_LIMIT} keyframes')
                print(f'  {label:<20}not run{over}')

    os.makedirs(args.out, exist_ok=True)
    suffix = '_lf' if args.large_font else ''
    out_path = os.path.join(args.out,
                            f'loop_closure_power_{args.sequence}{suffix}.png')
    plot(series, events, args, out_path)
    print(f'\nwrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
