#!/usr/bin/env python3
"""Per-thread time breakdown for a Nitro-SLAM stats directory.

Reads the "<key>: <value>" series written by the Stats/ singletons and reports,
for each thread, the total time per iteration split into its phases plus the
unaccounted-for remainder ("other").

Requires the binary to have been built with -DREGISTER_STATS=ON.

Usage: ./breakdown.py <statsDir> [--csv <out.csv>]

  --csv joins every Loop Closing series into one row per iteration, so the
  closures can be selected in pandas:
      df = pd.read_csv(out); df[df.loopClosed == 1]
"""

import os
import sys
from collections import defaultdict


def load(path):
    """Return {key: summed value}. Series that fire more than once per iteration
    (searchByProjection) repeat their key, so values are summed, not overwritten."""
    out = defaultdict(float)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            try:
                out[int(k.strip())] += float(v.strip())
            except ValueError:
                continue
    return dict(out)


def report(title, data_dir, total_name, phases, notes=(), only_keys=None):
    total = load(os.path.join(data_dir, total_name + '.txt'))
    if total is not None and only_keys is not None:
        total = {k: v for k, v in total.items() if k in only_keys}
    print()
    print('=' * 78)
    print(f'  {title}')
    print('=' * 78)
    if not total:
        print(f'  no data: {os.path.join(data_dir, total_name + ".txt")}')
        print('  (was the binary built with -DREGISTER_STATS=ON?)')
        return

    keys = set(total)
    n = len(keys)
    total_sum = sum(total.values())

    series = {}
    missing = []
    for label, fname in phases:
        s = load(os.path.join(data_dir, fname + '.txt'))
        if s is None:
            missing.append(fname)
            continue
        series[label] = s

    print(f'  iterations: {n}    total: {total_sum:.1f} ms'
          f'    mean: {total_sum / n:.3f} ms/iteration')
    print()
    print(f'  {"phase":<26}{"total ms":>12}{"mean ms":>12}{"% total":>10}{"samples":>10}')
    print('  ' + '-' * 70)

    accounted = 0.0
    for label, s in series.items():
        # Restrict to iterations present in the total, so a phase recorded on a
        # thread's warm-up or shutdown cannot inflate the share.
        vals = [v for k, v in s.items() if k in keys]
        tot = sum(vals)
        accounted += tot
        pct = 100.0 * tot / total_sum if total_sum else 0.0
        print(f'  {label:<26}{tot:>12.1f}{tot / n:>12.3f}{pct:>9.1f}%{len(vals):>10}')

    other = total_sum - accounted
    pct = 100.0 * other / total_sum if total_sum else 0.0
    print('  ' + '-' * 70)
    print(f'  {"other":<26}{other:>12.1f}{other / n:>12.3f}{pct:>9.1f}%')
    print(f'  {"TOTAL":<26}{total_sum:>12.1f}{total_sum / n:>12.3f}{100.0:>9.1f}%')

    if other < -0.001 * abs(total_sum):
        print()
        print('  WARNING: negative remainder -- phases overlap or double-count.')
    for note in notes:
        print(f'  note: {note}')
    if missing:
        print(f'  missing series: {", ".join(missing)}')


def write_loop_csv(lc_dir, out_path):
    """Join the Loop Closing series into one tidy row per iteration."""
    cols = ['loopClosing_time', 'placeRecognition_time', 'loopCorrection_time',
            'loopFusion_time', 'searchAndFuse_time', 'graphOptimization_time',
            'searchByProjection_time', 'globalBA_time',
            'loopDetected', 'loopClosed', 'loopRejected', 'mergeDetected',
            'numKFs', 'numMPs']
    data = {c: (load(os.path.join(lc_dir, c + '.txt')) or {}) for c in cols}
    keys = sorted(data['loopClosing_time'])
    if not keys:
        print(f'  no Loop Closing data to export from {lc_dir}')
        return
    with open(out_path, 'w') as f:
        f.write('kfId,' + ','.join(cols) + '\n')
        for k in keys:
            # Phases that did not run in an iteration are genuinely zero time, not
            # missing data -- except globalBA, which may simply not have finished.
            row = [str(k)]
            for c in cols:
                if c == 'globalBA_time' and k not in data[c]:
                    row.append('')
                else:
                    row.append(f'{data[c].get(k, 0):g}')
            f.write(','.join(row) + '\n')
    print(f'  wrote {len(keys)} rows to {out_path}')


def main():
    args = sys.argv[1:]
    out_csv = None
    if '--csv' in args:
        i = args.index('--csv')
        if i + 1 >= len(args):
            print(__doc__)
            return 1
        out_csv = args[i + 1]
        args = args[:i] + args[i + 2:]
    if len(args) != 1:
        print(__doc__)
        return 1
    stats = args[0]

    report(
        'FastTrack (Tracking thread) -- per frame',
        os.path.join(stats, 'Tracking', 'data'),
        'tracking_time',
        [
            ('orb extraction',      'orbExtraction_time'),
            ('stereo matching',     'stereoMatch_time'),
            ('pose prediction',     'trackWithMotionModel_time'),
            ('track local map',     'trackLocalMap_time'),
            ('keyframe creation',   'createKF_time'),
            ('relocalization',      'relocalization_time'),
        ],
        notes=(
            'total is wall time of System::TrackStereo, so "other" includes any '
            'blocking on Local Mapping.',
        ),
    )

    report(
        'TurboMap (Local Mapping thread) -- per iteration',
        os.path.join(stats, 'LocalMapping', 'data'),
        'localMapping_time',
        [
            ('process keyframe',    'processKF_time'),
            ('map point culling',   'MPCulling_time'),
            ('kp search + triang.', 'MPCreation_time'),
            ('map point fusion',    'searchInNeighbors_time'),
            ('local BA',            'LBA_time'),
            ('keyframe culling',    'KFCulling_time'),
        ],
        notes=(
            'local BA and keyframe culling are skipped when a new keyframe arrives '
            'mid-iteration, hence fewer samples than iterations.',
        ),
    )

    lc_dir = os.path.join(stats, 'LoopClosing', 'data')

    report(
        'FastLoop (Loop Closing thread) -- amortised over every iteration',
        lc_dir,
        'loopClosing_time',
        [
            ('region detection',    'placeRecognition_time'),
            ('loop correction',     'loopCorrection_time'),
        ],
        notes=(
            'loop correction fires only on corrected loops; most iterations are '
            'region detection that found nothing.',
        ),
    )

    # The view that matters: cost of the iterations that actually closed a loop,
    # selected by filtering loopClosing_time on the loopClosed flag.
    closed_keys = {k for k, v in (load(os.path.join(lc_dir, 'loopClosed.txt')) or {}).items() if v}
    report(
        'FastLoop -- CLOSED LOOPS ONLY, per closure',
        lc_dir,
        'loopClosing_time',
        [
            ('region detection',    'placeRecognition_time'),
            ('loop correction',     'loopCorrection_time'),
        ],
        notes=('rows where loopClosed == 1.',),
        only_keys=closed_keys,
    )

    # Loop correction is itself a parent of fusion and graph optimization, so it
    # gets its own split rather than being flattened into the table above.
    report(
        'FastLoop -- inside loop correction',
        lc_dir,
        'loopCorrection_time',
        [
            ('loop fusion',         'loopFusion_time'),
            ('graph optimization',  'graphOptimization_time'),
        ],
        notes=('loop fusion includes searchAndFuse_time as a sub-part.',),
    )

    # Per-closure detail, including the map size the correction had to work over.
    totals = load(os.path.join(lc_dir, 'loopClosing_time.txt')) or {}
    closed = {k: v for k, v in totals.items() if k in closed_keys}
    kfs = load(os.path.join(lc_dir, 'numKFs.txt')) or {}
    mps = load(os.path.join(lc_dir, 'numMPs.txt')) or {}
    gba = load(os.path.join(lc_dir, 'globalBA_time.txt')) or {}
    corr = load(os.path.join(lc_dir, 'loopCorrection_time.txt')) or {}

    print()
    print('=' * 78)
    print('  FastLoop -- each closure, with map size at the time')
    print('=' * 78)
    if not closed:
        print('  no closures recorded in this run')
    else:
        print(f'  {"kfId":>8}{"total ms":>12}{"correction":>13}'
              f'{"globalBA":>12}{"numKFs":>10}{"numMPs":>10}')
        print('  ' + '-' * 63)
        for k in sorted(closed):
            g = f'{gba[k]:.1f}' if k in gba else 'pending'
            print(f'  {k:>8}{closed[k]:>12.1f}{corr.get(k, 0.0):>13.1f}'
                  f'{g:>12}{int(kfs.get(k, 0)):>10}{int(mps.get(k, 0)):>10}')
        print()
        print('  globalBA runs on mpThreadGBA and is NOT part of total; "pending"')
        print('  means it had not finished when stats were written.')

    if out_csv:
        print()
        write_loop_csv(lc_dir, out_csv)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
