#!/usr/bin/env python3
"""Report the sim3 alignment scale for stored trajectories.

A stereo-inertial system recovers metric scale, so the alignment scale should
sit near 1.0 (roughly +/-5%). A scale well outside that means the estimate is
metrically wrong, and its ATE -- which the sim3 alignment silently rescales
away -- understates how wrong. Compare:

  rmse_se3   ATE with scale fixed at 1.0 -- this is what evaluate3.py reports
             as absolute_translational_error.rmse (align()'s trans_error)
  rmse_sim3  ATE after scale-corrected alignment (align()'s trans_errorGT)

A large gap between the two is the tell. Note the scale is fitted over the
matched pairs only: if ground truth covers a much smaller spatial extent than
the estimate (TUM-VI outdoors mocap only covers the start/end room), the fit is
ill-conditioned and the scale says little -- hence the extent columns.

  ./outdoors_eval/scale_check.py outdoors5 --run 0 --system Nitro-SLAM
  ./outdoors_eval/scale_check.py outdoors3 --all-runs
"""
import argparse, glob, os, sys
import numpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'evaluation'))
import associate3, evaluate3

GT = os.path.expanduser('~/SLAM/Datasets/tumvi/dataset-{seq}_512_16/mav0/mocap0/data.csv')


def traj_path(results, system, seq, run):
    pat = os.path.join(REPO, results, system, '**', 'desktop', seq, str(run),
                       'trajectory', f'f_dataset-{seq}_stereoi.txt')
    hits = glob.glob(pat, recursive=True)
    return hits[0] if hits else None


def check(seq, system, run, results, max_difference):
    traj = traj_path(results, system, seq, run)
    if not traj:
        return None
    first = associate3.read_file_list(GT.format(seq=seq), False)
    second = associate3.read_file_list(traj, False)
    matches = associate3.associate(first, second, 0.0, float(max_difference))
    if len(matches) < 2:
        return dict(seq=seq, system=system, run=run, pairs=len(matches), error='too few matches')
    first_xyz = numpy.matrix([[float(v) for v in first[a][0:3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix([[float(v) for v in second[b][0:3]] for a, b in matches]).transpose()
    rot, transGT, err_gt, trans, err, scale = evaluate3.align(second_xyz, first_xyz)
    rmse = lambda e: float(numpy.sqrt(numpy.dot(e, e) / len(e)))
    extent = lambda m: float(numpy.linalg.norm(m.max(1) - m.min(1)))
    return dict(seq=seq, system=system, run=run, pairs=len(matches),
                scale=scale, rmse_se3=rmse(err), rmse_sim3=rmse(err_gt),
                gt_extent=extent(first_xyz), est_extent=extent(second_xyz), error='')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sequences', nargs='+')
    ap.add_argument('--system', default='Nitro-SLAM', choices=['Nitro-SLAM', 'ORB-SLAM3', 'both'])
    ap.add_argument('--run', type=int)
    ap.add_argument('--all-runs', action='store_true')
    ap.add_argument('--results', default='Results-tumvi')
    ap.add_argument('--max-difference', type=float, default=20000000)
    args = ap.parse_args()

    runs = range(5) if args.all_runs else [args.run if args.run is not None else 0]
    systems = ['ORB-SLAM3', 'Nitro-SLAM'] if args.system == 'both' else [args.system]

    print(f'{"sequence":<12}{"system":<11}{"run":>4}{"pairs":>7}{"scale":>9}'
          f'{"rmse_se3":>10}{"rmse_sim3":>11}{"gt_ext":>9}{"est_ext":>9}   verdict')
    for seq in args.sequences:
        for system in systems:
            for r in runs:
                res = check(seq, system, r, args.results, args.max_difference)
                if res is None:
                    continue
                if res.get('error'):
                    print(f'{seq:<12}{system:<11}{r:>4}{res["pairs"]:>7}'
                          f'{"":>9}{"":>10}{"":>11}{"":>9}{"":>9}   {res["error"]}')
                    continue
                off = abs(res['scale'] - 1.0)
                verdict = 'ok' if off <= 0.05 else ('SCALE OFF %.0f%%' % (off * 100))
                if res['gt_extent'] > 0 and res['est_extent'] / res['gt_extent'] > 3:
                    verdict += ' (ill-conditioned: est extent %.0fx gt)' % (
                        res['est_extent'] / res['gt_extent'])
                print(f'{seq:<12}{system:<11}{r:>4}{res["pairs"]:>7}{res["scale"]:>9.4f}'
                      f'{res["rmse_se3"]:>10.4f}{res["rmse_sim3"]:>11.4f}'
                      f'{res["gt_extent"]:>9.1f}{res["est_extent"]:>9.1f}   {verdict}')


if __name__ == '__main__':
    main()
