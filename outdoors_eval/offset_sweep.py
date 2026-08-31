#!/usr/bin/env python3
"""Sweep the alignment time offset to find the best fit for a stored trajectory.

evaluate3.py's --offset shifts the estimate's timestamps before association. A
real clock offset between ground truth and the estimate cannot be seen in
nearest-timestamp distance (that is bounded by sampling rate) -- it only shows
up as spatial misalignment. This sweeps offset, reporting RMSE and pair count.

Alignment matches evaluate3.py: Horn rotation with scale fixed at 1.0 (its
trans_error), which is the metric error a stereo-inertial system should be held
to. The scale-corrected value is reported alongside.
"""
import argparse, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, ncol=4):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line[0] in '#%':
            continue
        tok = line.replace(',', ' ').split()
        try:
            rows.append([float(x) for x in tok[:ncol]])
        except ValueError:
            continue
    a = np.array(rows)
    return a[:, 0], a[:, 1:4]


def associate(gt_t, est_t, offset, max_diff):
    """Nearest-GT match for each estimate stamp, within max_diff, GT used once."""
    shifted = est_t + offset
    idx = np.searchsorted(gt_t, shifted)
    lo = np.clip(idx - 1, 0, len(gt_t) - 1)
    hi = np.clip(idx, 0, len(gt_t) - 1)
    dlo, dhi = np.abs(shifted - gt_t[lo]), np.abs(shifted - gt_t[hi])
    best = np.where(dlo <= dhi, lo, hi)
    dist = np.minimum(dlo, dhi)
    keep = dist <= max_diff
    e_i = np.nonzero(keep)[0]
    g_i = best[keep]
    if len(e_i) == 0:
        return e_i, g_i
    # one GT sample per estimate: keep the closest where several collide
    order = np.lexsort((dist[keep], g_i))
    g_sorted, e_sorted = g_i[order], e_i[order]
    first = np.ones(len(g_sorted), bool)
    first[1:] = g_sorted[1:] != g_sorted[:-1]
    return e_sorted[first], g_sorted[first]


def align_rmse(model, data):
    """Horn alignment. Returns (rmse_scale_fixed, rmse_scale_corrected, scale)."""
    mc, dc = model - model.mean(1, keepdims=True), data - data.mean(1, keepdims=True)
    W = dc @ mc.T
    U, _, Vh = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U @ S @ Vh
    rotmodel = rot @ mc
    s = float((dc * rotmodel).sum() / (mc ** 2).sum())
    err_fixed = rot @ model + (data.mean(1, keepdims=True) - rot @ model.mean(1, keepdims=True)) - data
    err_scaled = s * rot @ model + (data.mean(1, keepdims=True) - s * rot @ model.mean(1, keepdims=True)) - data
    rmse = lambda e: float(np.sqrt((e ** 2).sum(0).mean()))
    return rmse(err_fixed), rmse(err_scaled), s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--est', required=True)
    ap.add_argument('--max-difference', type=float, default=2e9, help='ns (default 2 s)')
    ap.add_argument('--coarse', type=float, nargs=3, default=[-60e9, 60e9, 1e9],
                    metavar=('LO', 'HI', 'STEP'), help='offset sweep in ns')
    ap.add_argument('--refine-step', type=float, default=10e6, help='ns (default 10 ms)')
    ap.add_argument('--top', type=int, default=10)
    args = ap.parse_args()

    gt_t, gt_xyz = load(args.gt)
    est_t, est_xyz = load(args.est)
    print(f'gt {len(gt_t)} poses, est {len(est_t)} poses, '
          f'max_difference {args.max_difference/1e9:g} s')

    def score(off):
        e_i, g_i = associate(gt_t, est_t, off, args.max_difference)
        if len(e_i) < 10:
            return None
        f, sc, s = align_rmse(est_xyz[e_i].T, gt_xyz[g_i].T)
        return dict(offset=off, pairs=len(e_i), rmse=f, rmse_scaled=sc, scale=s)

    lo, hi, step = args.coarse
    coarse = [r for r in (score(o) for o in np.arange(lo, hi + step / 2, step)) if r]
    if not coarse:
        print('no offset produced usable matches')
        return 1
    best = min(coarse, key=lambda r: r['rmse'])
    print(f'\ncoarse best: offset {best["offset"]/1e9:+.3f} s  '
          f'rmse {best["rmse"]:.4f} m  pairs {best["pairs"]}')

    fine = [r for r in (score(o) for o in np.arange(best['offset'] - step,
                                                    best['offset'] + step + args.refine_step / 2,
                                                    args.refine_step)) if r]
    allr = sorted(coarse + fine, key=lambda r: r['rmse'])
    print(f'\ntop {args.top} offsets by scale-fixed RMSE:')
    print(f'{"offset(s)":>11}{"pairs":>8}{"rmse":>10}{"rmse_scaled":>13}{"scale":>9}')
    for r in allr[:args.top]:
        print(f'{r["offset"]/1e9:>11.3f}{r["pairs"]:>8}{r["rmse"]:>10.4f}'
              f'{r["rmse_scaled"]:>13.4f}{r["scale"]:>9.4f}')

    z = [r for r in allr if abs(r['offset']) < 1e-6]
    if z:
        print(f'\nzero offset (baseline): rmse {z[0]["rmse"]:.4f} m, pairs {z[0]["pairs"]}')
        b = allr[0]
        print(f'best is {(b["rmse"]-z[0]["rmse"])/z[0]["rmse"]*100:+.1f}% vs zero offset')
    return 0


if __name__ == '__main__':
    sys.exit(main())
