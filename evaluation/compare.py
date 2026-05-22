#!/usr/bin/env python3
import sys
import numpy as np
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_trajectory(txt_file, scale=1.0):
    """Load trajectory from TUM TXT file: timestamp tx ty tz qx qy qz qw"""
    data = np.loadtxt(txt_file)
    stamps = data[:,0]
    positions = data[:,1:4] * scale
    return stamps, positions

def plot_traj(ax, stamps, traj, style, color, label):
    """Plot 2D trajectory (x vs y)"""
    stamps_sorted_idx = np.argsort(stamps)
    x = traj[stamps_sorted_idx,0]
    y = traj[stamps_sorted_idx,1]
    ax.plot(x, y, style, color=color, label=label)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Plot estimated and original trajectories.')
    parser.add_argument('estimated_file', help='Estimated trajectory TXT file')
    parser.add_argument('original_file', help='Original trajectory TXT file')
    parser.add_argument('--scale', type=float, default=1.0, help='Optional scale factor for estimated trajectory')
    args = parser.parse_args()

    # Load trajectories
    est_stamps, est_positions = load_trajectory(args.estimated_file, scale=args.scale)
    orig_stamps, orig_positions = load_trajectory(args.original_file)

    # Plot trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_traj(ax, est_stamps, est_positions, '--', 'blue', 'FastLoop')
    plot_traj(ax, orig_stamps, orig_positions, '--', 'red', 'ORB-SLAM3')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend()
    plt.axis('equal')

    outname = Path(args.estimated_file).stem + "_comparison.png"
    plt.savefig(outname, format="png")
    print(f"Plot saved as {outname}")
