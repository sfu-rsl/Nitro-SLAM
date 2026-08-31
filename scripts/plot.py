import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def read_metrics(file_path):
    if(os.stat(file_path).st_size == 0):
        return []
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            x, y = map(float, line.split(':'))
            data[int(x)] = y
    max_index = max(data.keys())
    array = np.zeros(max_index + 1)
    for x, y in data.items():
        array[x] = y
    return array

def plot(data_file_path, saving_file_path):
    base_name = os.path.splitext(os.path.basename(data_file_path))[0]
    data = read_metrics(data_file_path)
    if len(data) == 0:
        return

    plt.figure(figsize=(20, 6))

    mean_val = np.mean(data)
    max_val = np.max(data)
    max_label_text = f'Max: {max_val:.2f}'
    mean_label_text = f'Mean: {mean_val:.2f}'

    plt.plot(data, c = 'b', label=max_label_text)
    plt.axhline(y=mean_val, color='g', linestyle='--', label=mean_label_text)
    plt.xlabel("frameId")
    plt.ylabel("time (ms)")
    plt.title(base_name)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(saving_file_path, bbox_inches='tight')

    plt.close()

if __name__ == "__main__":
    if len(sys.argv)-1 != 1:
        print(f"Usage: {sys.argv[0]} <stats_dir>")
        sys.exit(1)

    stats_dir = sys.argv[1]

    data_dir = stats_dir + '/Tracking/data'
    PE_dir = stats_dir + '/Tracking/data/PoseEstimationKernel'
    SLP_dir = stats_dir + '/Tracking/data/SearchLocalPointsKernel'
    SM_dir = stats_dir + '/Tracking/data/StereoMatchKernel'

    plot_dir = stats_dir + '/Tracking/plots'
    PE_plot_dir = stats_dir + '/Tracking/plots/PoseEstimationKernel/'
    SLP_plot_dir = stats_dir + '/Tracking/plots/SearchLocalPointsKernel/'
    SM_plot_dir = stats_dir + '/Tracking/plots/StereoMatchKernel/'
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(PE_plot_dir, exist_ok=True)
    os.makedirs(SLP_plot_dir, exist_ok=True)
    os.makedirs(SM_plot_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            if (filename == 'num_frames_lost.txt'): continue
            if (filename == 'kernel_initialization_time.txt'): continue
            data_file_path = os.path.join(data_dir, filename)
            relative_path = os.path.relpath(data_file_path, data_dir)
            plot_file_path = os.path.join(plot_dir, filename[:-4] + '.png')
            plot(data_file_path, plot_file_path)

    for filename in os.listdir(PE_dir):
        if filename.endswith('.txt'):
            data_file_path = os.path.join(PE_dir, filename)
            relative_path = os.path.relpath(data_file_path, PE_dir)
            plot_file_path = os.path.join(PE_plot_dir, filename[:-4] + '.png')
            plot(data_file_path, plot_file_path)

    for filename in os.listdir(SLP_dir):
        if filename.endswith('.txt'):
            data_file_path = os.path.join(SLP_dir, filename)
            relative_path = os.path.relpath(data_file_path, SLP_dir)
            plot_file_path = os.path.join(SLP_plot_dir, filename[:-4] + '.png')
            plot(data_file_path, plot_file_path)

    for filename in os.listdir(SM_dir):
        if filename.endswith('.txt'):
            data_file_path = os.path.join(SM_dir, filename)
            relative_path = os.path.relpath(data_file_path, SM_dir)
            plot_file_path = os.path.join(SM_plot_dir, filename[:-4] + '.png')
            plot(data_file_path, plot_file_path)

    data_dir = stats_dir + '/LocalMapping/data'
    fuse_dir = stats_dir + '/LocalMapping/data/FuseKernel'
    SFT_dir = stats_dir + '/LocalMapping/data/SearchForTriangulationKernel'

    plot_dir = stats_dir + '/LocalMapping/plots'
    fuse_plot_dir = stats_dir + '/LocalMapping/plots/FuseKernel/'
    SFT_plot_dir = stats_dir + '/LocalMapping/plots/SearchForTriangulationKernel/'
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(fuse_plot_dir, exist_ok=True)
    os.makedirs(SFT_plot_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            if (filename == 'num_frames_lost.txt'): continue
            if (filename == 'kernel_initialization_time.txt'): continue
            data_file_path = os.path.join(data_dir, filename)
            relative_path = os.path.relpath(data_file_path, data_dir)
            plot_file_path = os.path.join(plot_dir, filename[:-4] + '.png')
            plot(data_file_path, plot_file_path)

    for filename in os.listdir(fuse_dir):
        if filename.endswith('.txt'):
            data_file_path = os.path.join(fuse_dir, filename)
            relative_path = os.path.relpath(data_file_path, fuse_dir)
            plot_file_path = os.path.join(fuse_plot_dir, filename[:-4] + '.png')
            plot(data_file_path, plot_file_path)

    for filename in os.listdir(SFT_dir):
        if filename.endswith('.txt'):
            data_file_path = os.path.join(SFT_dir, filename)
            relative_path = os.path.relpath(data_file_path, SFT_dir)
            plot_file_path = os.path.join(SFT_plot_dir, filename[:-4] + '.png')
            plot(data_file_path, plot_file_path)