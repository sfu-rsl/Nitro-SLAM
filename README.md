# ?-SLAM

A unified, high-performance Visual-Inertial SLAM framework that integrates state-of-the-art GPU parallelization, system-level pipelining, and graph optimization techniques into **ORB-SLAM3**. 

This repository consolidates the optimization capabilities of four pioneering accelerated SLAM projects to maximize tracking throughput, minimize latency, and improve global consistency across resource-constrained and high-throughput environments.

---

## 🚀 Integrated Frameworks, Attributions & Papers

This project is built upon the contributions of the following upstream repositories and academic publications from the **SFU Robotics and Systems Laboratory (RSL)**:

| Module | Core Responsibility | Upstream Repository | Academic Paper Link |
| :--- | :--- | :--- | :--- |
| **ORB-SLAM3** | Baseline Visual-Inertial SLAM Engine | [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | [IEEE T-RO 2021](https://ieeexplore.ieee.org/document/9440682) |
| **FastTrack** | Front-end tracking and feature extraction acceleration | [sfu-rsl/FastTrack](https://github.com/sfu-rsl/FastTrack) | [IROS 2025](https://ieeexplore.ieee.org/document/11247316) |
| **TurboMap** | Global mapping concurrency and database optimization | [sfu-rsl/TurboMap](https://github.com/sfu-rsl/TurboMap) | [arXiv](https://arxiv.org/abs/2511.02036) |
| **FastLoop** | High-speed loop closure and backend tracking | [sfu-rsl/FastLoop](https://github.com/sfu-rsl/FastLoop) | [arXiv](https://arxiv.org/abs/2603.17201) |
| **Graphite** | GPU-accelerated graph optimization backend layers | [sfu-rsl/graphite](https://github.com/sfu-rsl/graphite) | [ICRA 2026](https://arxiv.org/abs/2509.26581) |

---

## 🛠️ Execution & Experimentation Scripts

The repository includes an automated evaluation pipeline via execution scripts located in the root directory. These scripts manage optimization flags, handle fine-grained kernel execution configurations, and automatically resolve dataset-specific paths for both the **EuRoC** and **TUM-VI** benchmarks.

### 1. Batch Experiments (`run_experiments.sh`)
Use this script to run multi-iteration, multi-dataset benchmark suites across a specific permutation of enabled modules.

```bash
./run_experiments.sh <FastTrack[0|1]> <TurboMap[0|1]> <FastLoop[0|1]> <version> <num_iterations>
```

**Example:** To benchmark a combination of FastTrack and TurboMap for 3 iterations under version tag v1.0:
```bash
./run_experiments.sh 1 1 0 v1.0 3
```

> 💡 **Note:** By default, this script runs benchmarks against a targeted subset of datasets. You can add more sequences, as mentioned in the script, to run comprehensive evaluations across all EuRoC and TUM-VI sequences.

### 2. Single Sequence Evaluation (`run_script.sh`)
Use this script to run an isolated experiment on a specific dataset sequence with optional, fine-grained control over GPU kernel configurations.

```bash
./run_script.sh <dataset_name> <FastTrack[0|1]> <TurboMap[0|1]> <FastLoop[0|1]> <output[0=stdout|1=file]> <version> [kernel_status1] [kernel_status2] [kernel_status3]
```

#### Positional Arguments:
* `dataset_name`: Name of the targeted sequence (e.g., `MH01`, `room3`, `corridor1`).
* `FastTrack[0|1]`: Toggle tracking acceleration front-end.
* `TurboMap[0|1]`: Toggle local mapping acceleration.
* `FastLoop[0|1]`: Toggle loop closure backend acceleration.
* `output`: `0` pipes engine output directly to `stdout`; `1` redirects standard streams to a log file within the results directory.
* `version`: Custom string identifier or identifier sequence (e.g., `v1.0.0`).
* `[kernel_statusX]`: *(Optional)* Bitmask strings to selectively toggle internal algorithmic sub-kernels for debugging or ablation studies.

#### Default Kernel Bitmasks:
If not explicitly passed, the framework defaults to fully operational bitmasks for activated components:
* **FastTrack:** `11110`
* **TurboMap:** `1111`
* **FastLoop:** `11111`

Each digit position within a passed bitmask acts as a binary switch (`1` to enable, `0` to disable) for a specific algorithmic sub-kernel or routine:

##### FastTrack (5-Digit Configuration)
1. **1**xxxx : ORB feature extraction acceleration on GPU.
2. x**1**xxx : Stereo feature matching optimization on GPU.
3. xx**1**xx : Local map points search acceleration on GPU.
4. xxx**1**x : Camera pose estimation acceleration on GPU.
5. xxxx**1** : Tracking pose optimization on/off toggle.

##### TurboMap (4-Digit Configuration)
1. **1**xxx : Map-point triangulation search acceleration on GPU.
2. x**1**xx : Map-point fusion acceleration on GPU.
3. xx**1**x : Redundant keyframe culling optimization on CPU.
4. xxx**1** : Local Bundle Adjustment (LBA) solver execution on GPU.

##### FastLoop (5-Digit Configuration)
1. **1**xxxx : Sim3 projection search/validation acceleration on GPU.
2. x**1**xxx : Relative pose and loop transformation correction on GPU.
3. xx**1**xx : Loop fusion window and duplicate map-point merging on GPU.
4. xxx**1**x : Pose Graph Optimization (PGO) backend engine on GPU.
5. xxxx**1** : Global Full Bundle Adjustment (BA) optimization layer execution.

---
---

## 📂 Results & Statistics Directory Tree

The pipeline dynamically generates an organized directory hierarchy for data parsing and visualization utilities. The folder naming convention automatically mirrors your active compilation switches:

```
./Results/
└── <System_Name>/                  # e.g., FastTrack&TurboMap, ORB-SLAM3, etc.
    └── <Kernel_Bitmasks>/          # e.g., 11110-1111 (Only appends for active modules)
        └── <Dataset_Name>/         # e.g., room3/
            └── <Version_Tag>/      # e.g., v1.0.0/
                ├── ostream.txt     # Captured stdout logs (if output=1)
                └── [Trajectory/Accuracy Stat Outputs]
```

---
