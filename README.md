# Nitro-SLAM

A unified, high-performance Visual-Inertial SLAM framework that integrates GPU parallelization, system-level pipelining, and graph optimization techniques into [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). 

This repository consolidates the optimization capabilities of four accelerated SLAM projects from SFU RSL (Reliable Systems Lab) to maximize tracking throughput, minimize latency, and improve global consistency across resource-constrained and high-throughput environments.

---

## 🚀 Integrated Frameworks, Attributions & Papers

This project integrates the following academic publications from the **SFU Reliable Systems Lab (RSL)**:

| Module | Core Responsibility | Upstream Repository | Academic Paper Link |
| :--- | :--- | :--- | :--- |
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
* **FastLoop:** `111111`

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

##### FastLoop (6-Digit Configuration)
1. **1**xxxxx : Merged Sim3 projection search across covisible keyframes for loop-candidate verification on GPU.
2. x**1**xxxx : Multi-keyframe (up to 3) common-region consistency check across the candidate's covisible keyframes on GPU.
3. xx**1**xxx : Loop fusion window and duplicate map-point merging on GPU.
4. xxx**1**xx : Low-level Sim3 projection-search kernel acceleration on GPU (used by the merged search routines above).
5. xxxx**1**x : Pose Graph Optimization (PGO) / Essential Graph backend engine on GPU.
6. xxxxx**1** : Global Full Inertial Bundle Adjustment (GBA) solver execution on GPU (falls back to a PCG iterative solver instead of the direct cuDSS solver once the map exceeds 200 keyframes).

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
