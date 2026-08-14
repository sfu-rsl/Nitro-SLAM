# Loop-closure ablation — outdoors5

**Question:** the stock ORB-SLAM3 path closes a loop on outdoors5 in 3/3 runs. Which of
the FastTrack / TurboMap / FastLoop kernels preserve that, and which break it?

**Status:** complete. 22 configs x 3 runs = 66 runs, 2026-08-13 06:15 to 2026-08-14 00:00 UTC,
plus 3 runs of the full 14-kernel set after the ORB extraction fix, 2026-08-14 01:33 UTC.

## Answer

**GPU ORB extraction was the one kernel that broke loop closure. It has been fixed.**

```
./run_script.sh outdoors5 1 1 1 1 <version> 11111 1111 111111
```

**All 14 GPU kernels, 3/3 closed, 0 crashes** (`orbfix.0-2`, 2026-08-14). This supersedes
the 13-kernel recommendation below, which was written when `FT-orb` still failed.

| pass | detected | closed | rejected | ATE (m) | KFs |
|---|---|---|---|---|---|
| 0 | 1 | **1** | 0 | 6.92 | 2546 |
| 1 | 3 | **1** | 2 | 5.70 | 2379 |
| 2 | 6 | **1** | 5 | 7.54 | 2465 |

Against the same config before the fix — 0/3 detected, 0/3 closed, ATE 9.17-13.66 m,
3380-3703 KFs. See [what was wrong with GPU ORB extraction](#what-was-wrong-with-gpu-orb-extraction).

The original single-kernel table, and the 13-kernel set that was the best available
configuration before the fix, are kept below as the record of how it was found.

## The maximal working set

Two combinations were measured, 3 runs each.

| set | kernels | closed | crashed | ATE (m) | median track |
|---|---|---|---|---|---|
| **`01111 / 1111 / 111111`** | **13** | **3/3** | **0** | 6.29, 6.69, 6.17 | 8.9 ms |
| `01101 / 1111 / 111111` | 12 | 2/3 | 0 | 7.51, 6.07, 5.62 | 8.4 ms |
| baseline | 0 | 3/3 | 0 | 6.93, 7.03, 5.06 | ~9.5 ms |

**Superseded:** take the 14-kernel set (`11111 / 1111 / 111111`) — GPU ORB extraction has
since been fixed and closes 3/3. Of the two sets measured here, the 13-kernel one was the
better: it adds GPU pose estimation over the 12-kernel set, closes 3/3 instead of 2/3, and
costs ~0.5 ms of tracking time — still ~7% faster than baseline.

No interaction effects appeared in either. Detection was healthy in all six runs, so
nothing resembling `FT-orb`'s zero-detection failure leaks in through combination, and
neither set crashed once.

### The 13-kernel set has the best gravity alignment measured

Comparing the best roll/pitch margin per run — `min` over detections of
`max(|phi(0)|, |phi(1)|)` — against the 0.008 gate:

| set | best margin per run | as % of threshold |
|---|---|---|
| **13-kernel** | 0.00218, 0.00352, 0.00245 | **27-44%** |
| 12-kernel | 0.00419, 0.00953 ✗, 0.00692 | 52-119% |
| baseline | 0.00595, 0.00605, 0.00708 | 74-89% |

The 13-kernel set's worst margin is better than the baseline's best, and roughly half the
threshold. Its runs also produced far fewer candidates (1, 1, 3) with almost no
rejections — initially that looked like luck, but the margins show the opposite: **it
generates fewer detections and they are much better gravity-aligned.** That is the
plausible mechanism for adding pose estimation, which directly affects how well the map
tracks gravity.

It also means the gate-threshold change below is **no longer needed for this
configuration** — the 13-kernel set clears 0.008 comfortably. It remains worth doing for
robustness, not to make the recommended config work.

---

## What counts as a closure

`LoopClosing::Run` prints three markers, all in [src/LoopClosing.cc:259-301](src/LoopClosing.cc#L259-L301):

| marker | meaning |
|---|---|
| `*Loop detected` | place recognition confirmed a candidate (`mnLoopNumCoincidences >= 3`) |
| `BAD LOOP!!!` | the inertial gravity gate rejected it — `fabs(phi(0)) < 0.008 && fabs(phi(1)) < 0.008 && fabs(phi(2)) < 0.349` failed |
| `Good loop found!` | gate passed, `CorrectLoop()` runs |

A config **closes** if at least one `Good loop found!` appears. Detections without a
closure mean the gate rejected them, which is a different failure from not detecting at
all — the table reports both, because they point at different code.

Note the gate threshold is hardcoded at 0.008 in the current tree; the
`LoopClosing.gravityDriftTh` setting from earlier experiments is not present.

## Where outdoors5 actually closes its loop

A correction to the assumption this investigation has been carrying.

The mocap file does show one long gap. Recomputed from
`dataset-outdoors5_512_16/mav0/mocap0/data.csv`: 887.4 s total, a single 749.5 s gap from
t=59.2 s to t=808.7 s, leaving a final contiguous block of **78.7 s**. That much is
confirmed.

But that gap is **mocap coverage, not visual revisit.** It means the drone left the
motion-capture room, not that it never re-sees mapped territory until the end. Loop
closure does not need ground truth.

Measured from this sweep's logs, using `LocalInertialBA` calls before the first
`Good loop found!` as a progress proxy:

| run | LBA calls before 1st closure | total in a full run | fraction |
|---|---|---|---|
| baseline pass 0 | 3132 | 4780 | 66% |
| baseline pass 1 | 2900 | 4604 | 63% |
| baseline pass 2 | 2856 | 4786 | 60% |
| `TM-fuse` pass 0 | 2674 | 4383 | 61% |
| `TM-fuse` pass 1 | 2617 | 4356 | 60% |
| `TM-fuse` pass 2 | 2578 | (crashed early) | — |

**Every run closes its first loop around 60-66% of the way through, not in the final
9%.** The counts cluster tightly at 2578-3132 across six runs from two different configs.
Corroborating this independently: `TM-fuse` pass 2 had already closed when it segfaulted
at 704 s wall — roughly 100 s of sequence time before the mocap block even resumes.

The proxy is keyframe-based rather than wall-clock, so the percentages are approximate.
The conclusion does not depend on the precision: a closure two-thirds of the way through
the keyframe stream is not one occurring in the last 79 seconds.

**Why this matters.** The "only 79 seconds to get it right" framing implies that anything
trimming work at shutdown lands on the sequence's one opportunity. That reasoning drove
earlier work on shutdown-queue truncation. On this evidence the real revisit window is
much earlier and much wider, which is consistent with what the sweep found: the one
remaining failure is in tracking, not in anything shutdown-related.

## Flag decoder

Bit positions are read left-to-right from the kernel-status string in
[Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc:81-97](Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc#L81-L97).

### FastTrack — `kernel_status_FT`, 5 bits

| bit | name | effect when 1 |
|---|---|---|
| 0 | `orbExtraction` | GPU ORB extraction |
| 1 | `stereoMatch` | GPU stereo matching (`StereoMatchKernel`) |
| 2 | `searchLocalPoints` | GPU `SearchLocalPointsKernel` |
| 3 | `poseEstimation` | GPU `PoseEstimationKernel` |
| 4 | `poseOptimization` | **not a GPU kernel** — 1 runs pose optimization, 0 *skips it entirely* ([src/Tracking.cc:3086](src/Tracking.cc#L3086)) |

Because bit 4 is on-by-default behaviour rather than a kernel, the neutral FastTrack
config is `00001`, not `00000`. `00000` is included separately as its own ablation
(pose optimization disabled).

### TurboMap — `kernel_status_TM`, 4-6 bits

| bit | name | effect when 1 |
|---|---|---|
| 0 | `searchForTriangulation` | GPU triangulation search |
| 1 | `fuse` | GPU `FuseKernel` |
| 2 | `keyframeCulling` | `KeyFrameCullingOptimized()` instead of stock `KeyFrameCulling()` ([src/LocalMapping.cc:298](src/LocalMapping.cc#L298)) |
| 3 | `LBA` | `OptimizerGPU::LocalInertialBA2` instead of `Optimizer::LocalInertialBA` ([src/LocalMapping.cc:239](src/LocalMapping.cc#L239)) |
| 4 | `newTriangulation` | optional — `TriangulationMatchKernel` instead of `SearchForTriangulationKernel` |
| 5 | `gpu2` | optional — `CreateNewMapPointsGPU2`, geometry on GPU too |

Bits 4 and 5 only matter when bit 0 is set. They are excluded from the single-kernel
sweep (prior measurement: all three GPU triangulation variants behave identically) and
are covered in the follow-up section instead.

### FastLoop — `kernel_status_FL`, 6 bits

| bit | name | effect when 1 |
|---|---|---|
| 0 | `mergedSBP` | GPU merged `SearchByProjection` |
| 1 | `merged3SBP` | GPU 3-way merged `SearchByProjection` |
| 2 | `searchAndFuse` | GPU `SearchAndFuseKernel` |
| 3 | `singleSBP` | GPU single `SearchByProjection` |
| 4 | `graphOpt` | `OptimizerGPU::OptimizeEssentialGraph4DoF` instead of the CPU one ([src/LoopClosing.cc:1637](src/LoopClosing.cc#L1637)) |
| 5 | `globalBA` | `OptimizerGPU::FullInertialBA`, **and** relaxes the guard that otherwise skips GBA when the map has >= 200 keyframes ([src/LoopClosing.cc:1674](src/LoopClosing.cc#L1674), [:2239](src/LoopClosing.cc#L2239)) |

Bit 5 is not a pure swap: on outdoors5 the map has ~2800-3400 keyframes, so turning it
on makes global BA *run* where the CPU path would have skipped it. Treat any result
there as a test of two changes at once.

## Method

[run_ablation.sh](run_ablation.sh) drives one kernel at a time with the other two
optimizations off, so every row is a single-variable change against the ORB-SLAM3
baseline. 19 configs, 3 runs each, on outdoors5.

Passes are **interleaved** — pass 0 covers all 19 configs, then pass 1, then pass 2 —
so the table is always at equal depth and the job can be stopped at a pass boundary
without leaving a lopsided comparison. Each run is ~16.5 min, so a pass is ~5.2 h and
the full sweep ~16 h.

```
./run_ablation.sh 3 0      # 3 passes starting from pass 0
python3 score_ablation.py  # summarise whatever has landed
```

Results land in `<root>/<system>/<kernel_dir>/outdoors5/abl.<pass>/ostream.txt`. The
single-kernel sweep is under `Results-ablation/`, the maximal run under `Results/`;
`score_ablation.py` scans both, or any roots given on the command line.
[score_ablation.py](score_ablation.py) walks those and reports, per config: crash count,
closures / runs, per-run detection counts, ATE RMSE, and final keyframe count.

The baseline is re-run as part of the sweep rather than reusing the existing `base.*`
runs, because those predate the current binary (built 05:27, baselines ran 04:47-05:19).

### Configs

| label | FT | TM | FL | kernel status |
|---|---|---|---|---|
| `baseline` | 0 | 0 | 0 | — |
| `FT-none` | 1 | 0 | 0 | `00001` |
| `TM-none` | 0 | 1 | 0 | `0000` |
| `FL-none` | 0 | 0 | 1 | `000000` |
| `FT-orb` | 1 | 0 | 0 | `10001` |
| `FT-stereo` | 1 | 0 | 0 | `01001` |
| `FT-searchlocal` | 1 | 0 | 0 | `00101` |
| `FT-poseest` | 1 | 0 | 0 | `00011` |
| `FT-noposeopt` | 1 | 0 | 0 | `00000` |
| `TM-tri` | 0 | 1 | 0 | `1000` |
| `TM-fuse` | 0 | 1 | 0 | `0100` |
| `TM-cull` | 0 | 1 | 0 | `0010` |
| `TM-lba` | 0 | 1 | 0 | `0001` |
| `FL-mergedsbp` | 0 | 0 | 1 | `100000` |
| `FL-merged3sbp` | 0 | 0 | 1 | `010000` |
| `FL-searchandfuse` | 0 | 0 | 1 | `001000` |
| `FL-singlesbp` | 0 | 0 | 1 | `000100` |
| `FL-pgo` | 0 | 0 | 1 | `000010` |
| `FL-gba` | 0 | 0 | 1 | `000001` |

The three `*-none` rows are controls: they activate an optimization's controller and its
CUDA allocations without enabling any kernel. If one of those regresses, the problem is
in setup/teardown rather than in any individual kernel.

## Results

### Summary — runs detected / closed / crashed

`python3 score_ablation.py --summary`. A run counts as *detected* if it printed at least
one `*Loop detected`, *closed* if it reached `Good loop found!`, *crashed* if it
segfaulted. A run can be both closed and crashed — the loop closed, then the process died
before writing its trajectory.

| config | runs | detected | closed | crashed |
|---|---|---|---|---|
| baseline | 3 | 3 | **3** | 0 |
| `FT-none` `[00001]` *(control)* | 3 | 3 | **3** | 0 |
| `FT-stereo` `[01001]` | 3 | 3 | **3** | 0 |
| `FT-searchlocal` `[00101]` | 3 | 3 | **3** | 0 |
| `FT-orb` `[10001]` | 3 | **0** | **0** | 0 |
| `FT-poseest` `[00011]` | 3 | 0 | **0** | **3** |
| `FT-noposeopt` `[00000]` *(not a kernel)* | 3 | 2 | 1 | 0 |
| `TM-none` `[0000]` *(control)* | 3 | 3 | **3** | 1 |
| `TM-tri` `[1000]` | 3 | 3 | 2 | 0 |
| `TM-fuse` `[0100]` | 3 | 3 | **3** | 1 |
| `TM-cull` `[0010]` | 3 | 3 | **3** | 0 |
| `TM-lba` `[0001]` | 3 | 2 | 2 | 1 |
| `FL-none` `[000000]` *(control)* | 3 | 2 | 2 | 1 |
| `FL-mergedsbp` `[100000]` | 3 | 3 | **3** | 0 |
| `FL-merged3sbp` `[010000]` | 3 | 3 | **3** | 0 |
| `FL-searchandfuse` `[001000]` | 3 | 3 | **3** | 0 |
| `FL-singlesbp` `[000100]` | 3 | 3 | **3** | 0 |
| `FL-pgo` `[000010]` | 3 | 3 | **3** | 0 |
| `FL-gba` `[000001]` | 3 | 3 | **3** | 0 |
| **TOTAL** | **57** | **48** | **46** | **7** |

**Detection is the bottleneck, not the gate.** 48 runs detected, 46 closed. Once
outdoors5 sees its loop it nearly always gets one candidate past the gravity gate, so the
failure that matters is not detecting at all. Only two runs in 57 detected something and
closed nothing.

That reframes the earlier expectation that the roll/pitch gate was the main obstacle.
Individual *detections* are still rejected constantly — a typical closing run detects 3-8
loops and passes exactly one — but at these detection rates one always gets through.

### Per-run detail

| config | closed | detections per run | ATE (m) | KFs |
|---|---|---|---|---|
| baseline | **3/3** | 8, 4, 4 | 6.93, 7.03, 5.06 | 2875, 3179, 3368 |
| `FT-none` | **3/3** | 3, 2, 11 | 5.46, 6.98, 7.39 | 3593, 2976, 3489 |
| `FT-stereo` | **3/3** | 2, 12, 1 | 7.44, 6.08, 5.51 | 2938, 2921, 2664 |
| `FT-searchlocal` | **3/3** | 6, 6, 1 | 6.09, 6.85, 5.25 | 2562, 2664, 2602 |
| `FT-orb` | **0/3** | 0, 0, 0 | 13.66, 10.94, 9.17 | 3703, 3380, 3587 |
| `FT-poseest` | **0/3** | — (3 crashes) | — | — |
| `FT-noposeopt` | 1/3 | 1, 1, 0 | 24.45, 17.18, 25.30 | 1520, 1490, 1489 |
| `TM-none` | **3/3** | 6, 6, 2 | 7.03, —, 8.29 | 2742, —, 3087 |
| `TM-tri` | 2/3 | 3, 5, 9 | 7.43, 5.94, 11.08 | 3120, 3622, 3437 |
| `TM-fuse` | **3/3** | 6, 6, 4 | 8.13, 8.08, — | 2619, 2785, — |
| `TM-cull` | **3/3** | 2, 3, 5 | 7.60, 10.59, 5.90 | 3187, 3372, 3527 |
| `TM-lba` | 2/3 | 0, 5, 2 | —, 5.73, 6.71 | —, 3756, 2786 |
| `FL-none` | 2/3 | 2, 0, 3 | 3.76, —, 6.39 | 3511, —, 2934 |
| `FL-mergedsbp` | **3/3** | 4, 1, 2 | 8.19, 6.56, 7.92 | 3481, 2870, 2946 |
| `FL-merged3sbp` | **3/3** | 1, 6, 1 | 7.51, 6.47, 4.77 | 2954, 3000, 3430 |
| `FL-searchandfuse` | **3/3** | 1, 1, 6 | 5.43, 7.55, 6.72 | 2767, 3055, 2894 |
| `FL-singlesbp` | **3/3** | 9, 3, 4 | 8.14, 7.83, 6.47 | 3432, 2987, 2889 |
| `FL-pgo` | **3/3** | 5, 7, 4 | 6.77, 6.43, 7.73 | 2913, 2800, 2958 |
| `FL-gba` | **3/3** | 8, 1, 3 | 6.16, 4.48, 5.53 | 2937, 3182, 3048 |

Missing ATE entries are crashed runs, which never wrote a trajectory.

Across every closing config, ATE sits in a **4.5-11 m** band with no ordering that tracks
the kernel under test. Only `FT-orb` (9.2-13.7 m) and `FT-noposeopt` (17-25 m) sit clearly
outside it.

---

## The two failures

Neither is a failure any more: `FT-poseest` was an artefact of testing it in isolation,
and `FT-orb` was three real bugs in the extractor, now fixed. Both sections keep the
original evidence.

### `FT-orb` — GPU ORB extraction never recognises the place

**0/3 closed, and 0/3 detected.** All three runs completed normally, so the background
crash rate has no bearing on this result.

| pass | detections | ATE (m) | KFs | baseline ATE that pass |
|---|---|---|---|---|
| 0 | 0 | 13.66 | 3703 | 6.93 |
| 1 | 0 | 10.94 | 3380 | 7.03 |
| 2 | 0 | 9.17 | 3587 | 5.06 |

The failure is upstream of the gravity gate: place recognition never produces a confirmed
candidate. **Diagnosed and fixed** — see below.

### What was wrong with GPU ORB extraction

Found with [Examples/Tests/test_orb_extraction.cc](Examples/Tests/test_orb_extraction.cc),
which runs the CPU and GPU extractors over the same frames and diffs keypoints,
descriptors, pyramid levels and DBoW2 vectors. It answers in seconds what a dataset run
answers in sixteen minutes:

```
./Examples/Tests/test_orb_extraction <image_dir> [num_images] [stride] [vocabulary.txt]
```

**The extractor was not deterministic.** The same image extracted twice produced 11%
different keypoints. Place recognition matches bags of words between frames, so an
extractor that cannot reproduce itself on an identical image cannot recognise a place it
has already mapped. Tracking tolerates it; loop detection cannot. That is exactly the
observed signature — normal completion, degraded ATE, zero detections.

Three causes, all in [fast.cu](src/fast.cu) and [resize.cu](src/resize.cu):

1. **All eight pyramid levels shared one FAST score plane.** `d_Rs` was indexed by pixel
   with no `level*cols*rows` offset, so level 3's scores overwrote level 0's and
   non-maximum suppression read whichever level's block arrived first.
2. **Suppression read neighbours in the kernel that wrote them.** `__syncthreads()` covers
   a block, not the grid, so scores at block boundaries were stale, half-written, or left
   over from the previous frame — the plane was never cleared.
3. **The pyramid was resampled from the wrong source, at the wrong sample points.** Every
   level was bilinearly sampled straight from level 0 rather than from the level above,
   at `x*scale` instead of cv::resize's `(x+0.5)*scale-0.5`. Levels 1-7 were different
   images from the CPU's.

Fixes: FAST split into a score kernel and a per-cell NMS kernel reading a completed plane,
with per-level planes and the cell grid and two-threshold fallback of
`ComputeKeyPointsOctTree`; keypoints sorted before `DistributeOctTree`, whose
first-max-wins tie-break made buffer order observable; successive per-level resize using
cv::resize's fixed-point tables; BORDER_REFLECT_101 in the descriptor, since the GPU
pyramid carries no border and a rotated patch reaches ~21 px while keypoints sit 19 px
from the edge. Separately, `mvImagePyramid`'s device-to-host copy was never synchronised
before `Frame::ComputeStereoMatches` read it.

| metric | before | after |
|---|---|---|
| GPU repeatability, same image twice | 88.9% | **100.0%** |
| CPU keypoints the GPU also finds | 43.1% | **99.9%** |
| descriptor mean Hamming | 9.72 bits | **0.28 bits** (77% bit-identical) |
| pyramid vs cv::resize | up to 67% of pixels differ | **bit-identical, every level** |
| DBoW2 score, same image twice on GPU | — | **1.0000** |
| DBoW2 score, neighbouring frames | — | GPU 0.0299 vs CPU 0.0295 |
| extraction time | — | GPU 0.85 ms vs CPU 2.47 ms |

The residual 0.28 bits is the Gaussian blur: the GPU accumulates in float where
cv::GaussianBlur uses a fixed-point separable filter for 8U. It shifts about a quarter of
descriptors by a bit or two, costs ~5% of the DBoW2 score against a CPU-extracted copy of
the same image, and does not affect GPU-to-GPU matching, which is what loop detection
actually does.

**Extraction speed was never the problem.** The TUM-VI example sleeps out the remainder of
each frame interval ([stereo_inertial_tum_vi.cc:355](Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc#L355)),
so frames enter at 20 Hz regardless of how fast extraction runs. In the failing `FT-orb`
runs tracking took 6.71 ms median against a 50 ms budget, never once exceeding it, and was
*faster* than the baseline's 9.45 ms median which closed 3/3. The inflated keyframe counts
were bad keypoints forcing new keyframes, not a flooded queue.

### `FT-poseest` — broken only in isolation

**The kernel is fine. Running it alone is what crashes.**

| config | FastTrack kernels on | closed | crashed | detections |
|---|---|---|---|---|
| `FT-poseest` `[00011]` | poseEstimation only | **0/3** | **3/3** | 0, 0, 0 |
| `FT-pose-in-pipeline` `[01111]` | + stereoMatch, searchLocalPoints | **3/3** | **0/3** | 2, 8, 14 |
| `maximal13` | + all TurboMap and FastLoop | **3/3** | **0/3** | 1, 1, 3 |

Six consecutive full-length runs with no crash, against three segfaults in three isolated
runs — two of those inside 15 s. ATE in the combined configurations (6.08-6.95 m) sits on
top of the baseline.

**The direction is counterintuitive, and it narrows the search.** The path that *fails* is
the one where `CudaFrame::setMemory` uploads keypoints and descriptors **from the host**.
The path that *works* is the one where those pointers are **aliased to the
StereoMatchKernel's GPU buffers** by `setMvKeys` / `setMDescriptors`, latched behind
`stereoMatchDataHasMovedForward`
([TrackingKernelController.cu:104-110](src/Kernels/TrackingKernelController.cu#L104-L110)).
So the suspect is something the host-upload path fails to populate correctly for this
kernel — not extra work or contention in the combined path.

Every shipping configuration enables stereo match, so this is a latent bug on a path
nothing uses, not a blocker. The analysis below describes the isolated runs and is kept
for whoever chases it.

**Isolated runs: 3 segfaults in 3 runs.**

| pass | wall time | `LocalInertialBA` calls before the fault | reached VIBA 1 |
|---|---|---|---|
| 0 | 323 s | 1631 | yes |
| 1 | 8 s | 6 | no |
| 2 | 14 s | 17 | yes |

The spread is the diagnosis: **two orders of magnitude between earliest and latest fault,
one run dying before IMU initialisation while another survived 1631 LBA calls.** That
rules out a deterministic fault at a particular frame, map size, or initialisation state,
and points at a race. Against the 7% background rate, three crashes in three runs is a
~0.03% coincidence.

No backtrace was captured: `core_pattern` pipes to apport, which is not running in this
container, so every core was discarded. Getting one is the obvious next step — point
`core_pattern` at a file, or run under gdb. Two of three runs faulted within 15 s, so
reproduction is cheap.

---

## The background crash rate

7 crashes in 57 runs = 12% overall. Three are `FT-poseest`, which is genuinely broken;
backing it out gives the real background rate: **4 crashes in 54 runs = 7%**.

| pass | config | duration | closed first? |
|---|---|---|---|
| 0 | `TM-lba` | 163 s | no |
| 0 | `FT-poseest` | 323 s | no |
| 1 | `TM-none` *(control)* | 935 s | **yes** |
| 1 | `FL-none` *(control)* | 327 s | no |
| 1 | `FT-poseest` | 8 s | no |
| 2 | `FT-poseest` | 14 s | no |
| 2 | `TM-fuse` | 704 s | **yes** |

Ruled out as causes: disk, memory, GPU thermals, and early map resets
(`Not enough motion for initializing` appears in successful and crashed runs alike).

Two methodological points this cost real time to learn:

**Judge crashes by repetition, not timing.** Crash timing tells you nothing about
attribution. `TM-lba` crashed at 163 s and looked like the most suspicious result in the
sweep; it then closed in both remaining passes. The control `FL-none` crashed at 327 s,
four seconds from `FT-poseest`'s pass-0 fault. 3/3 is conclusive; 1/3 is noise. An early
attempt in this document to split crashes into "mid-run implicates the kernel" versus
"shutdown is background" was wrong and has been removed.

**Calibrate against the controls, not against perfection.** The control `FL-none` scored
2/3 and `TM-none` crashed once. A config at 2/3 is therefore performing at control level,
which is why `TM-tri` and `TM-lba` are not scored as regressions.

At a 7% per-run crash rate a healthy config still misses a clean 3/3 about 20% of the
time. Three runs was just barely enough to separate 3/3 from 0/3; it would not have been
enough to resolve anything subtler.

Two crashes happened **after** a successful closure (`TM-none` pass 1, `TM-fuse` pass 2) —
the loop closed, `CorrectLoop()` ran through `PGO CPU!` and `Local Mapping RELEASE`, then
the process died before writing its trajectory. Those runs count as closures and are
marked `CRASH*` by the scorer.

---

## The gravity gate is worth 0.008 -> 0.025

Measured across **209 detections in 51 detecting runs** (every run in the sweep plus the
maximal set), where the margin is `max(|phi(0)|, |phi(1)|)` against the 0.008 threshold in
[src/LoopClosing.cc:270](src/LoopClosing.cc#L270).

| threshold | individual detections passing | **detecting runs that close** |
|---|---|---|
| **0.008** (current) | 50 / 209 (24%) | **48 / 51 (94%)** |
| 0.010 | — | 50 / 51 (98%) |
| 0.015 | 124 / 209 (59%) | 50 / 51 (98%) |
| 0.020 | 164 / 209 (78%) | 50 / 51 (98%) |
| 0.025 | — | **51 / 51 (100%)** |
| 0.030 | 194 / 209 (93%) | 51 / 51 (100%) |

**This corrects a long-standing assumption.** The gate does reject most individual
detections — 76% of them — which is where the earlier "37 of 38 rejected" figure comes
from. But that is the wrong denominator. Runs typically get several detections, so at the
**run** level the gate only decides the outcome 6% of the time. It is a real cost, not the
main obstacle.

The three runs it cost, and what each needed:

| run | detections | best margin | closes at |
|---|---|---|---|
| `TM-tri` | 9 | 0.00837 | 0.0085 |
| `maximal` | 4 | 0.00953 | 0.0096 |
| `FT-noposeopt` | 1 | 0.02209 | 0.0222 |

Two of the three are near-misses within 20% of the current threshold, and both are configs
otherwise scored as healthy. **Raising the threshold to 0.010 would take `TM-tri` and the
maximal set to 3/3**; 0.025 would close every detecting run in the sweep. The third,
`FT-noposeopt`, is the pose-optimization-disabled config whose map is badly drifted anyway.

The yaw term never binds: max observed `|phi(2)|` is 0.1185 against a 0.349 threshold.

This is a threshold on gravity-direction drift, so loosening it trades loop-closure
reliability against the risk of accepting a badly gravity-misaligned correction. 0.010 is
a small, well-evidenced step; 0.025 is a bigger claim that should be validated on ATE, not
just closure counts.

## Other observations

**All three controls behaved.** `FT-none`, `TM-none` and `FL-none` activate an
optimization's controller and its CUDA allocations with no kernel enabled, and all three
closed loops (3/3, 3/3, 2/3). Controller setup/teardown and the CUDA allocation path are
not implicated in any regression.

**`TM-tri`'s one non-closing run is a gate rejection, not a detection failure** — it
detected 9 loops and passed none. That is the opposite signature to `FT-orb` and the only
run in the sweep where the gravity gate clearly decided the outcome.

**`FL-mergedsbp` is clean at 3/3** despite the harness default `kernel_status_FL='001111'`
disabling merged projection search with the comment "disable merged projection search".
Either that was fixed or the problem only appears in combination with other kernels. The
default is worth revisiting.

**`FT-noposeopt` quantifies what pose optimization is worth:** ATE 17-25 m against a
baseline of 5-7 m, and a ~1500-keyframe map against ~3000. It still closed once. Not a
kernel result — it is the `poseOptimization` bit being off, which disables the step
entirely.

**`FL-gba` remains two changes at once** — it swaps in `OptimizerGPU::FullInertialBA` *and*
relaxes the `KeyFramesInMap()<200` guard so global BA runs at all on a ~3000-keyframe map.
Its 3/3 covers both changes jointly, not the GPU kernel alone.

---

## What to do next

1. **Ship `01111 / 1111 / 111111`** — 13 of 14 kernels, 3/3 closed, 0 crashes, ~7% faster
   tracking, ATE on baseline, and the best gate margins measured. This meets the original
   goal on its own; nothing else on this list is required to get there.
2. **Test `FT-orb` in a pipeline, 3 runs** (`11111 / 1111 / 111111`). `FT-poseest` proved
   the single-kernel design can produce false positives, and `FT-orb` is the last kernel
   condemned on isolated evidence. If it passes, the set goes to 14 of 14. Prior notes
   claim `FT=11111` once closed 3/3, which is weak corroboration.
3. **Fix the two latent bugs found while reading the pose path.** Neither is the cause of
   the isolation crash, but both are real:
   - Clamp the grid-cell copy in
     [CudaFrame.cu:150-157](src/Kernels/CudaWrappers/CudaFrame.cu#L150-L157) —
     `Frame::AssignFeaturesToGrid` puts no cap on cell occupancy, the slot is 20, and the
     unclamped size is then handed to the kernel as a read bound.
   - Bounds-check `bestIdx2` before
     `CurrentFrame.mvpMapPoints[bestIdx2] = pMP` at
     [ORBmatcher.cc:2222](src/ORBmatcher.cc#L2222). The CPU path is safe by construction;
     the GPU path trusts the kernel and turns a bad index into silent host heap corruption.
4. **Raise the gravity gate from 0.008 to 0.010** for robustness. One-line change at
   [src/LoopClosing.cc:270](src/LoopClosing.cc#L270), supported by 209 measured detections.
   No longer needed for the recommended config, which clears the current threshold with
   ~2x margin — this is insurance, and worth validating on ATE.
5. **Fix the 7% background segfault.** It cost a fifth of the runs here and will confound
   every future 3-run experiment. Two of the four background crashes landed after a
   successful closure, so the shutdown path is the place to start.
6. **Chase the isolated `FT-poseest` crash only if you care about that path.** Nothing
   ships without stereo match. If you do: redirect `core_pattern` away from apport, and
   look at what the host-upload branch of `CudaFrame::setMemory` leaves unpopulated.
7. **Do not spend more effort on the FastLoop kernels.** All six are clean at 3/3. The
   loop-closing GPU path is not what breaks outdoors5.

---

### Reference: pre-sweep baseline

Three ORB-SLAM3 runs on the previous binary, for context:

| run | detections | outcome | ATE RMSE | KFs |
|---|---|---|---|---|
| `base.0` | 2 | 1 bad, 1 **good** | 6.92 m | 2815 |
| `base.1` | 1 | 1 **good** | 9.11 m | 3354 |
| `base.2` | 7 | 6 bad, 1 **good** | 6.22 m | 3268 |

3/3 closed, matching the 3/3 the in-sweep baseline produced on the current binary.
