# GPU implementation — review notes

Review of the CUDA path, ordered by how confident I am that each is a real defect,
not by severity. Originally read-only; one entry (now **§A**) has since been confirmed
against a live failure and fixed. Everything else remains unchanged.

## Fixed

### A. `SearchLocalPointsKernel` map-point cap — confirmed root cause of the magistrale1 crash — FIXED

The fixed `MAX_NUM_MAPPOINTS` 25000 cap in `SearchLocalPointsKernel` is not latent: it is
exceeded in practice, and it was the cause of the long-standing intermittent magistrale1
abort. Captured directly:

```
[FATAL] SearchLocalPointsKernel::launch: 26196 map points exceeds MAX_NUM_MAPPOINTS 25000
```

The local map grows over a sequence and a loop closure enlarges it sharply in one frame.
Whether the post-closure local map crosses 25000 decides the run, which is why the outcome
was cleanly bimodal — the closure lands at the same point every time and the run either
survives it or dies there:

| outcome | durations (s) |
|---|---|
| crash | 747.9, 748.0, 748.4, 748.6, 748.8, 749.4 |
| success | 876.3, 876.4, 876.5, 877.0 |

Nitro-SLAM crashed 6 of 12 magistrale1 attempts; ORB-SLAM3 1 of 6 (the CPU path has no
such cap, so the baseline dies from something else — see §14 and §3).

**Before the guard existed, this was silent.** The `[FATAL]` check was added in `052f855`;
prior to that the overrun simply wrote past the pinned host and device allocations, which
is why every earlier crash died with no message at all. Every run in `Results-tumvi/` was
produced by that binary. Survivors peaked at 23666 (corridor2) and 23344 (corridor1) — a
6-7% margin — so magistrale1 need not be the only sequence at risk on a re-seeded batch.

**Fix applied:** `ensureCapacity()` / `freeBuffers()` on the `SearchAndFuseKernel` model.
Doubling growth from an initial 25000, so steady-state allocation is unchanged from the
existing runs and only genuinely larger local maps reallocate. This also fixed a leak:
the old `shutdown()` never freed the ten R-side `best*` buffers or `d_mDescriptor`.

Two caveats worth keeping in view. Raising the cap does not make a >25000 local map *fast*
— it removes the abort, not the cost. And CPU pose-graph optimisation (`kernel_status_FL`
bit 5 clear) appeared to dodge the crash 3/3, but that is incidental: it perturbs the
corrected map into a different size, it does not remove the cap.

## Confirmed defects

### 0. Gaussian blur kernel uses sigma where sigma^2 is required — `ORBextractor.cc:165`

```cpp
const double stdev = SIGMA;                                      // 2
const double constant = 1.0 / (2.0 * pi * stdev);                // should be 2*pi*stdev*stdev
K[...] = constant * (1 / exp((pow(h,2)+pow(w,2)) / (2 * stdev)));// should be /(2*stdev*stdev)
```

With `SIGMA 2` the exponent divisor is 4 instead of 8, so the effective sigma is
**1.414 rather than 2.0** — the GPU blurs noticeably less than the CPU path's
`GaussianBlur(Size(7,7), 2, 2, BORDER_REFLECT_101)` at `ORBextractor.cc:1500`. Center
weight is 0.0796 against a correct 0.0467. The kernel also sums to 0.977 rather than 1,
which is harmless for ORB (pixel-pair comparisons are invariant to uniform scaling).

Leading hypothesis for the outdoors5 loop-detection failure. GPU descriptors are
self-consistent, so being different from CPU would not by itself break matching — but
BoW quantises them against `ORBvoc.txt`, a fixed vocabulary trained on standard
sigma=2 ORB descriptors. Under-blurred descriptors have different bit statistics, land
in different and less stable words, and BoW similarity collapses. Consistent with the
observed pattern: tracking (direct Hamming matching) and `SearchByBoW` (coarse level-4
FeatureVector) are unaffected, only full BoW place recognition fails.

Test: correct both terms, normalise, re-run `1 1 1`, see whether detections return.

### 1. Stereo match: row-index sentinel is never written — `StereoMatchKernel.cu:377`

```cpp
int vRowIndicesFlat[nRows * MAX_FEATURES_IN_ROW_SLIDING_WINDOW] = {-1};
```

`= {-1}` sets **element 0** to `-1` and value-initialises the remaining
`nRows*200 - 1` elements to **0**. `flattenVRowIndices()` then fills only
`input[i].size()` entries per row, leaving the rest at 0.

But `findBestStereoMatchKernel` (line 151) treats `-1` as the end-of-list terminator:

```cpp
for (int iC = vRowIndicesCountInRow*(int)vL; iC < vRowIndicesCountInRow*((int)vL + 1); iC++) {
    const int iR = vRowIndices[iC];
    if (iR == -1) break;
```

Since the padding is 0 rather than -1, the loop never breaks. It runs the full 200
slots per row and treats every padding entry as **right-keypoint index 0**, scoring
the left keypoint against the same wrong keypoint repeatedly. Affects every row whose
candidate count is below 200, i.e. effectively all of them.

This is on FastTrack's stereo-match kernel (bit 1 of `kernel_status_FT`), active in
every `11111` run.

**FIXED**, together with §2, in `flattenVRowIndices()`: it now fills the whole buffer with
-1 before copying, so padding terminates the loop, and the caller's broken `= {-1}`
initialiser is gone.

Measured effect on the kernel: **none**. `kernel_exec_time` on room3 was 0.3435 ms before
and 0.3415 ms after. The wasted iterations were nearly free — every padding slot read the
same `keypointsR[0]`, so the loads were broadcast cache hits, the octave check rejected
them before any descriptor comparison, and all threads branched identically so there was
no divergence. This is a correctness fix only; an earlier guess that it was inflating the
reported stereo-match timings was wrong.

### 2. Stereo match: unchecked row overflow — `StereoMatchKernel.cu:623`

```cpp
memcpy(flat + i*MAX_FEATURES_IN_ROW_SLIDING_WINDOW, input[i].data(), sizeof(int) * input[i].size());
```

`input[i].size()` is never compared against `MAX_FEATURES_IN_ROW_SLIDING_WINDOW` (200).
A row with more than 200 candidates writes into the next row's slot, and the last row
writes past the end of the buffer.

**FIXED** alongside §1: the per-row copy is clamped to the window, and a row that would
overflow reports once on stderr rather than truncating silently. The warning did not fire
on room3, so 200 is not currently being reached there.

The stack VLA itself (§8) is untouched.

### 3. `CudaKeyFrameStorage` is shared across three threads with no locking

`std::mutex mtx` is declared (`CudaKeyFrameStorage.h:35`) and defined
(`CudaKeyFrameStorage.cu:18`) but **never locked** — there is no `lock_guard` or
`unique_lock` anywhere in the file. The static `mnId_to_kf` (`unordered_map`),
`allocator`, and `num_keyframes` are touched concurrently by:

- **Tracking** — `KeyFrame` ctor → `addCudaKeyFrame` (`KeyFrame.cc:102`), insert
- **LocalMapping** — `SetBadFlag` → `eraseCudaKeyFrame` (`KeyFrame.cc:705`); `FuseKernel`
  and `SearchForTriangulationKernel` → `getCudaKeyFrame`; `addFeatureVector`
- **LoopClosing** — `SearchByProjectionKernel` (×4) and `SearchAndFuseKernel` →
  `addCudaKeyFrame` and `getCudaKeyFrame`

A rehash during `emplace` frees the bucket array under a concurrent `find`.

Only reachable when `MappingKernelController::is_active || LoopClosingKernelController::is_active`,
so pure-CPU (`0 0 0`) runs never touch it.

### 4. Use-after-free across a kernel launch (follows from 3, but not fixed by locking)

`getCudaKeyFrame` returns a raw pointer. Once the caller releases any lock,
`eraseCudaKeyFrame` on another thread does:

```cpp
ptr->setAsEmpty();          // writes memory the other thread may be reading
allocator.deallocate(ptr);  // slot goes straight back to free_slots
```

The next `allocate()` hands the same slot out and placement-news over it. Callers pass
these pointers *into kernel launches* (`SearchForTriangulationKernel` collects neighbour
KF pointers), so the slot can be recycled while a kernel is still reading it. Locking
the accessors does not address this — it needs deferred reclamation or no reclamation.

### 5. `UnifiedChunkAllocator` is not thread-safe — `UnifiedChunkAllocator.h:36`

`watermark++` is non-atomic, `chunks.push_back` can reallocate under a reader, and two
threads in `free_slots.front()/pop()` can be handed the **same slot**.

## Probable defects

### 6. Gaussian blur border handling differs from the CPU path — `gaussian_blur.cu:49`

```cpp
const int index = min(max(image_index+(h*imageStep)+w, 0), new_cols*new_rows);
```

Clamps a **linear** index where the CPU path uses OpenCV `BORDER_REFLECT_101`. Two
consequences: clamping instead of reflection, and horizontal offsets past a row's end
wrap into the next row rather than reflecting. The upper bound should also be
`new_cols*new_rows - 1` — as written it permits a one-element out-of-bounds read.

Only affects pixels within 3 px of an edge for a 7×7 kernel, and `EDGE_THRESHOLD`
excludes most border keypoints, so impact is probably small — but it does mean GPU and
CPU descriptors are not bit-identical.

### 7. Unchecked buffer capacities at launch

Superseded in part: `SearchAndFuseKernel` and `FuseKernel` gained `ensureCapacity()` in
`7479e85`/`052f855`, and `SearchLocalPointsKernel` now grows too (§A). Current state:

| kernel | buffer capacity | launch-time size | behaviour |
|---|---|---|---|
| `SearchByProjectionKernel` | 4100 map points | `vpPoints.size()` | unchecked |
| `CudaKeyFrame::addFeatureVector` | `MAX_FEAT_VEC_SIZE` = 100 ints | `mFeatCount` | unchecked — see §15 |
| `SearchAndFuseKernel` | grows | `connectedKFs.size()` | `ensureCapacity()` |
| `FuseKernel` | grows | `neighKFs.size()` | `ensureCapacity()` |
| `SearchLocalPointsKernel` | grows from 25000 | `numPoints` | `ensureCapacity()` |

Measured on outdoors5 (`1 1 1`): `SearchByProjection` high-water mark was **711 / 4100**,
zero overflows — so that one is latent, not firing.

The original framing of this entry — that `SearchLocalPointsKernel`'s check was "the good
pattern" — was wrong in an instructive way. Detecting the overflow is necessary but not
sufficient: the check aborted the process, which is how a recoverable capacity problem
became a 50% run-loss rate on magistrale1. Growing beats both overrunning and aborting.

### 8. Stack VLAs sized by runtime data

- `FuseKernel.cu:509` — `CudaKeyFrame* neighKFsGPUAddress[neighKFSize];`
- `StereoMatchKernel.cu:377` — `int vRowIndicesFlat[nRows * 200];` (~400 KB at 512 rows)
- `StereoMatchKernel.cu:380` — `CudaKeyPoint gpuKeypointsL[N], gpuKeypointsR[Nr];`

VLAs are a GNU extension, not standard C++, and these are sized from map/frame data with
no upper bound. The 400 KB one fits an 8 MB stack but leaves little headroom if these
nest.

### 9. Return codes ignored on many CUDA calls

`SearchByProjectionKernel` (22), `SearchAndFuseKernel` (12), and `SearchLocalPointsKernel`
(13) call `cudaMalloc` / `cudaMallocHost` / `cudaMemcpy` without `checkCudaError`. An
allocation failure yields a null pointer that surfaces later as an unrelated crash.

### 10. `h_KeyFrames` / `d_KeyFrames` sized by the wrong type — `SearchByProjectionKernel.cu:14,21`

Declared `CudaKeyFrame**` but allocated `3 * sizeof(CudaKeyFrame)`. Over-allocates, so
harmless in effect, but the sizing is clearly not intentional.

## Shutdown / lifecycle

### 11. Kernel shutdown flag typo — `MappingKernelController.cu:70`, `LoopClosingKernelController.cu:66`

```cpp
localMappingFinished = _localMappingFinished ? true : localMappingFinished;
loopClosingFinished  = _localMappingFinished ? true : loopClosingFinished;   // <- should be _loopClosingFinished
```

LocalMapping always exits first and calls `shutdownKernels(true, false)`, which flips
**both** flags and tears down the loop-closing kernels and shared CUDA keyframe storage
while LoopClosing is still running.

### 12. Both worker threads drop queued keyframes at shutdown

`LocalMapping::Run` and `LoopClosing::Run` each `break` as soon as `CheckFinish()` is
true, regardless of queue contents. Bounded in practice — `NeedNewKeyFrame()`
(`Tracking.cc:3335`) caps the local-mapping queue at 3, and measured backlog at shutdown
was 0–1 — so this is a latent correctness issue rather than an active one.

### 13. GBA lifecycle: `mbRunningGBA` can latch true forever — `LoopClosing.cc`

`RunGlobalBundleAdjustment` has two early `return`s inside the `mMutexGBA` block
(superseded index; map became inertial mid-BA) that leave `mbRunningGBA == true`. The
three abort sites (`CorrectLoop`, `MergeLocal`, `MergeLocal2`) detach the thread without
clearing it either. Anything that waits on `isRunningGBA()` can therefore hang — which is
presumably why the wait in `System::Shutdown()` is commented out.

Related: `mnFullBAIdx` is declared `bool` but used as a generation counter
(`mnFullBAIdx++`). Incrementing a `bool` pins it at `true`, so the staleness check
`idx != mnFullBAIdx` stops working after the first abort.

### 14. `LocalMapping::Release()` destroys queued keyframes — `LocalMapping.cc:1448`

```cpp
for (auto lit = mlNewKeyFrames.begin(); lit != mlNewKeyFrames.end(); lit++) delete *lit;
```

Called at the end of `CorrectLoop()` and from the GBA map-update block. `CorrectLoop`
calls `EmptyQueue()` first so it processes what is queued, but tracking stays live and can
insert more before `Release()` lands. The **GBA path has no `EmptyQueue()` at all** — it
stops local mapping, walks the entire spanning tree, then calls `Release()`, deleting
everything that accumulated during the walk. Loss is bounded by the queue cap of 3 today;
it scales directly if that cap is raised.

There is a second hazard here beyond the dropped keyframes: `Release()` mutates and frees
from `mlNewKeyFrames` while holding `mMutexStop` and `mMutexFinish`, but **not**
`mMutexNewKFs` — the mutex guarding that list at its other three access points
(`LocalMapping.cc:434`, `:441`, `:448`). `InsertKeyFrame()` runs on the tracking thread, so
the two can mutate the same `std::list` concurrently.

Investigated as the prime suspect for the magistrale1 crash, since `Local Mapping RELEASE`
was the last line before every death. **It was not the cause** — that turned out to be §A.
Both were wrong guesses that fit the log tail; the crash was in tracking, downstream of the
closure, not in the loop-closing handoff itself. The race is still real and still unfixed,
it just is not what was killing those runs.

### 15. `mFeatVecStartIndexes` device buffer is sized 100, copy is sized by BoW node count — `CudaKeyFrame.cu:155`

```cpp
mFeatCount = featVec.size();                       // distinct BoW nodes for this keyframe
...
cudaMemcpy(mFeatVecStartIndexes, tmp_mFeatVecStartIndexes, mFeatCount*sizeof(int), ...);
```

`mFeatVecStartIndexes` is allocated at `MAX_FEAT_VEC_SIZE` = **100** ints
(`CudaKeyFrame.cu:45`), and `copyFeatVec()` applies no cap. `mFeatCount` is the number of
distinct level-4 BoW nodes for the keyframe, which for the ~1000-1500 features these
sequences produce routinely runs to several hundred. `cudaMemcpy` does not reliably fault
on a modest overrun within the same allocation arena, so this can corrupt whatever device
memory follows rather than failing loudly.

Circumstantial support that the cap is genuinely exceeded: `SearchForTriangulationKernel`
already carries truncation counters against this same constant — `g_sftTruncatedPairs`,
`g_sftDroppedNodes`, printed at shutdown against `" / cap " << MAX_FEAT_VEC_SIZE`. Reading
those off a real run is the cheapest way to confirm.

The companion buffer `mFeatVec` (`MAX_FEAT_PER_WORD*MAX_FEAT_VEC_SIZE` = 10000 uints
against a total-feature-count copy of ~1000-1500) has ample headroom and is not at risk.

**FIXED**: both buffers now carry capacities and grow in `addFeatureVector()`. The
enclosing `CudaKeyFrame` lives in managed memory and the GPU dereferences these pointers
out of the struct, so swapping them needs no re-upload; pool slots are recycled without
re-running the constructor, so a grown keyframe stays grown. `freeMemory()` zeroes the
capacities so a reused slot cannot mistake freed pointers for live buffers. The two host
scratch VLAs were replaced with `std::vector` sized to the actual node/feature counts,
which also drops a needless `mFeatCount * 100` over-allocation.

### 16. Grid stride: one runtime constant, one compile-time constant — `CudaKeyFrame.cu:126,136`

`flatMGrid` is strided by the runtime `CudaUtils::keypointsPerCell`; `flatMGridRight` by the
compile-time `KEYPOINTS_PER_CELL`. Both are 20 today so the two agree and nothing breaks —
this is purely latent, and bites the moment `keypointsPerCell` is tuned. Neither loop caps
`num_keypoints` per cell against the stride either, so a dense cell overruns into the next
one (same shape as §2).

## Regression check on the fixes

room3 was chosen because it is one of the few sequences with **full-length** ground truth
— EuRoC and the TUM-VI `room` sequences have continuous mocap, while the other TUM-VI
sequences are only covered at the start and end, inside the mocap room. ATE on the rest is
therefore a start/end measurement, not a whole-trajectory one.

| build | n | mean ATE | values |
|---|---|---|---|
| HEAD without §1/§2/§15 | 4 | 0.01119 | 0.00834, 0.00859, 0.01335, 0.01448 |
| HEAD with §1/§2/§15 | 4 | 0.00999 | 0.00797, 0.01006, 0.01085, 0.01108 |
| pre-patch (`Results-tumvi`) | 5 | 0.00830 | 0.00704 - 0.01033 |

Mann-Whitney on the first two rows gives U=6 at n=4,4 — not significant, so the fixes make
no detectable difference to accuracy, and certainly do not make it worse.

Separately worth noting: the shift from 0.00830 to 0.01119 is present in HEAD *without*
these fixes, so it comes from the GPU-path changes in `7479e85`/`052f855` (rotation
re-projection, resizable kernel buffers), not from anything here. n is small and the
Mann-Whitney against the pre-patch runs is p≈0.09, so it is suggestive rather than
established — but room3 has trustworthy full-length ground truth, which makes it worth
a look before the next full batch.

## Not defects (checked and clear)

- The CPU loop-closing path in `LoopClosing.cc` is line-for-line identical to
  `base_LC.cc` — `NewDetectCommonRegions`, `DetectAndReffineSim3FromLastKF`,
  `DetectCommonRegionsFromLastKF` and `FindMatchesByProjection` match exactly, and the
  CPU branch of `DetectCommonRegionsFromBoW` differs only by a brace from the added
  `if/else` nesting.
- `local_inertial_ba` and `pose_graph_optimizer` are single-threaded per instance
  (`LocalInertialBA2` from LocalMapping, `OptimizeEssentialGraph4DoF` from LoopClosing);
  FIBA uses neither.
- `SearchByProjectionKernel`'s own buffers have a single owner thread — all five GPU
  entry points resolve to loop-closing-only `ORBmatcher` overloads. Tracking's
  `SearchByProjection(Frame&, ...)` overloads route to a separate kernel.
