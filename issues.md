# GPU implementation — review notes

Read-only review of the CUDA path. Nothing here has been changed. Ordered by how
confident I am that it is a real defect, not by severity.

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

### 2. Stereo match: unchecked row overflow — `StereoMatchKernel.cu:623`

```cpp
memcpy(flat + i*MAX_FEATURES_IN_ROW_SLIDING_WINDOW, input[i].data(), sizeof(int) * input[i].size());
```

`input[i].size()` is never compared against `MAX_FEATURES_IN_ROW_SLIDING_WINDOW` (200).
A row with more than 200 candidates writes into the next row's slot, and the last row
writes past the end of the buffer.

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

`SearchLocalPointsKernel.cu:359` is the good pattern — it compares `numPoints` against
`MAX_NUM_MAPPOINTS` and reports. Nothing else does:

| kernel | buffer capacity | launch-time size | checked |
|---|---|---|---|
| `SearchByProjectionKernel` | 4100 map points | `vpPoints.size()` | no |
| `SearchAndFuseKernel` | 100 connected KFs | `connectedKFs.size()` | no |
| `FuseKernel` | `MAX_NEIGHBOR_KF_COUNT` = 100 | `neighKFs.size()` | no |
| `SearchLocalPointsKernel` | `MAX_NUM_MAPPOINTS` = 25000 | `numPoints` | **yes** |

Measured on outdoors5 (`1 1 1`): `SearchByProjection` high-water mark was **711 / 4100**,
zero overflows — so that one is latent, not firing. The other two are unmeasured.

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
