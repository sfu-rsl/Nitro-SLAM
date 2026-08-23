# Jetson (Orin) port — root causes and fixes

Everything here was found while getting Nitro-SLAM (`1 1 1`) to complete sequences on a
Jetson Orin Nano (JetPack R36.4.4, CUDA 12.6, sm_87). All four defects are **latent on
desktop and fatal on Tegra** — the code is not wrong in a way x86 + discrete GPU can
observe, which is why they only appeared here.

Written so the fixes can be replaced with different ones: each entry states the
mechanism and the constraint that has to be satisfied, then what was actually done and
what else would work.

## Device facts these all reduce to

Measured on this board, not assumed:

```
integrated                          = 1
unifiedAddressing                   = 1
concurrentManagedAccess             = 0     <- the big one
pageableMemoryAccess                = 0
canUseHostPointerForRegisteredMem   = 0
```

A discrete Pascal-or-later desktop GPU reports `concurrentManagedAccess = 1` and
`pageableMemoryAccess = 1`. Defects 1 and 2 are both direct consequences of those two
zeroes.

## A note on how all of these presented

Every one of these surfaced as a segfault somewhere unrelated to its cause, because of
`checkCudaError` (`src/Kernels/CudaUtils.cu:92`):

```cpp
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ...;
        if (TrackingKernelController::is_active) TrackingKernelController::shutdownKernels();
        ...
        exit(EXIT_FAILURE);
    }
}
```

Whichever thread hit an error tore down kernel memory while the other two threads were
still running and still using it. The teardown then segfaulted, and the segfault was what
appeared in the terminal.

**This is now fixed** (see "Error handling" below), but it is why every symptom recorded
here points somewhere other than its cause. Two consequences worth keeping in mind when
reading old logs:

- **The stderr line printed just before the crash is the real error.** `run_script.sh`
  redirects only stdout into `ostream.txt`, so that line lands in the *batch* log, not
  the per-run log. Defect 4 was misread as a benign `rc=1` for exactly this reason.
- A `cudaGetLastError()` failure is reported against whatever kernel launched *next*,
  not the call that actually failed. That is the whole of defect 1.



---

## 1. `cudaMemPrefetchAsync` is unsupported on Tegra

**Symptom.** `Failed to launch findBestStereoMatchKernel kernel: invalid device ordinal,
status code: 101`, then a segfault. Both MH01 and room3, immediately after the first
keyframe. The stereo match kernel is not involved.

**Mechanism.** `cudaMemPrefetchAsync` requires `concurrentManagedAccess`. On Tegra it
fails with `cudaErrorInvalidDevice` (101):

```
cudaMallocManaged            -> 0   (no error)
cudaMemPrefetchAsync(device) -> 101 (invalid device ordinal)
cudaGetLastError()           -> 101 (invalid device ordinal)   <- latched
```

The call site ignored its return value, so 101 sat in the thread's last-error slot. The
tracking thread creates a KeyFrame (`CudaKeyFrameAllocator::create`), then on the next
frame `StereoMatchKernel::launch` calls `checkCudaError(cudaGetLastError(), ...)` and
picks up the stale error. The kernel launched fine.

**Constraint.** Do not call `cudaMemPrefetchAsync` on a device with
`concurrentManagedAccess == 0`; and never leave a CUDA error unconsumed, because the
blame lands on an unrelated kernel.

**What was done.** A `DEVICE_JETSON` compile-time guard, auto-detected at configure time
with a manual override:

- `CMakeLists.txt:9-33` — `if(DEFINED DEVICE_JETSON)` uses the override; otherwise probes
  `/etc/nv_tegra_release`, then `/proc/device-tree/model`, defaulting `OFF`. Two markers
  because they fail independently (no `/etc/nv_tegra_release` in some containers, no
  `/proc/device-tree` in others). `-DDEVICE_JETSON=ON` also covers cross-compiling.
- `src/Kernels/CudaKeyFrameAllocator.cu:63` and
  `include/Kernels/UnifiedChunkAllocator.h:28` — `#ifndef DEVICE_JETSON` around the hint.

**Alternatives.** A runtime probe of `cudaDevAttrConcurrentManagedAccess` (cached) works
too and needs no build-system support — that was the first implementation, replaced with
the compile-time guard on request. Prefetch is only a placement hint, so dropping it
costs nothing on a device with one physical memory pool.

## 2. Host access to managed memory while a kernel is running

This is the one that caused the LIBA crashes, and the most likely to need rethinking.

**Symptom.** Segfaults inside `graphite` during `LocalInertialBA2`, at a **different
place every run** — `setup_hessian_computation`, `get_pose`, `add_mono_factor`. No
device-side errors under `compute-sanitizer`, which instead prints:

> The application may have hit an error when dereferencing Unified Memory from the host.

**Mechanism.** With `concurrentManagedAccess == 0` the CPU may not touch
`cudaMallocManaged` memory while *any* kernel is running on the device — the access
segfaults immediately, with no CUDA error raised. Reduced to:

```cpp
cudaMallocManaged(&managed, N);
managed[0] = 1;                 // fine, no kernel running
spin<<<1,1>>>(flag, 4000000000L);
managed[1] = 7;                 // SIGSEGV
```

Nitro-SLAM runs Tracking, LocalMapping and LoopClosing concurrently, all launching
kernels, and the host dereferences managed memory throughout graph building. So the
fault lands on whichever managed access happens to execute while another thread has a
kernel in flight — hence the moving crash site.

Affected allocations (all of them, at the time):

| what | where |
|---|---|
| `graphite::managed_vector` — factor observations, precision matrices, constraint data, chi2, loss; vertex block ids and active state | `Thirdparty/graphite/include/graphite/vector.hpp` (`uninitialized_allocator`) |
| CudaKeyFrame slot pool — host fills each slot via `CudaKeyFrame::setMemory` | `include/Kernels/UnifiedChunkAllocator.h:19` |
| cached `Camera` objects, written host-side as `*cam = Camera(params)` | `FIBA.cu` (2), `LIBA.cu` (1), `LocalBA.cu` (4) |

**Constraint.** Either (a) no host access to managed memory can overlap any kernel on
any thread, or (b) the memory must not be managed memory. (a) is not reachable here
without serialising the three threads, which defeats the architecture.

**What was done.** Pinned mapped memory (`cudaHostAlloc(..., cudaHostAllocMapped)`),
which on Tegra is physically shared, reachable from the device through the *same*
pointer (`unifiedAddressing == 1`), and carries no concurrency restriction. Verified:

```
kernel wrote through host pointer: p[2]=42 (expect 42)
host access during kernel: OK (p[3]=7)
```

Routed through one switch, `allocateSharedMemory` / `freeSharedMemory`
(`include/Kernels/CudaUtils.h:59`, `src/Kernels/CudaUtils.cu:108`) — `cudaHostAlloc` under
`DEVICE_JETSON`, plain `cudaMallocManaged` otherwise. graphite cannot see Nitro-SLAM
headers, so it carries its own guarded `allocate`/`deallocate` override in
`vector.hpp:26-46`.

**Side effect: it is faster.** Mean LIBA time went from 231–357 ms to 70–133 ms, because
managed memory was taking a migration fault on every host touch. Do not assume a
replacement approach is free — this one was strictly better on both axes.

**Alternatives, if reimplementing.**
- `cudaMallocManaged(..., cudaMemAttachHost)` plus `cudaStreamAttachMemAsync` per stream.
  Correct in principle, but every buffer needs attaching to the stream that uses it, and
  getting one wrong reintroduces exactly this bug in a rarer form.
- Explicit device buffers with staged copies. Most portable and most invasive; graphite's
  `managed_vector` exists precisely to avoid that.
- Sprinkling `cudaDeviceSynchronize()` before host access. Do not — it has to hold for
  *every other thread's* kernels too, so it means global serialisation.

Note `managed_vector` is the only managed allocation in graphite; there are no
`thrust::universal_vector` instantiations, and the two `prefetch_vector_*` helpers in
`utils.hpp` are never called.

## 3. GPU rotations vs Sophus's float epsilon

**Symptom.** Intermittent abort, roughly 1 run in 5 before it was fixed:

```
Sophus ensure failed in SO3<float>(const Transformation&) ... so3.hpp:470
R is not orthogonal:
            1 -8.51828e-07   7.0997e-07
-8.51828e-07      1.00001  1.32253e-07
  7.1915e-07  1.20964e-07      1.00001
```

**Mechanism.** The GPU optimizers work in double and return rotations a few ulps off
orthogonal. `Sophus::SE3f Tcw(pose.Rcw.cast<float>(), ...)` constructs an `SO3f`, whose
constructor checks `isOrthogonal` against **float** epsilon (~1e-5). `R*Rᵀ - I` here has
~2e-5 entries, so a numerically fine pose aborts the process.

Not strictly Tegra-only — it is FP-rounding-dependent, and aarch64 with `-march=native`
rounds differently from x86. It may simply never cross the threshold on your desktop.

**Constraint.** Any rotation handed from a GPU optimizer into a Sophus type must be
re-projected onto SO(3) first. The CPU paths already do this via `IMU::NormalizeRotation`
(`src/ImuTypes.cc:34`); the GPU write-backs just never did.

**What was done.** `Sophus::makeRotationMatrix` (`Thirdparty/Sophus/sophus/rotation_matrix.hpp:62`),
applied at all seven write-backs, not just the one that fired. It takes the argument at
the optimizer's own scalar type, so the cast down to float stays the only precision loss.

Use Sophus's function rather than a hand-rolled SVD. The first version of this fix
returned plain `U*Vᵀ`, which does not constrain the determinant and can therefore return
a reflection (det = -1); that would trip Sophus's *second* assertion,
`det(R) is not positive`, i.e. the same class of abort in rarer form.
`makeRotationMatrix` folds the sign in:

```cpp
Scalar d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
Diag(N - 1, N - 1) = d;              // so that det(U*D*V') = +1
return svd.matrixU() * Diag * svd.matrixV().transpose();
```

Worth noting `IMU::NormalizeRotation` (`src/ImuTypes.cc:34`), which the CPU paths use, is
the plain `U*Vᵀ` form and has the same gap. It is upstream ORB-SLAM3 code and was left
alone, but `Sophus::makeRotationMatrix` would be a strict improvement there too.

| file | sites | path |
|---|---|---|
| `src/Optimizer.cc:6119,6141` | 2 | `LocalInertialBA2` (the one that fired) |
| `src/Optimizer.cc:6444` | 1 | pose graph 4DoF → `g2o::Sim3` |
| `src/LocalBA.cu:924,954` | 2 | local BA |
| `src/FIBA.cu:681,687` | 2 | full inertial BA / GBA |

**Alternatives.** `Sophus::SO3<S>::fitToSO3(R)` wraps the same call if you would rather
build the `SO3` directly. Quaternion normalisation is cheaper than a 3×3 SVD if this ever
shows on a profile (it does not — LIBA is ~100 ms and this runs once per keyframe).
Relaxing Sophus's epsilon is the wrong lever: it hides genuine divergence too.

## 4. Fixed-capacity buffers filled with runtime sizes

This is item #7 in `issues.md`. It fired in `SearchAndFuseKernel` at about 1 run in 55;
two sibling kernels had the identical pattern and were converted at the same time.

**Symptom.** During a loop closure:

```
*Loop detected
Good loop found!
Local Mapping STOP
Failed to copy h_MapPoints to host: invalid argument, status code: 1
```

then a segfault in another thread's `checkCudaError` teardown.

**Mechanism.** Buffers are preallocated at 3000 map points / 100 keyframes, but
`launch()` shadows those names with the runtime sizes and never compares:

```cpp
int    connectedKFSize  = connectedKFs.size();    // capacity is 100
size_t mapPointVecSize  = vpMapPoints.size();     // capacity is 3000
```

The fill loops then write `h_MapPoints[numValidPoints]` past the pinned host array, and
`cudaMemcpy(d_MapPoints, h_MapPoints, numValidPoints * sizeof(CudaMapPoint), H2D)` asks
CUDA to write past the device allocation — which is `cudaErrorInvalidValue`. Only fires
on a loop closure large enough to exceed 3000, hence the low rate.

**Constraint.** Capacity must cover the runtime size before the fill loops run. Clamping
is not acceptable — dropping points silently changes the fuse result.

**What was done.** `SearchAndFuseKernel::ensureCapacity()`
(`src/Kernels/SearchAndFuseKernel.cu:40`), called at
`src/Kernels/SearchAndFuseKernel.cu:277` before the fill loops, doubling as needed from
the 3000/100 starting point. Also added the missing `checkCudaError` on those
allocations, and fixed the `d_KeyFrames` memcpy, which used `sizeof(CudaKeyFrame)` on an
array of *pointers* — it over-read the host array and only worked because both sides
were over-allocated with the same wrong size.

**Alternatives.** A one-shot allocation sized from the map's actual point count would
avoid the realloc entirely; growth was chosen because it needs no knowledge of the
caller. If you prefer a hard cap, it must be enforced *before* the fill loops and fail
loudly, not at the memcpy.

### The same pattern in two more kernels

Both converted to the same `ensureCapacity` shape.

**`SearchByProjectionKernel` — confirmed firing, not latent.** Map-point buffers were
fixed at 4100 while all four launch paths filled them with `vpPoints.size()`. It already
had instrumentation that *warned* on overflow without preventing it, and the high-water
recorded in `issues.md` was a comfortable 711/4100, which is why it was written off as
latent.

That measurement was from outdoors5. On the loop-heavy EuRoC V-sequences it **exceeds
4100**: a validation run needed 4116. With the old warn-only code that call would have
written past the pinned host buffer and asked cudaMemcpy to overrun the device
allocation — the same `cudaErrorInvalidValue` failure as `SearchAndFuseKernel`, just on a
different sequence mix. `ensureMapPointCapacity()` now grows instead.

The lesson is about the measurement, not the code: a high-water mark is only evidence for
the workload it was measured on. 711/4100 on one sequence said nothing about V102.

Its keyframe-side buffers (`h_KeyFrames`, `h_Ow`, `h_Tcw`) stay fixed at 3 deliberately:
the caller computes `covKFsSize = std::min<size_t>(3, vpCurrentCovKFs.size())`
(`src/LoopClosing.cc:958`, `:1152`), so 3 is a guarantee, not an estimate. Do not "fix"
those to grow — there is nothing to fix.

**`FuseKernel`** — device buffers were fixed at `MAX_NEIGHBOR_KF_COUNT` (100, doubled for
fisheye) while `launchV2` copied `neighKFs.size()` entries unchecked. ORB-SLAM3's
`SearchInNeighbors` adds second-order covisibles, so exceeding 100 is reachable in a
dense map. Also fixed: `shutdown()` never cleared `memory_is_initialized`, so a
shutdown/re-init cycle would have double-freed.

**Grow the dimensions independently.** This is the part worth getting right, and the
first attempt got it wrong in a way that only the growth logging exposed.

`d_currKFMapPoints` looks over-allocated at `MAX_NEIGHBOR_KF_COUNT * maxFeatures`, since
the kernel only indexes `[0, numPoints)`. It is not. `numPoints` is not bounded by the
feature count: `LocalMapping::SearchInNeighbors` also calls the single-keyframe path with
`vpFuseCandidates`, reserved as `vpTargetKFs.size() * vpMapPointMatches.size()`
(`src/LocalMapping.cc:1469`). Measured on EuRoC with 1200 features, that path asks for
**up to ~5500 points with `numKFs == 1`**. Shrinking the buffer to `maxFeatures` makes it
reallocate three times per sequence.

Worse, sizing the pair buffers (`d_bestDists`, `d_bestIdxs`) as
`kfCapacity * pointCapacity` multiplies two worst cases that never co-occur — the
large-`numPoints` case always arrives with one keyframe. Combined with a growth rule that
doubled *both* dimensions whenever *either* was exceeded, capacity ran
100/1220 -> 800/9760 and the pair buffers reached ~7.8M ints (~31 MB each) against the
original 122K (~0.5 MB): a ~64x memory regression on a board already peaking at 1.6 GB.

So each kernel now tracks three capacities and grows them separately:

| group | sized by |
|---|---|
| keyframe buffers | `numKFs` |
| map-point buffers | `numPoints` |
| pair buffers (`bestDists`/`bestIdxs`) | the call's actual `numPoints * numKFs` (x2 fisheye) |

Starting sizes equal the original fixed allocations, so this is hardening with no
steady-state memory change. The fisheye x2 stays explicit in the pair count, because the
doubling is a property of the kernel's indexing (each keyframe is processed left and
right), not of the neighbour list.

Each of the three logs when it actually grows, per group. Keep something like it. Without
it a clean test run cannot be distinguished from the case never occurring — and it is
what caught the sizing regression above, which 9 otherwise-clean runs had happily hidden.

---

## Error handling

Not a Tegra defect, but the reason every defect above presented as something else.

`checkCudaError` used to tear down all three kernel controllers and call `exit()`. The
failing thread does not own those GPU buffers, so the other two threads carried on into
freed memory; and `exit()` runs static destructors while they are still inside CUDA.
Either way the process died somewhere unrelated to the actual error.

Replaced with `fatalError()` (`src/Kernels/CudaUtils.cu`), which:

- **does not tear anything down.** There is nothing to release on a fatal path — the
  driver reclaims the context at process exit — and releasing it is precisely what
  caused the secondary crashes.
- **uses `std::_Exit`, not `exit`**, so no static destructors or atexit handlers run
  while other threads are live in CUDA.
- **holds a mutex through the exit**, so a second thread failing at the same moment
  blocks rather than interleaving with or truncating the first message. That is not
  hypothetical: the run that caught defect 4 had `SearchAndFuseKernel` failing on the
  loop-closing thread and `SearchLocalPointsKernel` failing on the tracking thread
  together.

Applied at all seven sites that had the `shutdownKernels(...); exit(EXIT_FAILURE);`
shape: `CudaUtils.cu` (checkCudaError), `CudaKeyFrameAllocator.cu`,
`SearchForTriangulationKernel.cu` (x2), `FuseKernel.cu` (x2), `SearchLocalPointsKernel.cu`.

Consequence for tooling: a fatal CUDA error now exits **rc=1 with a `[FATAL]` line**
rather than segfaulting with rc=139. `run_timing_batch.sh` only tests `rc -ne 0`, so it
is unaffected, but anything keying on 139 needs updating.

## Still open

1. **The graphite change lives in a submodule** (`Thirdparty/graphite`,
   `sfu-rsl/graphite`). It needs committing and pushing there or it is lost on re-clone.
2. **Thread-safety items 3-5 in `issues.md`** were not revisited. In particular #4, a
   keyframe slot returned to the free list while a kernel may still be reading it, is
   not addressed by locking and would present as a rare non-deterministic fault — the
   same shape as defect 4, so do not assume the crash budget is now zero without
   evidence.

## Verification as of writing

Full-pipeline (`1 1 1`) runs after defects 1–3 were fixed, before defect 4 was:

| run set | result |
|---|---|
| MH01 + room3, 4 iterations each | 8 / 8 complete |
| batch `jetson`, 5 iterations, 11 EuRoC sequences, both configs | ORB-SLAM3 55/55, Nitro-SLAM 54/55 |
| V103 × 30 under gdb | 30 / 30 clean |
| mixed V10x/V203 native, `LD_PRELOAD` segv handler | caught defect 4 on run 16 |

The single batch failure was defect 4. Batch timing: mean tracking speedup **1.57×** over
ORB-SLAM3 (48.5→31.5 ms on MH01, similar across the set), GPU peak ~1.6 GB vs 69 MB.

After the defect-4 fix, the same native hunt (60 runs, V101/V102/V103/V203):

| | before fix | after fix |
|---|---|---|
| crashes | 1 (run 16, V102) | **0 / 60** |
| runs that exceeded the old 3000-point capacity | — | **6 / 60** |

The second row is the part that matters. The growth path engaged six times, needing
3021, 3156, 3190, 3394, 3655 and 3693 map points — every one of them past the old fixed
capacity, i.e. the exact condition that used to fail the memcpy, now handled. A clean
sweep on its own would not have distinguished a working fix from the case not recurring.

Connected-keyframe counts stayed at 15-19 against a capacity of 100, so the map-point
buffer was the binding constraint, as diagnosed. The overflow rate is higher here (10%)
than in the batch (~2%) because this rotation is deliberately loop-closure-heavy.

### After the two sibling kernels and the error-handling change

Same hunt, 40 native runs, V101/V102/V103/V203:

| | result |
|---|---|
| runs completing a full trajectory | **40 / 40** |
| crashes | 0 |
| `[FATAL]` lines | 0 |
| capacity growths | 4, all map-point-only |

The four growths were `SearchAndFuse` 3000 -> 6000 (needed 3113, 3139, 3151) and
`SearchByProjection` 4100 -> 8200 (needed 4116). Two things to read from that:

- Only the map-point group ever grew; keyframe and pair capacities stayed at their
  starting sizes. That is the independent-growth rule working. An earlier version that
  doubled every dimension whenever one was exceeded would have quadrupled the pair
  buffers on each of these.
- The growth code is now exercised in two of the three kernels, so it is no longer
  untested code on a rarely-taken path.

**What 40 runs does not establish.** Against a fault at defect 4's rate (1 in 55), 40
clean runs is only ~52% detection power; 60 gives 67%, 110 gives 87%. These changes are
deterministic (buffer sizing and an error path), so their failure modes appear in the
first run or two rather than rarely — which is what this run set is good for. It is not
strong evidence about rare races. The full batch is the better regression test if that
matters, and it also regenerates the timing and memory numbers against the final binary,
which the ones above are not.

## Reproducing / debugging on this board

- `compute-sanitizer --tool memcheck --report-api-errors all` reports **zero** device
  errors for defects 2 and 4 — both are host-side. Its "dereferencing Unified Memory
  from the host" note is the tell for defect 2.
- Core dumps are piped to apport, which discards them for unpackaged binaries, and there
  is no passwordless sudo to change `kernel.core_pattern`. To get a backtrace from a
  native run, `LD_PRELOAD` a `SIGSEGV` handler calling `backtrace_symbols_fd` — it
  survives CUDA's own handler if installed from a library constructor. gdb also works but
  its ptrace overhead can mask the races.
- For a rare non-deterministic fault, count runs before claiming a fix: at a 1-in-55
  rate, 30 clean runs happen ~58% of the time by chance.
