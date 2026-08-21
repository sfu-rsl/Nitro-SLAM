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

Whichever thread hits an error tears down kernel memory while the other two threads are
still running and still using it. The teardown then segfaults, and the segfault is what
you see in the terminal. Two consequences worth keeping in mind while debugging:

- **The stderr line printed just before the crash is the real error.** `run_script.sh`
  redirects only stdout into `ostream.txt`, so that line lands in the *batch* log, not
  the per-run log. Defect 4 was misread as a benign `rc=1` for exactly this reason.
- A `cudaGetLastError()` failure is reported against whatever kernel launched *next*,
  not the call that actually failed. That is the whole of defect 1.

Worth fixing on its own merits — see "Still open" below.

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

## 4. `SearchAndFuseKernel` fixed capacity vs runtime size

This is item #7 in `issues.md`, now confirmed firing. Rate was about 1 in 55 full runs.

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

---

## Still open

Not fixed, listed in the order I would take them.

1. **Same unchecked-capacity pattern as defect 4**, in two more kernels.
   `SearchByProjectionKernel` (capacity 4100) only *warns* via `checkMapPointCapacity`
   and still overruns; measured high-water 711/4100, so latent. `FuseKernel`
   (`MAX_NEIGHBOR_KF_COUNT` 100) is unmeasured and never fired in 110 runs, but
   ORB-SLAM3's `SearchInNeighbors` adds second-order covisibles, so exceeding 100 is
   possible in a dense map. The `ensureCapacity` pattern from defect 4 ports directly.
2. **`checkCudaError` teardown** — see the note at the top. It turns every CUDA error
   into a segfault in an unrelated thread and destroys the diagnosis. Reporting the
   error and letting the process exit without cross-thread teardown would have saved
   most of the debugging in this document.
3. **The graphite change lives in a submodule** (`Thirdparty/graphite`,
   `sfu-rsl/graphite`). It needs committing and pushing there or it is lost on re-clone.

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
