// Single-kernel pose optimization for tracking.
//
// The Graphite graph pipeline in PoseOptimization.cu is built for problems with
// many block columns: every Levenberg-Marquardt iteration costs on the order of
// twenty kernel launches plus host synchronisations, and the linear system is
// factorised on the host.  Tracking's pose optimizations are the opposite shape
// -- six to thirty unknowns, a few hundred factors -- so per-iteration latency,
// not arithmetic, is the entire cost, and g2o on the CPU wins.
//
// Here the whole thing (four outlier-rejection rounds x up to ten LM iterations,
// including the dense factorisation and the chi2 classification) runs inside a
// single block of a single kernel launch, with the state in shared memory.  The
// residual and Jacobian code is the same hd_fn functors the Graphite path uses,
// so the two agree on the math; only the schedule differs.
#include "Optimizer.h"

#include "G2oTypes.h"
#include "GPUTypes.h"
#include "FusedLM.h"

#include <cuda_runtime.h>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

namespace ORB_SLAM3 {
namespace OptimizerGPU {

using fused::FusedLMOptions;
using fused::FusedLMScratch;
using fused::fused_levenberg_marquardt;

namespace {

// Scalar used inside the fused kernels.
//
// This kernel deliberately runs on a single SM so that the LM loop needs no
// launches, and consumer GPUs issue FP64 at 1/64 the FP32 rate (two FP64 units
// per SM on GA102, one per two SMs' worth of FP32 throughput on Orin).  One
// block of doubles therefore has less arithmetic throughput available to it
// than a single CPU core, which is exactly the wrong trade for this shape of
// problem.  Single precision is what makes the fused schedule pay off; the
// state is a local refinement of an already-good pose, and the Hessian columns
// are rescaled to unit diagonal before the solve, so the conditioning that
// would justify FP64 is not present.  Everything the rest of the system sees --
// the recovered pose, and the marginalisation prior handed to the next frame --
// is still computed in double on the host.
using FT = float;

constexpr int kBlockThreads = 128;
constexpr FT kNoRobust = FT(1e30); // Huber delta^2 large enough to disable it

// ---------------------------------------------------------------------------
// Device-side factor record.  One per visual observation; laid out so the whole
// batch is a single host-to-device copy.
// ---------------------------------------------------------------------------
struct VisualFactor {
    FT Xw[3];
    FT obs[3];      // u, v and (stereo only) the right-image u
    FT invSigma2;
    int cam_idx;
    int is_stereo;
    int is_close;   // mono only: the inertial variants use a laxer threshold
};

__device__ __forceinline__ FT huber_weight(FT raw, FT delta2) {
    return raw <= delta2 ? FT(1) : sqrtf(delta2 / raw);
}
__device__ __forceinline__ FT huber_loss(FT raw, FT delta2) {
    return raw <= delta2 ? raw : FT(2) * sqrtf(raw * delta2) - delta2;
}

// Cooperative word-wise copy of a trivially copyable object.
__device__ __forceinline__ void block_copy(void *dst, const void *src, int nbytes) {
    int *d = reinterpret_cast<int *>(dst);
    const int *s = reinterpret_cast<const int *>(src);
    const int n = nbytes / static_cast<int>(sizeof(int));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        d[i] = s[i];
    }
}

// Per-thread accumulator for the visual block: the 6x6 Hessian's upper
// triangle, its gradient and chi2.
//
// Every visual factor touches the same 42 entries of the shared normal
// equations, so accumulating them with shared-memory atomics puts all 128
// threads on the same handful of addresses -- every warp instruction replays
// 32 times, and that contention, not the projection math, was the dominant
// cost of the kernel. Threads accumulate privately instead and the warp
// reduces once at the end of the factor loop.
struct VisualAccum {
    FT H[21]; // upper triangle, row major
    FT b[6];
    FT chi2;

    __device__ __forceinline__ void clear() {
        for (int i = 0; i < 21; i++) H[i] = 0;
        for (int i = 0; i < 6; i++) b[i] = 0;
        chi2 = 0;
    }

    __device__ __forceinline__ void warp_reduce() {
        for (int off = 16; off > 0; off >>= 1) {
            for (int i = 0; i < 21; i++) H[i] += __shfl_down_sync(0xffffffffu, H[i], off);
            for (int i = 0; i < 6; i++) b[i] += __shfl_down_sync(0xffffffffu, b[i], off);
            chi2 += __shfl_down_sync(0xffffffffu, chi2, off);
        }
    }

    // One scatter per warp into the shared normal equations, `col0` being the
    // Hessian column where the observed pose block starts.
    template <int Dim>
    __device__ __forceinline__ void flush(FT *Hs, FT *bs, FT *chi2s, int col0) const {
        if ((threadIdx.x & 31) != 0) return;
        int t = 0;
        for (int a = 0; a < 6; a++) {
            for (int c = a; c < 6; c++, t++) {
                const FT v = H[t];
                atomicAdd(&Hs[(col0 + a) * Dim + col0 + c], v);
                if (c != a) atomicAdd(&Hs[(col0 + c) * Dim + col0 + a], v);
            }
        }
        for (int a = 0; a < 6; a++) atomicAdd(&bs[col0 + a], b[a]);
        atomicAdd(chi2s, chi2);
    }
};

// Accumulate one visual factor's contribution into a thread's accumulator.
template <typename Camera>
__device__ __forceinline__ void
accumulate_visual(const gpu::ImuCamPose<FT, Camera> &pose, const VisualFactor &f,
                  FT huber_mono2, FT huber_stereo2, VisualAccum &acc) {
    using MonoC   = gpu::MonoConstraintOnlyPose<FT, FT, graphite::HuberLoss<FT, 2>, Camera>;
    using StereoC = gpu::StereoConstraintOnlyPose<FT, FT, graphite::HuberLoss<FT, 3>, Camera>;

    gpu::PoseOnlyData<FT> d;
    d.Xw << f.Xw[0], f.Xw[1], f.Xw[2];
    d.cam_idx = f.cam_idx;

    const int E = f.is_stereo ? 3 : 2;
    FT e[3];
    FT J[3 * 6]; // column major, E x 6

    if (f.is_stereo) {
        gpu::Vec3<FT> obs;
        obs << f.obs[0], f.obs[1], f.obs[2];
        const gpu::Vec3<FT> proj = pose.ProjectStereo(d.Xw, d.cam_idx);
        e[0] = obs[0] - proj[0];
        e[1] = obs[1] - proj[1];
        e[2] = obs[2] - proj[2];
        StereoC::template jacobian<FT, 0>(pose, obs, d, J);
    } else {
        gpu::Vec2<FT> obs;
        obs << f.obs[0], f.obs[1];
        const gpu::Vec2<FT> proj = pose.Project(d.Xw, d.cam_idx);
        e[0] = obs[0] - proj[0];
        e[1] = obs[1] - proj[1];
        MonoC::template jacobian<FT, 0>(pose, obs, d, J);
    }

    // Omega = invSigma2 * I, so the robustified quantities are all scalars.
    FT r2 = 0;
    for (int i = 0; i < E; i++) r2 += e[i] * e[i];
    const FT raw = r2 * f.invSigma2;
    const FT delta2 = f.is_stereo ? huber_stereo2 : huber_mono2;
    const FT w = huber_weight(raw, delta2) * f.invSigma2;

    acc.chi2 += huber_loss(raw, delta2);

    int t = 0;
    for (int a = 0; a < 6; a++) {
        const FT *Ja = J + a * E;
        FT bv = 0;
        for (int i = 0; i < E; i++) bv -= Ja[i] * w * e[i];
        acc.b[a] += bv;
        for (int c = a; c < 6; c++, t++) {
            const FT *Jc = J + c * E;
            FT v = 0;
            for (int i = 0; i < E; i++) v += Ja[i] * w * Jc[i];
            acc.H[t] += v;
        }
    }
}

template <typename Camera>
__device__ __forceinline__ FT visual_chi2(const gpu::ImuCamPose<FT, Camera> &pose,
                                          const VisualFactor &f, FT huber_mono2,
                                          FT huber_stereo2, bool robust) {
    gpu::Vec3<FT> Xw;
    Xw << f.Xw[0], f.Xw[1], f.Xw[2];

    FT r2;
    if (f.is_stereo) {
        gpu::Vec3<FT> obs;
        obs << f.obs[0], f.obs[1], f.obs[2];
        r2 = (obs - pose.ProjectStereo(Xw, f.cam_idx)).squaredNorm();
    } else {
        gpu::Vec2<FT> obs;
        obs << f.obs[0], f.obs[1];
        r2 = (obs - pose.Project(Xw, f.cam_idx)).squaredNorm();
    }
    const FT raw = r2 * f.invSigma2;
    if (!robust) return raw;
    return huber_loss(raw, f.is_stereo ? huber_stereo2 : huber_mono2);
}

// Generic "dense J block" accumulation: H[col0..][col0..] += J^T Omega J and
// b[col0..] -= J^T Omega e, for a residual of dimension E whose Jacobian spans
// `Cols` consecutive Hessian columns starting at `col0`.  J is column major
// (E x Cols), Omega is E x E row major.
// `OJ` is caller-provided workspace of E*Cols floats.  It lives in shared
// memory rather than on the stack: these arrays run to a kilobyte apiece, and
// the device stack (1 KB per thread by default) is not where they fit.
template <int Dim, int E, int Cols>
__device__ __forceinline__ void accumulate_dense(const FT *J, const FT *Omega,
                                                 const FT *e, FT w, FT *H, FT *b,
                                                 int col0, FT *OJ) {
    // The trip counts here reach 24x24x9; leaving them to the compiler's
    // default heuristics produces a fully unrolled kernel that takes minutes to
    // compile and thrashes the instruction cache, so the outer loops are pinned.
    FT Oe[E];
#pragma unroll 1
    for (int i = 0; i < E; i++) {
        FT p = 0;
        for (int j = 0; j < E; j++) p += Omega[i * E + j] * e[j];
        Oe[i] = w * p;
    }
    // Omega*J once, rather than re-forming it inside the H_{a,c} double loop.
#pragma unroll 1
    for (int c = 0; c < Cols; c++) {
        const FT *Jc = J + c * E;
#pragma unroll 1
        for (int i = 0; i < E; i++) {
            FT p = 0;
            for (int j = 0; j < E; j++) p += Omega[i * E + j] * Jc[j];
            OJ[c * E + i] = w * p;
        }
    }
#pragma unroll 1
    for (int a = 0; a < Cols; a++) {
        const FT *Ja = J + a * E;
        FT bv = 0;
        for (int i = 0; i < E; i++) bv -= Ja[i] * Oe[i];
        atomicAdd(&b[col0 + a], bv);
#pragma unroll 1
        for (int c = 0; c < Cols; c++) {
            const FT *OJc = OJ + c * E;
            FT v = 0;
            for (int i = 0; i < E; i++) v += Ja[i] * OJc[i];
            atomicAdd(&H[(col0 + a) * Dim + col0 + c], v);
        }
    }
}

// Block-cooperative form of the above, for the factors whose Jacobian spans a
// wide slice of the state: the inertial factor is 9x24 and the IMU prior 15x15,
// so forming J^T Omega J on a single thread is some fourteen thousand serial
// FLOPs per LM iteration -- more than the whole visual pass. The residual and
// Jacobian still come from one thread (they are a chain of SO(3) maps), but the
// products are spread over the block. `J`, `e`, `OJ` and `Oe` are shared.
template <int Dim, int E, int Cols>
__device__ void accumulate_dense_block(const FT *J, const FT *Omega, const FT *e,
                                       FT w, FT *H, FT *b, int col0, FT *OJ,
                                       FT *Oe) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    for (int idx = tid; idx < E * Cols; idx += nt) {
        const int c = idx / E, i = idx % E;
        FT p = 0;
        for (int j = 0; j < E; j++) p += Omega[i * E + j] * J[c * E + j];
        OJ[idx] = w * p;
    }
    for (int i = tid; i < E; i += nt) {
        FT p = 0;
        for (int j = 0; j < E; j++) p += Omega[i * E + j] * e[j];
        Oe[i] = w * p;
    }
    __syncthreads();

    for (int idx = tid; idx < Cols * Cols; idx += nt) {
        const int a = idx / Cols, c = idx % Cols;
        FT v = 0;
        for (int i = 0; i < E; i++) v += J[a * E + i] * OJ[c * E + i];
        atomicAdd(&H[(col0 + a) * Dim + col0 + c], v);
    }
    for (int a = tid; a < Cols; a += nt) {
        FT bv = 0;
        for (int i = 0; i < E; i++) bv -= J[a * E + i] * Oe[i];
        atomicAdd(&b[col0 + a], bv);
    }
    __syncthreads();
}

template <int E>
__device__ __forceinline__ FT weighted_sq(const FT *Omega, const FT *e) {
    FT v = 0;
    for (int i = 0; i < E; i++) {
        FT p = 0;
        for (int j = 0; j < E; j++) p += Omega[i * E + j] * e[j];
        v += p * e[i];
    }
    return v;
}

// ---------------------------------------------------------------------------
// Problem 1: pose only (6 unknowns), the pre-IMU-initialisation path.
// ---------------------------------------------------------------------------
template <typename Camera> struct PoseOnlyProblem {
    using State = gpu::ImuCamPose<FT, Camera>;
    static constexpr int dim = 6;

    const VisualFactor *factors;
    const unsigned char *active;
    int nfac;
    FT huber_mono2;
    FT huber_stereo2;

    __device__ void linearize(const State &s, FT *H, FT *b, FT *chi2) const {
        VisualAccum acc;
        acc.clear();
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            if (!active[i]) continue;
            accumulate_visual<Camera>(s, factors[i], huber_mono2, huber_stereo2, acc);
        }
        acc.warp_reduce();
        acc.flush<dim>(H, b, chi2, 0);
    }

    __device__ void chi2(const State &s, FT *chi2) const {
        FT acc = 0;
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            if (!active[i]) continue;
            acc += visual_chi2(s, factors[i], huber_mono2, huber_stereo2, true);
        }
        if (acc != FT(0)) atomicAdd(chi2, acc);
    }

    __device__ void update(State &s, const FT *dx) const { s.Update(dx); }
};

template <typename Camera>
__global__ __launch_bounds__(kBlockThreads) void
pose_only_kernel(gpu::ImuCamPose<FT, Camera> *pose_io, const Camera *cams,
                 int num_cams, const VisualFactor *factors, int nfac,
                 unsigned char *active, unsigned char *outlier, int *out_nbad,
                 FusedLMOptions opt, FT chi2_mono, FT chi2_stereo, int rounds) {
    using State = gpu::ImuCamPose<FT, Camera>;

    __shared__ __align__(16) unsigned char s_init[sizeof(State)];
    __shared__ __align__(16) unsigned char s_cur[sizeof(State)];
    __shared__ __align__(16) unsigned char s_bak[sizeof(State)];
    __shared__ __align__(16) unsigned char s_cams[2 * sizeof(Camera)];
    __shared__ FusedLMScratch<6, FT> sc;
    __shared__ int s_nbad;

    State &initial = *reinterpret_cast<State *>(s_init);
    State &state = *reinterpret_cast<State *>(s_cur);
    State &backup = *reinterpret_cast<State *>(s_bak);
    Camera *scams = reinterpret_cast<Camera *>(s_cams);

    block_copy(s_init, pose_io, sizeof(State));
    block_copy(s_cams, cams, num_cams * sizeof(Camera));
    __syncthreads();
    if (threadIdx.x == 0) {
        initial.pCamera[0] = &scams[0];
        initial.pCamera[1] = num_cams > 1 ? &scams[1] : nullptr;
        s_nbad = 0;
    }
    __syncthreads();

    PoseOnlyProblem<Camera> problem{factors, active, nfac, FT(5.991), FT(7.815)};

    int nbad = 0;
    for (int round = 0; round < rounds; round++) {
        if (threadIdx.x == 0) state = initial;
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            active[i] = outlier[i] ? 0 : 1;
        }
        // g2o drops the Huber kernel after the third round ("if(it==2)
        // e->setRobustKernel(0)"); an effectively infinite delta is the same
        // thing without a second code path.
        if (round >= 3) {
            problem.huber_mono2 = kNoRobust;
            problem.huber_stereo2 = kNoRobust;
        }
        __syncthreads();

        fused_levenberg_marquardt<6, FT>(problem, state, backup, sc, opt);

        if (threadIdx.x == 0) s_nbad = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            // Raw (non-robustified) chi2, matching g2o's e->chi2().
            const FT c = visual_chi2(state, factors[i], FT(5.991), FT(7.815), false);
            const bool bad = c > (factors[i].is_stereo ? chi2_stereo : chi2_mono);
            outlier[i] = bad ? 1 : 0;
            if (bad) atomicAdd(&s_nbad, 1);
        }
        __syncthreads();
        nbad = s_nbad;

        if (nfac < 10) break;
    }

    if (threadIdx.x == 0) {
        state.pCamera[0] = nullptr;
        state.pCamera[1] = nullptr;
        *pose_io = state;
        *out_nbad = nbad;
    }
}

// ---------------------------------------------------------------------------
// Problem 2: frame pose + velocity + biases against a fixed keyframe
// (15 unknowns), the PoseInertialOptimizationLastKeyFrame path.
// ---------------------------------------------------------------------------
template <typename Camera> struct NavState {
    gpu::ImuCamPose<FT, Camera> pose;
    gpu::Vec3<FT> vel;
    gpu::Vec3<FT> gbias;
    gpu::Vec3<FT> abias;
};

template <typename Camera> struct InertialKFProblem {
    using State = NavState<Camera>;
    static constexpr int dim = 15;

    const VisualFactor *factors;
    const unsigned char *active;
    int nfac;
    FT huber_mono2;
    FT huber_stereo2;

    const NavState<Camera> *anchor; // keyframe side; fixed
    const gpu::InertialConstraintData<FT> *imu;
    const FT *Omega9; // 9x9 row major
    const FT *InfoG;  // 3x3 row major
    const FT *InfoA;  // 3x3 row major
    FT *scratch;      // shared-memory workspace, 2 * 9 * 9 floats

    using InerC = gpu::InertialConstraint<FT, FT, graphite::DefaultLoss<FT, 9>,
                                          gpu::PoseDescriptor<FT, FT, Camera>>;

    // Residual and, optionally, the 9x9 stacked Jacobian over columns 0..8
    // (frame pose then frame velocity, which is exactly the state layout).
    // Each per-vertex block is column major with 9 rows, and the blocks are
    // contiguous in the stacked layout, so the constraint writes its blocks
    // directly into `J45` instead of into temporaries that would then be copied.
    // `J` is the stacked 9x24 Jacobian; the keyframe side is fixed, so only its
    // last nine columns ([pose2 | vel2], starting at column 15) are used, and
    // they land contiguously at J + 15*9 -- exactly this problem's 9x9 block.
    __device__ void inertial_residual(const State &s, FT *e, FT *J) const {
        InerC::template error<FT>(anchor->pose, anchor->vel, anchor->gbias,
                                  anchor->abias, s.pose, s.vel, *imu, e);
        if (J) {
            InerC::template jacobian_all<FT>(anchor->pose, anchor->vel, anchor->gbias,
                                             anchor->abias, s.pose, s.vel, *imu, J);
        }
    }

    __device__ FT bias_residuals(const State &s, FT *eg, FT *ea) const {
        for (int i = 0; i < 3; i++) {
            eg[i] = s.gbias[i] - anchor->gbias[i];
            ea[i] = s.abias[i] - anchor->abias[i];
        }
        return weighted_sq<3>(InfoG, eg) + weighted_sq<3>(InfoA, ea);
    }

    __device__ void linearize(const State &s, FT *H, FT *b, FT *chi2) const {
        VisualAccum acc;
        acc.clear();
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            if (!active[i]) continue;
            accumulate_visual<Camera>(s.pose, factors[i], huber_mono2, huber_stereo2, acc);
        }
        acc.warp_reduce();
        acc.flush<dim>(H, b, chi2, 0);

        // The three non-visual factors are one-of-a-kind; give each its own
        // thread rather than serialising them behind thread 0.
        FT *J = scratch;             // 9 x 24 stacked Jacobian
        FT *OJ = scratch + 216;      // 9 x 9 Omega*J for the used columns
        FT *e_in = scratch + 297;    // 9
        FT *Oe = scratch + 306;      // 9
        if (threadIdx.x == 0) {
            inertial_residual(s, e_in, J);
            atomicAdd(chi2, weighted_sq<9>(Omega9, e_in));
        } else if (threadIdx.x == 32) {
            // Gyro random walk: e = bg_frame - bg_kf, J = I on columns 9..11.
            FT e[3];
            for (int i = 0; i < 3; i++) e[i] = s.gbias[i] - anchor->gbias[i];
            atomicAdd(chi2, weighted_sq<3>(InfoG, e));
            FT J[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, OJ[9];
            accumulate_dense<dim, 3, 3>(J, InfoG, e, FT(1), H, b, 9, OJ);
        } else if (threadIdx.x == 64) {
            FT e[3];
            for (int i = 0; i < 3; i++) e[i] = s.abias[i] - anchor->abias[i];
            atomicAdd(chi2, weighted_sq<3>(InfoA, e));
            FT Ji[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, OJi[9];
            accumulate_dense<dim, 3, 3>(Ji, InfoA, e, FT(1), H, b, 12, OJi);
        }
        __syncthreads();
        // Columns 15..23 of the stacked Jacobian are [pose2 | vel2] -- this
        // problem's whole state -- and are contiguous at J + 15*9.
        accumulate_dense_block<dim, 9, 9>(J + 15 * 9, Omega9, e_in, FT(1), H, b, 0,
                                          OJ, Oe);
    }

    __device__ void chi2(const State &s, FT *chi2) const {
        FT acc = 0;
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            if (!active[i]) continue;
            acc += visual_chi2(s.pose, factors[i], huber_mono2, huber_stereo2, true);
        }
        if (threadIdx.x == 0) {
            FT e[9];
            inertial_residual(s, e, nullptr);
            acc += weighted_sq<9>(Omega9, e);
        }
        if (threadIdx.x == 32) {
            FT eg[3], ea[3];
            acc += bias_residuals(s, eg, ea);
        }
        if (acc != FT(0)) atomicAdd(chi2, acc);
    }

    __device__ void update(State &s, const FT *dx) const {
        s.pose.Update(dx);
        for (int i = 0; i < 3; i++) {
            s.vel[i]   += dx[6 + i];
            s.gbias[i] += dx[9 + i];
            s.abias[i] += dx[12 + i];
        }
    }
};

template <typename Camera>
__global__ __launch_bounds__(kBlockThreads) void inertial_kf_kernel(
    NavState<Camera> *state_io, const NavState<Camera> *anchor_in,
    const Camera *cams, int num_cams, const VisualFactor *factors, int nfac,
    unsigned char *active, unsigned char *outlier, int *out_counts,
    const gpu::InertialConstraintData<FT> *imu, const FT *Omega9, const FT *InfoG,
    const FT *InfoA, FusedLMOptions opt, const FT *chi2_mono,
    const FT *chi2_stereo, int rounds) {
    using State = NavState<Camera>;

    __shared__ __align__(16) unsigned char s_cur[sizeof(State)];
    __shared__ __align__(16) unsigned char s_bak[sizeof(State)];
    __shared__ __align__(16) unsigned char s_anchor[sizeof(State)];
    __shared__ __align__(16) unsigned char s_cams[2 * sizeof(Camera)];
    __shared__ FusedLMScratch<15, FT> sc;
    __shared__ FT s_work[216 + 81 + 9 + 9]; // J, Omega*J, residual, Omega*e
    __shared__ int s_counts[3]; // nbad, inliers_mono, inliers_stereo

    State &state = *reinterpret_cast<State *>(s_cur);
    State &backup = *reinterpret_cast<State *>(s_bak);
    State &anchor = *reinterpret_cast<State *>(s_anchor);
    Camera *scams = reinterpret_cast<Camera *>(s_cams);

    block_copy(s_cur, state_io, sizeof(State));
    block_copy(s_anchor, anchor_in, sizeof(State));
    block_copy(s_cams, cams, num_cams * sizeof(Camera));
    __syncthreads();
    if (threadIdx.x == 0) {
        state.pose.pCamera[0] = &scams[0];
        state.pose.pCamera[1] = num_cams > 1 ? &scams[1] : nullptr;
        anchor.pose.pCamera[0] = &scams[0];
        anchor.pose.pCamera[1] = num_cams > 1 ? &scams[1] : nullptr;
    }
    __syncthreads();

    InertialKFProblem<Camera> problem{factors, active, nfac,   FT(5.991), FT(7.815),
                                      &anchor, imu,    Omega9, InfoG,     InfoA,
                                      s_work};

    int nbad = 0, nin_mono = 0, nin_stereo = 0;
    for (int round = 0; round < rounds; round++) {
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            active[i] = outlier[i] ? 0 : 1;
        }
        if (round >= 3) {
            problem.huber_mono2 = kNoRobust;
            problem.huber_stereo2 = kNoRobust;
        }
        __syncthreads();

        fused_levenberg_marquardt<15, FT>(problem, state, backup, sc, opt);

        if (threadIdx.x < 3) s_counts[threadIdx.x] = 0;
        __syncthreads();

        const FT cm = chi2_mono[round];
        const FT cs = chi2_stereo[round];
        const FT cclose = FT(1.5) * cm;
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            const VisualFactor &f = factors[i];
            gpu::Vec3<FT> Xw;
            Xw << f.Xw[0], f.Xw[1], f.Xw[2];
            const FT c = visual_chi2(state.pose, f, FT(5.991), FT(7.815), false);
            bool bad;
            if (f.is_stereo) {
                bad = c > cs;
            } else {
                bad = (f.is_close ? c > cclose : c > cm) ||
                      !state.pose.isDepthPositive(Xw, f.cam_idx);
            }
            outlier[i] = bad ? 1 : 0;
            if (bad) atomicAdd(&s_counts[0], 1);
            else atomicAdd(&s_counts[f.is_stereo ? 2 : 1], 1);
        }
        __syncthreads();
        nbad = s_counts[0];
        nin_mono = s_counts[1];
        nin_stereo = s_counts[2];

        // +3 for the always-active inertial, gyro- and acc-random-walk factors.
        if (nin_mono + nin_stereo + 3 < 10) break;
    }

    if (threadIdx.x == 0) {
        state.pose.pCamera[0] = nullptr;
        state.pose.pCamera[1] = nullptr;
        *state_io = state;
        out_counts[0] = nbad;
        out_counts[1] = nin_mono;
        out_counts[2] = nin_stereo;
    }
}

// ---------------------------------------------------------------------------
// Problem 3: current frame and previous frame jointly (30 unknowns), the
// PoseInertialOptimizationLastFrame path.  The Hessian layout matches the CPU
// implementation -- previous frame first -- so the marginalisation that
// produces the next prior is the same Schur complement over rows 0..14.
// ---------------------------------------------------------------------------
template <typename Camera> struct TwoFrameState {
    gpu::ImuCamPose<FT, Camera> prev_pose;   //  0.. 5
    gpu::Vec3<FT> prev_vel;                  //  6.. 8
    gpu::Vec3<FT> prev_gbias;                //  9..11
    gpu::Vec3<FT> prev_abias;                // 12..14
    gpu::ImuCamPose<FT, Camera> pose;        // 15..20
    gpu::Vec3<FT> vel;                       // 21..23
    gpu::Vec3<FT> gbias;                     // 24..26
    gpu::Vec3<FT> abias;                     // 27..29
};

// EdgePriorPoseImu's prior state, held on the device alongside its information.
struct ImuPriorData {
    FT Rwb[9]; // row major
    FT twb[3];
    FT vwb[3];
    FT bg[3];
    FT ba[3];
};

template <typename Camera> struct InertialFrameProblem {
    using State = TwoFrameState<Camera>;
    static constexpr int dim = 30;

    const VisualFactor *factors;
    const unsigned char *active;
    int nfac;
    FT huber_mono2;
    FT huber_stereo2;

    const gpu::InertialConstraintData<FT> *imu;
    const FT *Omega9;   //  9x9
    const FT *InfoG;    //  3x3
    const FT *InfoA;    //  3x3
    const FT *PriorH;   // 15x15
    const ImuPriorData *prior;
    int has_prior;
    FT *scratch; // shared-memory workspace, see kernel

    using PoseDesc = gpu::PoseDescriptor<FT, FT, Camera>;
    using InerC = gpu::InertialConstraint<FT, FT, graphite::DefaultLoss<FT, 9>, PoseDesc>;
    using PriorC = gpu::ImuPriorConstraint<FT, FT, graphite::HuberLoss<FT, 15>, PoseDesc>;

    __device__ typename PriorC::Data prior_data() const {
        typename PriorC::Data d;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) d.Rwb(r, c) = prior->Rwb[r * 3 + c];
            d.twb[r] = prior->twb[r];
            d.vwb[r] = prior->vwb[r];
            d.bg[r] = prior->bg[r];
            d.ba[r] = prior->ba[r];
        }
        return d;
    }

    // Stacked 9x24 Jacobian over columns 0..23 (prev pose, prev vel, prev gyro
    // bias, prev acc bias, current pose, current velocity).
    __device__ void inertial_residual(const State &s, FT *e, FT *J) const {
        InerC::template error<FT>(s.prev_pose, s.prev_vel, s.prev_gbias,
                                  s.prev_abias, s.pose, s.vel, *imu, e);
        if (!J) return;
        // Column offsets of each vertex block in the stacked 9x24 Jacobian.
        InerC::template jacobian_all<FT>(s.prev_pose, s.prev_vel, s.prev_gbias,
                                         s.prev_abias, s.pose, s.vel, *imu, J);
    }

    // Stacked 15x15 prior Jacobian over columns 0..14.
    __device__ void prior_residual(const State &s, FT *e, FT *J) const {
        const typename PriorC::Data d = prior_data();
        PriorC::template error<FT>(s.prev_pose, s.prev_vel, s.prev_gbias, s.prev_abias, d, e);
        if (!J) return;
        PriorC::template jacobian<FT, 0>(s.prev_pose, s.prev_vel, s.prev_gbias, s.prev_abias, d, J + 0 * 15);
        PriorC::template jacobian<FT, 1>(s.prev_pose, s.prev_vel, s.prev_gbias, s.prev_abias, d, J + 6 * 15);
        PriorC::template jacobian<FT, 2>(s.prev_pose, s.prev_vel, s.prev_gbias, s.prev_abias, d, J + 9 * 15);
        PriorC::template jacobian<FT, 3>(s.prev_pose, s.prev_vel, s.prev_gbias, s.prev_abias, d, J + 12 * 15);
    }

    // Random-walk factor between two 3-vectors at `col_prev` and `col_curr`.
    __device__ void accumulate_rw(const FT *Info, const FT *bprev, const FT *bcurr,
                                  int col_prev, int col_curr, FT *H, FT *b,
                                  FT *chi2) const {
        FT e[3];
        for (int i = 0; i < 3; i++) e[i] = bcurr[i] - bprev[i];
        FT Oe[3];
        for (int i = 0; i < 3; i++) {
            FT p = 0;
            for (int j = 0; j < 3; j++) p += Info[i * 3 + j] * e[j];
            Oe[i] = p;
        }
        FT c = 0;
        for (int i = 0; i < 3; i++) c += Oe[i] * e[i];
        atomicAdd(chi2, c);

        // J_prev = -I, J_curr = +I.
        for (int i = 0; i < 3; i++) {
            atomicAdd(&b[col_prev + i], Oe[i]);
            atomicAdd(&b[col_curr + i], -Oe[i]);
            for (int j = 0; j < 3; j++) {
                const FT v = Info[i * 3 + j];
                atomicAdd(&H[(col_prev + i) * dim + col_prev + j], v);
                atomicAdd(&H[(col_curr + i) * dim + col_curr + j], v);
                atomicAdd(&H[(col_prev + i) * dim + col_curr + j], -v);
                atomicAdd(&H[(col_curr + i) * dim + col_prev + j], -v);
            }
        }
    }

    __device__ void linearize(const State &s, FT *H, FT *b, FT *chi2) const {
        VisualAccum acc;
        acc.clear();
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            if (!active[i]) continue;
            accumulate_visual<Camera>(s.pose, factors[i], huber_mono2, huber_stereo2, acc);
        }
        acc.warp_reduce();
        acc.flush<dim>(H, b, chi2, 15);

        // Shared scratch: inertial J(9x24), Omega*J, residual and Omega*e, then
        // the same four for the 15x15 prior, then the prior's robust weight.
        FT *Ji = scratch;              // 216
        FT *OJi = scratch + 216;       // 216
        FT *ei = scratch + 432;        // 9
        FT *Oei = scratch + 441;       // 9
        FT *Jp = scratch + 450;        // 225
        FT *OJp = scratch + 675;       // 225
        FT *ep = scratch + 900;        // 15
        FT *Oep = scratch + 915;       // 15
        FT *wp = scratch + 930;        // 1

        if (threadIdx.x == 0) {
            inertial_residual(s, ei, Ji);
            atomicAdd(chi2, weighted_sq<9>(Omega9, ei));
        } else if (threadIdx.x == 32 && has_prior) {
            prior_residual(s, ep, Jp);
            const FT raw = weighted_sq<15>(PriorH, ep);
            // EdgePriorPoseImu carries a Huber kernel with delta = 5.
            const FT delta2 = FT(25);
            atomicAdd(chi2, huber_loss(raw, delta2));
            *wp = huber_weight(raw, delta2);
        } else if (threadIdx.x == 64) {
            accumulate_rw(InfoG, &s.prev_gbias[0], &s.gbias[0], 9, 24, H, b, chi2);
        } else if (threadIdx.x == 96) {
            accumulate_rw(InfoA, &s.prev_abias[0], &s.abias[0], 12, 27, H, b, chi2);
        }
        __syncthreads();

        accumulate_dense_block<dim, 9, 24>(Ji, Omega9, ei, FT(1), H, b, 0, OJi, Oei);
        if (has_prior) {
            accumulate_dense_block<dim, 15, 15>(Jp, PriorH, ep, *wp, H, b, 0, OJp, Oep);
        }
    }

    __device__ void chi2(const State &s, FT *chi2) const {
        FT acc = 0;
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            if (!active[i]) continue;
            acc += visual_chi2(s.pose, factors[i], huber_mono2, huber_stereo2, true);
        }
        if (threadIdx.x == 0) {
            FT e[9];
            inertial_residual(s, e, nullptr);
            acc += weighted_sq<9>(Omega9, e);
        } else if (threadIdx.x == 32 && has_prior) {
            FT e[15];
            prior_residual(s, e, nullptr);
            acc += huber_loss(weighted_sq<15>(PriorH, e), FT(25));
        } else if (threadIdx.x == 64) {
            FT eg[3], ea[3];
            for (int i = 0; i < 3; i++) {
                eg[i] = s.gbias[i] - s.prev_gbias[i];
                ea[i] = s.abias[i] - s.prev_abias[i];
            }
            acc += weighted_sq<3>(InfoG, eg) + weighted_sq<3>(InfoA, ea);
        }
        if (acc != FT(0)) atomicAdd(chi2, acc);
    }

    __device__ void update(State &s, const FT *dx) const {
        s.prev_pose.Update(dx);
        s.pose.Update(dx + 15);
        for (int i = 0; i < 3; i++) {
            s.prev_vel[i]   += dx[6 + i];
            s.prev_gbias[i] += dx[9 + i];
            s.prev_abias[i] += dx[12 + i];
            s.vel[i]        += dx[21 + i];
            s.gbias[i]      += dx[24 + i];
            s.abias[i]      += dx[27 + i];
        }
    }
};

template <typename Camera>
__global__ __launch_bounds__(kBlockThreads) void inertial_frame_kernel(
    TwoFrameState<Camera> *state_io, const Camera *cams, int num_cams,
    const VisualFactor *factors, int nfac, unsigned char *active,
    unsigned char *outlier, int *out_counts,
    const gpu::InertialConstraintData<FT> *imu, const FT *Omega9, const FT *InfoG,
    const FT *InfoA, const FT *PriorH, const ImuPriorData *prior, int has_prior,
    FusedLMOptions opt, const FT *chi2_mono, const FT *chi2_stereo, int rounds) {
    using State = TwoFrameState<Camera>;

    __shared__ __align__(16) unsigned char s_cur[sizeof(State)];
    __shared__ __align__(16) unsigned char s_bak[sizeof(State)];
    __shared__ __align__(16) unsigned char s_cams[2 * sizeof(Camera)];
    __shared__ FusedLMScratch<30, FT> sc;
    // 9x24 inertial Jacobian and its Omega*J, then the 15x15 prior pair.
    __shared__ FT s_work[931]; // see InertialFrameProblem::linearize
    __shared__ int s_counts[3];

    State &state = *reinterpret_cast<State *>(s_cur);
    State &backup = *reinterpret_cast<State *>(s_bak);
    Camera *scams = reinterpret_cast<Camera *>(s_cams);

    block_copy(s_cur, state_io, sizeof(State));
    block_copy(s_cams, cams, num_cams * sizeof(Camera));
    __syncthreads();
    if (threadIdx.x == 0) {
        state.pose.pCamera[0] = &scams[0];
        state.pose.pCamera[1] = num_cams > 1 ? &scams[1] : nullptr;
        state.prev_pose.pCamera[0] = &scams[0];
        state.prev_pose.pCamera[1] = num_cams > 1 ? &scams[1] : nullptr;
    }
    __syncthreads();

    InertialFrameProblem<Camera> problem{factors, active, nfac,  FT(5.991), FT(7.815),
                                         imu,     Omega9, InfoG, InfoA,     PriorH,
                                         prior,   has_prior, s_work};

    int nbad = 0, nin_mono = 0, nin_stereo = 0;
    for (int round = 0; round < rounds; round++) {
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            active[i] = outlier[i] ? 0 : 1;
        }
        if (round >= 3) {
            problem.huber_mono2 = kNoRobust;
            problem.huber_stereo2 = kNoRobust;
        }
        __syncthreads();

        fused_levenberg_marquardt<30, FT>(problem, state, backup, sc, opt);

        if (threadIdx.x < 3) s_counts[threadIdx.x] = 0;
        __syncthreads();

        const FT cm = chi2_mono[round];
        const FT cs = chi2_stereo[round];
        const FT cclose = FT(1.5) * cm;
        for (int i = threadIdx.x; i < nfac; i += blockDim.x) {
            const VisualFactor &f = factors[i];
            gpu::Vec3<FT> Xw;
            Xw << f.Xw[0], f.Xw[1], f.Xw[2];
            const FT c = visual_chi2(state.pose, f, FT(5.991), FT(7.815), false);
            bool bad;
            if (f.is_stereo) {
                bad = c > cs;
            } else {
                bad = (f.is_close ? c > cclose : c > cm) ||
                      !state.pose.isDepthPositive(Xw, f.cam_idx);
            }
            outlier[i] = bad ? 1 : 0;
            if (bad) atomicAdd(&s_counts[0], 1);
            else atomicAdd(&s_counts[f.is_stereo ? 2 : 1], 1);
        }
        __syncthreads();
        nbad = s_counts[0];
        nin_mono = s_counts[1];
        nin_stereo = s_counts[2];

        if (nin_mono + nin_stereo + 3 < 10) break;
    }

    if (threadIdx.x == 0) {
        state.pose.pCamera[0] = nullptr;
        state.pose.pCamera[1] = nullptr;
        state.prev_pose.pCamera[0] = nullptr;
        state.prev_pose.pCamera[1] = nullptr;
        *state_io = state;
        out_counts[0] = nbad;
        out_counts[1] = nin_mono;
        out_counts[2] = nin_stereo;
    }
}

// ---------------------------------------------------------------------------
// Per-thread persistent device/host buffers.  Tracking calls these once a
// frame; allocating and freeing on every call would cost more than the solve.
// ---------------------------------------------------------------------------
template <typename Camera> struct FusedContext {
    cudaStream_t stream = nullptr;

    VisualFactor *h_factors = nullptr;
    VisualFactor *d_factors = nullptr;
    unsigned char *h_outlier = nullptr;
    unsigned char *d_outlier = nullptr;
    unsigned char *d_active = nullptr;
    size_t capacity = 0;

    std::vector<int> frame_idx;

    // One pinned staging block for everything else, so a call is two copies.
    void *h_misc = nullptr;
    void *d_misc = nullptr;
    size_t misc_size = 0;

    void ensure_stream() {
        if (stream) return;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        // The residual/Jacobian functors are large enough that nvcc keeps them
        // as real device calls, and their Eigen temporaries do not fit in the
        // 1 KB default per-thread device stack -- the symptom is a "Warp
        // Illegal Address" rather than anything that names the stack.
        size_t stack = 0;
        if (cudaDeviceGetLimit(&stack, cudaLimitStackSize) == cudaSuccess &&
            stack < 4096) {
            cudaDeviceSetLimit(cudaLimitStackSize, 4096);
        }
    }

    // Grows geometrically and reallocates every buffer together, so the factor
    // count is bounded only by what the device can allocate. Returns false if
    // it cannot, which the callers treat as "fall back to the CPU solver"
    // rather than dereferencing a null pointer.
    bool ensure_factors(size_t n) {
        if (n <= capacity) return true;
        size_t cap = capacity ? capacity : 512;
        while (cap < n) cap *= 2;
        if (h_factors) {
            cudaFreeHost(h_factors);
            cudaFreeHost(h_outlier);
            cudaFree(d_factors);
            cudaFree(d_outlier);
            cudaFree(d_active);
            h_factors = nullptr;
            h_outlier = nullptr;
            d_factors = nullptr;
            d_outlier = nullptr;
            d_active = nullptr;
            capacity = 0;
        }
        bool ok = cudaMallocHost(&h_factors, cap * sizeof(VisualFactor)) == cudaSuccess;
        ok = ok && cudaMallocHost(&h_outlier, cap) == cudaSuccess;
        ok = ok && cudaMalloc(&d_factors, cap * sizeof(VisualFactor)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_outlier, cap) == cudaSuccess;
        ok = ok && cudaMalloc(&d_active, cap) == cudaSuccess;
        if (!ok) {
            cudaGetLastError(); // clear the sticky error before returning
            return false;
        }
        capacity = cap;
        return true;
    }

    bool ensure_misc(size_t bytes) {
        if (bytes <= misc_size) return true;
        if (h_misc) {
            cudaFreeHost(h_misc);
            cudaFree(d_misc);
            h_misc = nullptr;
            d_misc = nullptr;
            misc_size = 0;
        }
        bool ok = cudaMallocHost(&h_misc, bytes) == cudaSuccess;
        ok = ok && cudaMalloc(&d_misc, bytes) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            return false;
        }
        misc_size = bytes;
        return true;
    }
};

template <typename Camera> FusedContext<Camera> &context() {
    static thread_local FusedContext<Camera> ctx;
    return ctx;
}

inline size_t align_up(size_t x) {
    constexpr size_t a = 16;
    return (x + a - 1) / a * a;
}

// Fill the two camera models the rig may have; returns how many there are.
template <typename Cam, typename FPT>
int fill_cameras(Frame *pFrame, Cam *cams) {
    std::array<FPT, Cam::parameter_size> p;
    for (size_t k = 0; k < p.size(); k++) p[k] = FPT(pFrame->mpCamera->getParameter(k));
    new (&cams[0]) Cam(p);
    if (!pFrame->mpCamera2) return 1;
    for (size_t k = 0; k < p.size(); k++) p[k] = FPT(pFrame->mpCamera2->getParameter(k));
    new (&cams[1]) Cam(p);
    return 2;
}

FusedLMOptions default_options() {
    FusedLMOptions opt;
    opt.iterations = 10;
    opt.initial_damping = 1e-4;
    opt.use_identity = false;
    return opt;
}

} // namespace

// ---------------------------------------------------------------------------
// PoseOptimization (visual only)
// ---------------------------------------------------------------------------
template <typename Camera, size_t max_cameras>
static int PoseOptimizationFusedInternal(Frame *pFrame) {
    using Pose = gpu::ImuCamPose<FT, Camera>;

    FusedContext<Camera> &ctx = context<Camera>();
    ctx.ensure_stream();

    const int N = pFrame->N;
    if (!ctx.ensure_factors(static_cast<size_t>(N > 0 ? N : 1))) {
        return Optimizer::PoseOptimization(pFrame);
    }

    std::vector<int> &frame_idx = ctx.frame_idx;
    frame_idx.clear();
    frame_idx.reserve(N);
    int nfac = 0;
    {
        std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);
        for (int i = 0; i < N; i++) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (!pMP) continue;
            pFrame->mvbOutlier[i] = false;

            VisualFactor &f = ctx.h_factors[nfac];
            const Eigen::Vector3f Xw = pMP->GetWorldPos();
            f.Xw[0] = Xw[0]; f.Xw[1] = Xw[1]; f.Xw[2] = Xw[2];
            f.is_close = 0;

            if (!pFrame->mpCamera2) {
                const cv::KeyPoint &kp = pFrame->mvKeysUn[i];
                f.cam_idx = 0;
                f.obs[0] = kp.pt.x;
                f.obs[1] = kp.pt.y;
                if (pFrame->mvuRight[i] < 0) {
                    f.is_stereo = 0;
                    f.obs[2] = 0;
                } else {
                    f.is_stereo = 1;
                    f.obs[2] = pFrame->mvuRight[i];
                }
                f.invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
            } else {
                const cv::KeyPoint &kp = (i < pFrame->Nleft)
                                             ? pFrame->mvKeys[i]
                                             : pFrame->mvKeysRight[i - pFrame->Nleft];
                f.cam_idx = (i < pFrame->Nleft) ? 0 : 1;
                f.is_stereo = 0;
                f.obs[0] = kp.pt.x;
                f.obs[1] = kp.pt.y;
                f.obs[2] = 0;
                f.invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
            }
            ctx.h_outlier[nfac] = 0;
            frame_idx.push_back(i);
            nfac++;
        }
    }

    if (nfac < 3) return 0;

    const size_t off_pose = 0;
    const size_t off_cams = align_up(sizeof(Pose));
    const size_t off_nbad = off_cams + align_up(max_cameras * sizeof(Camera));
    const size_t misc_bytes = off_nbad + align_up(sizeof(int));
    if (!ctx.ensure_misc(misc_bytes)) {
        return Optimizer::PoseOptimization(pFrame);
    }

    char *hb = static_cast<char *>(ctx.h_misc);
    char *db = static_cast<char *>(ctx.d_misc);
    auto *h_pose = reinterpret_cast<Pose *>(hb + off_pose);
    auto *h_cams = reinterpret_cast<Camera *>(hb + off_cams);
    auto *h_nbad = reinterpret_cast<int *>(hb + off_nbad);

    const int num_cams = fill_cameras<Camera, FT>(pFrame, h_cams);
    Camera *cam_ptrs[2] = {&h_cams[0], num_cams > 1 ? &h_cams[1] : nullptr};
    new (h_pose) Pose(pFrame, cam_ptrs);
    *h_nbad = 0;

    cudaMemcpyAsync(ctx.d_misc, ctx.h_misc, misc_bytes, cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_factors, ctx.h_factors, nfac * sizeof(VisualFactor),
                    cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_outlier, ctx.h_outlier, nfac, cudaMemcpyHostToDevice, ctx.stream);

    pose_only_kernel<Camera><<<1, kBlockThreads, 0, ctx.stream>>>(
        reinterpret_cast<Pose *>(db + off_pose),
        reinterpret_cast<Camera *>(db + off_cams), num_cams, ctx.d_factors, nfac,
        ctx.d_active, ctx.d_outlier, reinterpret_cast<int *>(db + off_nbad),
        default_options(), FT(5.991), FT(7.815), 4);

    cudaMemcpyAsync(ctx.h_misc, ctx.d_misc, misc_bytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaMemcpyAsync(ctx.h_outlier, ctx.d_outlier, nfac, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    for (int j = 0; j < nfac; j++) {
        pFrame->mvbOutlier[frame_idx[j]] = ctx.h_outlier[j] != 0;
    }
    pFrame->SetPose(Sophus::SE3f(h_pose->Rcw[0].template cast<float>(),
                                 h_pose->tcw[0].template cast<float>()));
    return nfac - *h_nbad;
}

int PoseOptimizationFused(Frame *pFrame) {
    if (pFrame->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        return PoseOptimizationFusedInternal<gpu::PinholeCamera<FT>, 2>(pFrame);
    }
    return PoseOptimizationFusedInternal<gpu::KannalaBrandt8Camera<FT>, 2>(pFrame);
}

// ---------------------------------------------------------------------------
// Shared host-side helpers for the two inertial variants
// ---------------------------------------------------------------------------
namespace {

// Build the visual factor list for the inertial variants (which weight by the
// camera's angular uncertainty and tag close points).  Returns the factor count
// and fills the mono/stereo tallies.
template <typename Camera>
int build_inertial_factors(Frame *pFrame, FusedContext<Camera> &ctx,
                           std::vector<int> &frame_idx, int &nMono, int &nStereo) {
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft != -1);

    frame_idx.clear();
    frame_idx.reserve(N);
    int nfac = 0;
    nMono = 0;
    nStereo = 0;

    std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);
    for (int i = 0; i < N; i++) {
        MapPoint *pMP = pFrame->mvpMapPoints[i];
        if (!pMP) continue;
        const Eigen::Vector3f Xw = pMP->GetWorldPos();

        auto emit = [&](int cam_idx, const cv::KeyPoint &kp, bool stereo, float ur,
                        bool close) {
            VisualFactor &f = ctx.h_factors[nfac];
            f.Xw[0] = Xw[0]; f.Xw[1] = Xw[1]; f.Xw[2] = Xw[2];
            f.obs[0] = kp.pt.x;
            f.obs[1] = kp.pt.y;
            f.obs[2] = stereo ? ur : 0.f;
            f.cam_idx = cam_idx;
            f.is_stereo = stereo ? 1 : 0;
            f.is_close = close ? 1 : 0;
            const Eigen::Vector2d o(kp.pt.x, kp.pt.y);
            const float unc2 = pFrame->mpCamera->uncertainty2(o);
            f.invSigma2 = pFrame->mvInvLevelSigma2[kp.octave] / unc2;
            ctx.h_outlier[nfac] = 0;
            frame_idx.push_back(i);
            nfac++;
        };

        pFrame->mvbOutlier[i] = false;
        if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft) {
            const cv::KeyPoint &kp = (i < Nleft) ? pFrame->mvKeys[i] : pFrame->mvKeysUn[i];
            emit(0, kp, false, 0.f, pMP->mTrackDepth < 10.f);
            nMono++;
        } else if (!bRight) {
            const cv::KeyPoint &kp = pFrame->mvKeysUn[i];
            emit(0, kp, true, pFrame->mvuRight[i], false);
            nStereo++;
        }
        if (bRight && i >= Nleft) {
            const cv::KeyPoint &kp = pFrame->mvKeysRight[i - Nleft];
            emit(1, kp, false, 0.f, pMP->mTrackDepth < 10.f);
            nMono++;
        }
    }
    return nfac;
}

// The recovery pass both inertial variants run when tracking is about to be
// lost: re-admit observations under much laxer thresholds.
template <typename Camera>
int relax_outliers(Frame *pFrame, const FusedContext<Camera> &ctx, int nfac,
                   const std::vector<int> &frame_idx,
                   const gpu::ImuCamPose<FT, Camera> &pose) {
    int nBad = 0;
    const FT chi2MonoOut = 18.f, chi2StereoOut = 24.f;
    for (int j = 0; j < nfac; j++) {
        const VisualFactor &f = ctx.h_factors[j];
        gpu::Vec3<FT> Xw;
        Xw << f.Xw[0], f.Xw[1], f.Xw[2];
        FT r2;
        if (f.is_stereo) {
            gpu::Vec3<FT> obs;
            obs << f.obs[0], f.obs[1], f.obs[2];
            r2 = (obs - pose.ProjectStereo(Xw, f.cam_idx)).squaredNorm();
        } else {
            gpu::Vec2<FT> obs;
            obs << f.obs[0], f.obs[1];
            r2 = (obs - pose.Project(Xw, f.cam_idx)).squaredNorm();
        }
        const FT c = r2 * f.invSigma2;
        if (c < (f.is_stereo ? chi2StereoOut : chi2MonoOut))
            pFrame->mvbOutlier[frame_idx[j]] = false;
        else
            nBad++;
    }
    return nBad;
}

// Widen a converged pose to double.  The marginalisation prior is consumed by
// the CPU optimizer, so it is built in double even though the solve ran in
// single precision; going through the members (rather than reconstructing from
// the Frame) keeps the previous frame's optimized-but-not-written-back state
// available, which is what the CPU path linearises at.
template <typename CameraD, typename CameraF>
gpu::ImuCamPose<double, CameraD> widen(const gpu::ImuCamPose<FT, CameraF> &p,
                                       CameraD *cams, int num_cams) {
    gpu::ImuCamPose<double, CameraD> o;
    o.Rwb = p.Rwb.template cast<double>();
    o.twb = p.twb.template cast<double>();
    for (int i = 0; i < num_cams; i++) {
        o.Rcw[i] = p.Rcw[i].template cast<double>();
        o.tcw[i] = p.tcw[i].template cast<double>();
        o.Rcb[i] = p.Rcb[i].template cast<double>();
        o.tcb[i] = p.tcb[i].template cast<double>();
        o.Rbc[i] = p.Rbc[i].template cast<double>();
        o.tbc[i] = p.tbc[i].template cast<double>();
        o.pCamera[i] = &cams[i];
    }
    o.num_cams = num_cams;
    o.bf = double(p.bf);
    o.Rwb0 = p.Rwb0.template cast<double>();
    o.DR = p.DR.template cast<double>();
    o.its = p.its;
    return o;
}

// Visual block of the marginalisation Hessian, evaluated in double at the
// optimized pose so the prior handed to the next frame keeps full precision.
template <typename CameraD>
void accumulate_visual_hessian(Frame *pFrame, const VisualFactor *factors, int nfac,
                               const std::vector<int> &frame_idx,
                               const gpu::ImuCamPose<double, CameraD> &pose,
                               Eigen::Ref<Eigen::Matrix<double, 6, 6> > Hout) {
    using MonoC = gpu::MonoConstraintOnlyPose<double, double, graphite::HuberLoss<double, 2>, CameraD>;
    using StereoC = gpu::StereoConstraintOnlyPose<double, double, graphite::HuberLoss<double, 3>, CameraD>;
    for (int j = 0; j < nfac; j++) {
        if (pFrame->mvbOutlier[frame_idx[j]]) continue;
        const VisualFactor &f = factors[j];
        gpu::PoseOnlyData<double> d;
        d.Xw << f.Xw[0], f.Xw[1], f.Xw[2];
        d.cam_idx = f.cam_idx;
        if (f.is_stereo) {
            gpu::Vec3<double> obs;
            obs << f.obs[0], f.obs[1], f.obs[2];
            double Jm[3 * 6];
            StereoC::template jacobian<double, 0>(pose, obs, d, Jm);
            Eigen::Map<const Eigen::Matrix<double, 3, 6> > J(Jm);
            Hout += J.transpose() * double(f.invSigma2) * J;
        } else {
            gpu::Vec2<double> obs;
            obs << f.obs[0], f.obs[1];
            double Jm[2 * 6];
            MonoC::template jacobian<double, 0>(pose, obs, d, Jm);
            Eigen::Map<const Eigen::Matrix<double, 2, 6> > J(Jm);
            Hout += J.transpose() * double(f.invSigma2) * J;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// PoseInertialOptimizationLastKeyFrame
// ---------------------------------------------------------------------------
template <typename Camera, typename CameraD, size_t max_cameras>
static int PoseInertialOptimizationLastKeyFrameFusedInternal(Frame *pFrame,
                                                             bool bRecInit) {
    using Pose = gpu::ImuCamPose<FT, Camera>;
    using State = NavState<Camera>;
    using ImuData = gpu::InertialConstraintData<FT>;

    FusedContext<Camera> &ctx = context<Camera>();
    ctx.ensure_stream();

    KeyFrame *pKF = pFrame->mpLastKeyFrame;
    if (!ctx.ensure_factors(static_cast<size_t>(pFrame->N > 0 ? pFrame->N : 1))) {
        return Optimizer::PoseInertialOptimizationLastKeyFrame(pFrame, bRecInit);
    }

    std::vector<int> &frame_idx = ctx.frame_idx;
    int nMono = 0, nStereo = 0;
    const int nfac = build_inertial_factors(pFrame, ctx, frame_idx, nMono, nStereo);
    const int nInitialCorrespondences = nMono + nStereo;

    size_t off = 0;
    const size_t off_state = off;   off += align_up(sizeof(State));
    const size_t off_anchor = off;  off += align_up(sizeof(State));
    const size_t off_cams = off;    off += align_up(max_cameras * sizeof(Camera));
    const size_t off_imu = off;     off += align_up(sizeof(ImuData));
    const size_t off_omega = off;   off += align_up(81 * sizeof(FT));
    const size_t off_infog = off;   off += align_up(9 * sizeof(FT));
    const size_t off_infoa = off;   off += align_up(9 * sizeof(FT));
    const size_t off_chi2m = off;   off += align_up(4 * sizeof(FT));
    const size_t off_chi2s = off;   off += align_up(4 * sizeof(FT));
    const size_t off_counts = off;  off += align_up(3 * sizeof(int));
    const size_t misc_bytes = off;
    if (!ctx.ensure_misc(misc_bytes)) {
        return Optimizer::PoseInertialOptimizationLastKeyFrame(pFrame, bRecInit);
    }

    char *hb = static_cast<char *>(ctx.h_misc);
    char *db = static_cast<char *>(ctx.d_misc);
    auto *h_state = reinterpret_cast<State *>(hb + off_state);
    auto *h_anchor = reinterpret_cast<State *>(hb + off_anchor);
    auto *h_cams = reinterpret_cast<Camera *>(hb + off_cams);
    auto *h_omega = reinterpret_cast<FT *>(hb + off_omega);
    auto *h_infog = reinterpret_cast<FT *>(hb + off_infog);
    auto *h_infoa = reinterpret_cast<FT *>(hb + off_infoa);
    auto *h_chi2m = reinterpret_cast<FT *>(hb + off_chi2m);
    auto *h_chi2s = reinterpret_cast<FT *>(hb + off_chi2s);
    auto *h_counts = reinterpret_cast<int *>(hb + off_counts);

    const int num_cams = fill_cameras<Camera, FT>(pFrame, h_cams);
    Camera *cam_ptrs[2] = {&h_cams[0], num_cams > 1 ? &h_cams[1] : nullptr};

    new (&h_state->pose) Pose(pFrame, cam_ptrs);
    h_state->vel = pFrame->GetVelocity().cast<FT>();
    h_state->gbias << FT(pFrame->mImuBias.bwx), FT(pFrame->mImuBias.bwy), FT(pFrame->mImuBias.bwz);
    h_state->abias << FT(pFrame->mImuBias.bax), FT(pFrame->mImuBias.bay), FT(pFrame->mImuBias.baz);

    new (&h_anchor->pose) Pose(pKF, cam_ptrs);
    h_anchor->vel = pKF->GetVelocity().cast<FT>();
    h_anchor->gbias = pKF->GetGyroBias().cast<FT>();
    h_anchor->abias = pKF->GetAccBias().cast<FT>();

    ImuData imu_data(pFrame->mpImuPreintegrated);
    std::memcpy(hb + off_imu, &imu_data, sizeof(ImuData));

    const Eigen::Matrix<double, 9, 9> Omega =
        imu_data.template get_information_matrix<double>(pFrame->mpImuPreintegrated);
    for (int r = 0; r < 9; r++)
        for (int c = 0; c < 9; c++) h_omega[r * 9 + c] = FT(Omega(r, c));

    const Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
    const Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            h_infog[r * 3 + c] = FT(InfoG(r, c));
            h_infoa[r * 3 + c] = FT(InfoA(r, c));
        }

    const float chi2Mono[4]   = {12.f, 7.5f, 5.991f, 5.991f};
    const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
    for (int k = 0; k < 4; k++) {
        h_chi2m[k] = FT(chi2Mono[k]);
        h_chi2s[k] = FT(chi2Stereo[k]);
    }
    h_counts[0] = h_counts[1] = h_counts[2] = 0;

    cudaMemcpyAsync(ctx.d_misc, ctx.h_misc, misc_bytes, cudaMemcpyHostToDevice, ctx.stream);
    if (nfac > 0) {
        cudaMemcpyAsync(ctx.d_factors, ctx.h_factors, nfac * sizeof(VisualFactor),
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_outlier, ctx.h_outlier, nfac, cudaMemcpyHostToDevice, ctx.stream);
    }

    inertial_kf_kernel<Camera><<<1, kBlockThreads, 0, ctx.stream>>>(
        reinterpret_cast<State *>(db + off_state),
        reinterpret_cast<State *>(db + off_anchor),
        reinterpret_cast<Camera *>(db + off_cams), num_cams, ctx.d_factors, nfac,
        ctx.d_active, ctx.d_outlier, reinterpret_cast<int *>(db + off_counts),
        reinterpret_cast<ImuData *>(db + off_imu),
        reinterpret_cast<FT *>(db + off_omega),
        reinterpret_cast<FT *>(db + off_infog),
        reinterpret_cast<FT *>(db + off_infoa), default_options(),
        reinterpret_cast<FT *>(db + off_chi2m),
        reinterpret_cast<FT *>(db + off_chi2s), 4);

    cudaMemcpyAsync(ctx.h_misc, ctx.d_misc, misc_bytes, cudaMemcpyDeviceToHost, ctx.stream);
    if (nfac > 0) {
        cudaMemcpyAsync(ctx.h_outlier, ctx.d_outlier, nfac, cudaMemcpyDeviceToHost, ctx.stream);
    }
    cudaStreamSynchronize(ctx.stream);

    int nBad = h_counts[0];
    const int nInliers = h_counts[1] + h_counts[2];
    for (int j = 0; j < nfac; j++) {
        pFrame->mvbOutlier[frame_idx[j]] = ctx.h_outlier[j] != 0;
    }

    h_state->pose.pCamera[0] = &h_cams[0];
    h_state->pose.pCamera[1] = num_cams > 1 ? &h_cams[1] : nullptr;
    if (nInliers < 30 && !bRecInit) {
        nBad = relax_outliers(pFrame, ctx, nfac, frame_idx, h_state->pose);
    }

    pFrame->SetImuPoseVelocity(h_state->pose.Rwb.template cast<float>(),
                               h_state->pose.twb.template cast<float>(),
                               h_state->vel.template cast<float>());
    pFrame->mImuBias = IMU::Bias(h_state->abias[0], h_state->abias[1], h_state->abias[2],
                                 h_state->gbias[0], h_state->gbias[1], h_state->gbias[2]);

    // ---- Marginalisation prior for the next frame, evaluated in double. ----
    alignas(CameraD) unsigned char camsD_buf[2 * sizeof(CameraD)];
    CameraD *camsD = reinterpret_cast<CameraD *>(camsD_buf);
    fill_cameras<CameraD, double>(pFrame, camsD);
    CameraD *camD_ptrs[2] = {&camsD[0], num_cams > 1 ? &camsD[1] : nullptr};
    const gpu::ImuCamPose<double, CameraD> poseD = widen<CameraD>(h_state->pose, camsD, num_cams);
    const gpu::ImuCamPose<double, CameraD> anchorD(pKF, camD_ptrs);
    const gpu::Vec3<double> velD = h_state->vel.template cast<double>();
    const gpu::Vec3<double> gbD(h_state->gbias[0], h_state->gbias[1], h_state->gbias[2]);
    const gpu::Vec3<double> abD(h_state->abias[0], h_state->abias[1], h_state->abias[2]);
    const gpu::Vec3<double> anchor_velD = pKF->GetVelocity().cast<double>();
    const gpu::Vec3<double> anchor_gbD = pKF->GetGyroBias().cast<double>();
    const gpu::Vec3<double> anchor_abD = pKF->GetAccBias().cast<double>();

    using InerCD = gpu::InertialConstraint<double, double, graphite::DefaultLoss<double, 9>,
                                           gpu::PoseDescriptor<double, double, CameraD> >;
    const gpu::InertialConstraintData<double> imuD(pFrame->mpImuPreintegrated);

    Eigen::Matrix<double, 15, 15> H;
    H.setZero();
    {
        double J4[9 * 6], J5[9 * 3];
        InerCD::template jacobian<double, 4>(anchorD, anchor_velD, anchor_gbD, anchor_abD,
                                             poseD, velD, imuD, J4);
        InerCD::template jacobian<double, 5>(anchorD, anchor_velD, anchor_gbD, anchor_abD,
                                             poseD, velD, imuD, J5);
        Eigen::Map<const Eigen::Matrix<double, 9, 6> > J4m(J4);
        Eigen::Map<const Eigen::Matrix<double, 9, 3> > J5m(J5);
        Eigen::Matrix<double, 9, 9> Jstack;
        Jstack.block<9, 6>(0, 0) = J4m;
        Jstack.block<9, 3>(0, 6) = J5m;
        H.block<9, 9>(0, 0) += Jstack.transpose() * Omega * Jstack;
    }
    H.block<3, 3>(9, 9) += InfoG;
    H.block<3, 3>(12, 12) += InfoA;
    {
        Eigen::Matrix<double, 6, 6> Hv = Eigen::Matrix<double, 6, 6>::Zero();
        accumulate_visual_hessian<CameraD>(pFrame, ctx.h_factors, nfac, frame_idx, poseD, Hv);
        H.block<6, 6>(0, 0) += Hv;
    }

    pFrame->mpcpi = NewConstraintPoseImu(poseD.Rwb, poseD.twb, velD, gbD, abD, H);
    return nInitialCorrespondences - nBad;
}

int PoseInertialOptimizationLastKeyFrameFused(Frame *pFrame, bool bRecInit) {
    if (pFrame->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        return PoseInertialOptimizationLastKeyFrameFusedInternal<
            gpu::PinholeCamera<FT>, gpu::PinholeCamera<double>, 2>(pFrame, bRecInit);
    }
    return PoseInertialOptimizationLastKeyFrameFusedInternal<
        gpu::KannalaBrandt8Camera<FT>, gpu::KannalaBrandt8Camera<double>, 2>(pFrame, bRecInit);
}

// ---------------------------------------------------------------------------
// PoseInertialOptimizationLastFrame
// ---------------------------------------------------------------------------
template <typename Camera, typename CameraD, size_t max_cameras>
static int PoseInertialOptimizationLastFrameFusedInternal(Frame *pFrame, bool bRecInit) {
    using Pose = gpu::ImuCamPose<FT, Camera>;
    using State = TwoFrameState<Camera>;
    using ImuData = gpu::InertialConstraintData<FT>;

    Frame *pFp = pFrame->mpPrevFrame;
    if (!pFp || !pFp->mpcpi) {
        // Without the previous frame's prior this problem is under-constrained;
        // the CPU path is the reference behaviour here.
        return Optimizer::PoseInertialOptimizationLastFrame(pFrame, bRecInit);
    }

    FusedContext<Camera> &ctx = context<Camera>();
    ctx.ensure_stream();
    if (!ctx.ensure_factors(static_cast<size_t>(pFrame->N > 0 ? pFrame->N : 1))) {
        return Optimizer::PoseInertialOptimizationLastFrame(pFrame, bRecInit);
    }

    std::vector<int> &frame_idx = ctx.frame_idx;
    int nMono = 0, nStereo = 0;
    const int nfac = build_inertial_factors(pFrame, ctx, frame_idx, nMono, nStereo);
    const int nInitialCorrespondences = nMono + nStereo;

    size_t off = 0;
    const size_t off_state = off;   off += align_up(sizeof(State));
    const size_t off_cams = off;    off += align_up(max_cameras * sizeof(Camera));
    const size_t off_imu = off;     off += align_up(sizeof(ImuData));
    const size_t off_omega = off;   off += align_up(81 * sizeof(FT));
    const size_t off_infog = off;   off += align_up(9 * sizeof(FT));
    const size_t off_infoa = off;   off += align_up(9 * sizeof(FT));
    const size_t off_priorH = off;  off += align_up(225 * sizeof(FT));
    const size_t off_prior = off;   off += align_up(sizeof(ImuPriorData));
    const size_t off_chi2m = off;   off += align_up(4 * sizeof(FT));
    const size_t off_chi2s = off;   off += align_up(4 * sizeof(FT));
    const size_t off_counts = off;  off += align_up(3 * sizeof(int));
    const size_t misc_bytes = off;
    if (!ctx.ensure_misc(misc_bytes)) {
        return Optimizer::PoseInertialOptimizationLastFrame(pFrame, bRecInit);
    }

    char *hb = static_cast<char *>(ctx.h_misc);
    char *db = static_cast<char *>(ctx.d_misc);
    auto *h_state = reinterpret_cast<State *>(hb + off_state);
    auto *h_cams = reinterpret_cast<Camera *>(hb + off_cams);
    auto *h_omega = reinterpret_cast<FT *>(hb + off_omega);
    auto *h_infog = reinterpret_cast<FT *>(hb + off_infog);
    auto *h_infoa = reinterpret_cast<FT *>(hb + off_infoa);
    auto *h_priorH = reinterpret_cast<FT *>(hb + off_priorH);
    auto *h_prior = reinterpret_cast<ImuPriorData *>(hb + off_prior);
    auto *h_chi2m = reinterpret_cast<FT *>(hb + off_chi2m);
    auto *h_chi2s = reinterpret_cast<FT *>(hb + off_chi2s);
    auto *h_counts = reinterpret_cast<int *>(hb + off_counts);

    const int num_cams = fill_cameras<Camera, FT>(pFrame, h_cams);
    Camera *cam_ptrs[2] = {&h_cams[0], num_cams > 1 ? &h_cams[1] : nullptr};

    new (&h_state->pose) Pose(pFrame, cam_ptrs);
    h_state->vel = pFrame->GetVelocity().cast<FT>();
    h_state->gbias << FT(pFrame->mImuBias.bwx), FT(pFrame->mImuBias.bwy), FT(pFrame->mImuBias.bwz);
    h_state->abias << FT(pFrame->mImuBias.bax), FT(pFrame->mImuBias.bay), FT(pFrame->mImuBias.baz);

    new (&h_state->prev_pose) Pose(pFp, cam_ptrs);
    h_state->prev_vel = pFp->GetVelocity().cast<FT>();
    h_state->prev_gbias << FT(pFp->mImuBias.bwx), FT(pFp->mImuBias.bwy), FT(pFp->mImuBias.bwz);
    h_state->prev_abias << FT(pFp->mImuBias.bax), FT(pFp->mImuBias.bay), FT(pFp->mImuBias.baz);

    ImuData imu_data(pFrame->mpImuPreintegratedFrame);
    std::memcpy(hb + off_imu, &imu_data, sizeof(ImuData));

    const Eigen::Matrix<double, 9, 9> Omega =
        imu_data.template get_information_matrix<double>(pFrame->mpImuPreintegratedFrame);
    for (int r = 0; r < 9; r++)
        for (int c = 0; c < 9; c++) h_omega[r * 9 + c] = FT(Omega(r, c));

    const Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3, 3>(9, 9).cast<double>().inverse();
    const Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3, 3>(12, 12).cast<double>().inverse();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            h_infog[r * 3 + c] = FT(InfoG(r, c));
            h_infoa[r * 3 + c] = FT(InfoA(r, c));
        }

    for (int r = 0; r < 15; r++)
        for (int c = 0; c < 15; c++) h_priorH[r * 15 + c] = FT(pFp->mpcpi->H(r, c));
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) h_prior->Rwb[r * 3 + c] = FT(pFp->mpcpi->Rwb(r, c));
        h_prior->twb[r] = FT(pFp->mpcpi->twb[r]);
        h_prior->vwb[r] = FT(pFp->mpcpi->vwb[r]);
        h_prior->bg[r] = FT(pFp->mpcpi->bg[r]);
        h_prior->ba[r] = FT(pFp->mpcpi->ba[r]);
    }

    const float chi2Mono[4]   = {12.f, 7.5f, 5.991f, 5.991f};
    const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
    for (int k = 0; k < 4; k++) {
        h_chi2m[k] = FT(chi2Mono[k]);
        h_chi2s[k] = FT(chi2Stereo[k]);
    }
    h_counts[0] = h_counts[1] = h_counts[2] = 0;

    cudaMemcpyAsync(ctx.d_misc, ctx.h_misc, misc_bytes, cudaMemcpyHostToDevice, ctx.stream);
    if (nfac > 0) {
        cudaMemcpyAsync(ctx.d_factors, ctx.h_factors, nfac * sizeof(VisualFactor),
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_outlier, ctx.h_outlier, nfac, cudaMemcpyHostToDevice, ctx.stream);
    }

    inertial_frame_kernel<Camera><<<1, kBlockThreads, 0, ctx.stream>>>(
        reinterpret_cast<State *>(db + off_state),
        reinterpret_cast<Camera *>(db + off_cams), num_cams, ctx.d_factors, nfac,
        ctx.d_active, ctx.d_outlier, reinterpret_cast<int *>(db + off_counts),
        reinterpret_cast<ImuData *>(db + off_imu),
        reinterpret_cast<FT *>(db + off_omega),
        reinterpret_cast<FT *>(db + off_infog),
        reinterpret_cast<FT *>(db + off_infoa),
        reinterpret_cast<FT *>(db + off_priorH),
        reinterpret_cast<ImuPriorData *>(db + off_prior), 1, default_options(),
        reinterpret_cast<FT *>(db + off_chi2m),
        reinterpret_cast<FT *>(db + off_chi2s), 4);

    cudaMemcpyAsync(ctx.h_misc, ctx.d_misc, misc_bytes, cudaMemcpyDeviceToHost, ctx.stream);
    if (nfac > 0) {
        cudaMemcpyAsync(ctx.h_outlier, ctx.d_outlier, nfac, cudaMemcpyDeviceToHost, ctx.stream);
    }
    cudaStreamSynchronize(ctx.stream);

    int nBad = h_counts[0];
    const int nInliers = h_counts[1] + h_counts[2];
    for (int j = 0; j < nfac; j++) {
        pFrame->mvbOutlier[frame_idx[j]] = ctx.h_outlier[j] != 0;
    }

    h_state->pose.pCamera[0] = &h_cams[0];
    h_state->pose.pCamera[1] = num_cams > 1 ? &h_cams[1] : nullptr;
    if (nInliers < 30 && !bRecInit) {
        nBad = relax_outliers(pFrame, ctx, nfac, frame_idx, h_state->pose);
    }

    pFrame->SetImuPoseVelocity(h_state->pose.Rwb.template cast<float>(),
                               h_state->pose.twb.template cast<float>(),
                               h_state->vel.template cast<float>());
    pFrame->mImuBias = IMU::Bias(h_state->abias[0], h_state->abias[1], h_state->abias[2],
                                 h_state->gbias[0], h_state->gbias[1], h_state->gbias[2]);
    // ---- 30x30 Hessian, then marginalise the previous frame out of it. ----
    alignas(CameraD) unsigned char camsD_buf[2 * sizeof(CameraD)];
    CameraD *camsD = reinterpret_cast<CameraD *>(camsD_buf);
    fill_cameras<CameraD, double>(pFrame, camsD);
    const gpu::ImuCamPose<double, CameraD> poseD = widen<CameraD>(h_state->pose, camsD, num_cams);
    const gpu::ImuCamPose<double, CameraD> prevD = widen<CameraD>(h_state->prev_pose, camsD, num_cams);
    const gpu::Vec3<double> velD = h_state->vel.template cast<double>();
    const gpu::Vec3<double> gbD(h_state->gbias[0], h_state->gbias[1], h_state->gbias[2]);
    const gpu::Vec3<double> abD(h_state->abias[0], h_state->abias[1], h_state->abias[2]);
    const gpu::Vec3<double> prev_velD = h_state->prev_vel.template cast<double>();
    const gpu::Vec3<double> prev_gbD(h_state->prev_gbias[0], h_state->prev_gbias[1], h_state->prev_gbias[2]);
    const gpu::Vec3<double> prev_abD(h_state->prev_abias[0], h_state->prev_abias[1], h_state->prev_abias[2]);

    using PoseDescD = gpu::PoseDescriptor<double, double, CameraD>;
    using InerCD = gpu::InertialConstraint<double, double, graphite::DefaultLoss<double, 9>, PoseDescD>;
    using PriorCD = gpu::ImuPriorConstraint<double, double, graphite::HuberLoss<double, 15>, PoseDescD>;
    const gpu::InertialConstraintData<double> imuD(pFrame->mpImuPreintegratedFrame);

    Eigen::Matrix<double, 30, 30, Eigen::RowMajor> H;
    H.setZero();
    {
        // [prev pose | prev vel | prev bg | prev ba | pose | vel] -> columns 0..23
        double Jb[6][9 * 6];
        InerCD::template jacobian<double, 0>(prevD, prev_velD, prev_gbD, prev_abD, poseD, velD, imuD, Jb[0]);
        InerCD::template jacobian<double, 1>(prevD, prev_velD, prev_gbD, prev_abD, poseD, velD, imuD, Jb[1]);
        InerCD::template jacobian<double, 2>(prevD, prev_velD, prev_gbD, prev_abD, poseD, velD, imuD, Jb[2]);
        InerCD::template jacobian<double, 3>(prevD, prev_velD, prev_gbD, prev_abD, poseD, velD, imuD, Jb[3]);
        InerCD::template jacobian<double, 4>(prevD, prev_velD, prev_gbD, prev_abD, poseD, velD, imuD, Jb[4]);
        InerCD::template jacobian<double, 5>(prevD, prev_velD, prev_gbD, prev_abD, poseD, velD, imuD, Jb[5]);
        const int width[6] = {6, 3, 3, 3, 6, 3};
        Eigen::Matrix<double, 9, 24> J;
        int col = 0;
        for (int blk = 0; blk < 6; blk++)
            for (int c = 0; c < width[blk]; c++, col++)
                for (int r = 0; r < 9; r++) J(r, col) = Jb[blk][c * 9 + r];
        H.block<24, 24>(0, 0) += J.transpose() * Omega * J;
    }
    // Random walks couple prev/current biases.
    H.block<3, 3>(9, 9) += InfoG;
    H.block<3, 3>(9, 24) -= InfoG;
    H.block<3, 3>(24, 9) -= InfoG;
    H.block<3, 3>(24, 24) += InfoG;
    H.block<3, 3>(12, 12) += InfoA;
    H.block<3, 3>(12, 27) -= InfoA;
    H.block<3, 3>(27, 12) -= InfoA;
    H.block<3, 3>(27, 27) += InfoA;
    {
        typename PriorCD::Data pd;
        pd.Rwb = pFp->mpcpi->Rwb;
        pd.twb = pFp->mpcpi->twb;
        pd.vwb = pFp->mpcpi->vwb;
        pd.bg = pFp->mpcpi->bg;
        pd.ba = pFp->mpcpi->ba;
        double Jb[4][15 * 6];
        PriorCD::template jacobian<double, 0>(prevD, prev_velD, prev_gbD, prev_abD, pd, Jb[0]);
        PriorCD::template jacobian<double, 1>(prevD, prev_velD, prev_gbD, prev_abD, pd, Jb[1]);
        PriorCD::template jacobian<double, 2>(prevD, prev_velD, prev_gbD, prev_abD, pd, Jb[2]);
        PriorCD::template jacobian<double, 3>(prevD, prev_velD, prev_gbD, prev_abD, pd, Jb[3]);
        const int width[4] = {6, 3, 3, 3};
        Eigen::Matrix<double, 15, 15> J;
        int col = 0;
        for (int blk = 0; blk < 4; blk++)
            for (int c = 0; c < width[blk]; c++, col++)
                for (int r = 0; r < 15; r++) J(r, col) = Jb[blk][c * 15 + r];
        H.block<15, 15>(0, 0) += J.transpose() * pFp->mpcpi->H * J;
    }
    {
        Eigen::Matrix<double, 6, 6> Hv = Eigen::Matrix<double, 6, 6>::Zero();
        accumulate_visual_hessian<CameraD>(pFrame, ctx.h_factors, nfac, frame_idx, poseD, Hv);
        H.block<6, 6>(15, 15) += Hv;
    }

    Eigen::Matrix<double, 15, 15> Hprior;
    Optimizer::MarginalizeFrameHessian(H.data(), Hprior.data());

    pFrame->mpcpi = NewConstraintPoseImu(poseD.Rwb, poseD.twb, velD, gbD, abD, Hprior);
    DeleteConstraintPoseImu(pFp->mpcpi);
    pFp->mpcpi = nullptr;

    return nInitialCorrespondences - nBad;
}

int PoseInertialOptimizationLastFrameFused(Frame *pFrame, bool bRecInit) {
    if (pFrame->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        return PoseInertialOptimizationLastFrameFusedInternal<
            gpu::PinholeCamera<FT>, gpu::PinholeCamera<double>, 2>(pFrame, bRecInit);
    }
    return PoseInertialOptimizationLastFrameFusedInternal<
        gpu::KannalaBrandt8Camera<FT>, gpu::KannalaBrandt8Camera<double>, 2>(pFrame, bRecInit);
}

} // namespace OptimizerGPU
} // namespace ORB_SLAM3
