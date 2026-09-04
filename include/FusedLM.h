/// @file FusedLM.h
///
/// Single-block, fully fused Levenberg-Marquardt for *small* problems.
///
/// The Graphite graph pipeline used by src/PoseOptimization.cu issues
/// on the order of twenty kernel launches and host synchronisations per LM
/// iteration.  That is the right trade-off when the Hessian has thousands of
/// block columns and each launch has real work to hide the latency behind, but
/// it is the wrong one for a problem whose state is six to thirty scalars and
/// whose factor count is in the hundreds: there the wall clock is essentially
/// (number of launches) x (launch + sync latency), and a CPU solver wins
/// outright.
///
/// This header runs the *entire* LM loop inside one kernel, on one block.  The
/// state lives in shared memory, the normal equations are accumulated with
/// shared-memory atomics, and the dense factorisation is done in registers, so
/// a whole optimisation costs a single launch.
///
/// The caller supplies a `Problem` policy with device-side methods:
///
///   static constexpr int dim;                 // state dimension
///   using State = ...;                        // trivially copyable
///   __device__ void linearize(const State&, T* H, T* b, T* chi2) const;
///   __device__ void chi2(const State&, T* chi2) const;
///   __device__ void update(State&, const T* dx) const;
///
/// `linearize` and `chi2` are called by every thread of the block and must
/// stride over the factors themselves, accumulating into the (already zeroed)
/// outputs with atomicAdd.  `H` is dim x dim row-major and must be filled
/// symmetrically; `b` is -J^T Omega r, matching graphite's compute_b_kernel.
/// `update` is called by a single thread.
///
/// The numerics deliberately mirror graphite::optimizer::levenberg_marquardt2
/// (Thirdparty/graphite/include/graphite/optimizer/levenberg_marquardt.hpp)
/// (Jacobian column scaling, diagonal-scaled damping, Nielsen's mu/nu update
/// and the ORB-SLAM-style relative-improvement early exit) so that switching a
/// problem over to the fused path changes its speed and not its answer.
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ORB_SLAM3 {
namespace fused {

/// Options for the fused solver.  Mirrors the subset of
/// LevenbergMarquardtOptions that makes sense without a host-side solver.
struct FusedLMOptions {
  int iterations = 10;
  double initial_damping = 1e-4;
  bool use_identity = false;
  /// Stop after this many consecutive accepted-but-negligible steps.
  int max_negligible_steps = 3;
  /// Reuse the damping factor from the previous call instead of resetting it.
  bool warm_start_damping = false;
};

/// Shared-memory scratch for a `dim`-dimensional problem.  The caller declares
/// one of these in shared memory; nothing here is allocated dynamically.
template <int Dim, typename T> struct FusedLMScratch {
  /// Scaled Hessian at the current linearisation point.
  T H0[Dim * Dim];
  /// Working copy: H0 plus damping, destroyed by the factorisation.
  T H[Dim * Dim];
  /// Scaled gradient (-J^T Omega r).
  T b[Dim];
  /// Jacobian column scaling, 1 / (eps + sqrt(diag H_raw)).
  T scale[Dim];
  /// Step in the scaled parameterisation, and in the real one.
  T delta[Dim];
  T dx[Dim];

  T chi2;
  T new_chi2;
  T mu;
  T nu;
  int num_bad;
  int solve_ok;
  bool run;
  bool step_accepted;
};

/// Solve `A x = b` for small dense symmetric positive definite `A` by LDL^T.
///
/// Warp-parallel with one matrix row per lane, which is what makes this usable
/// inside the LM loop: a single-threaded factorisation of a 30x30 system is
/// some 4500 dependent shared-memory operations, and at roughly thirty cycles
/// of latency apiece that alone cost more per iteration than everything else
/// in the kernel combined.  Only warp 0 participates; the rest of the block
/// waits at the caller's barrier.  Within the warp the steps are ordered by
/// __syncwarp(), which is effectively free next to __syncthreads().
///
/// `A` is dim x dim row-major and is overwritten.  `ok` (shared) is set to 0 if
/// the matrix is not numerically positive definite, which the caller treats the
/// same way a failed host solve is treated: reject the step and raise damping.
template <int Dim, typename T>
__device__ inline void dense_ldlt_solve(T *A, const T *b, T *x, int *ok) {
  static_assert(Dim <= 32, "the warp-parallel solve keeps one row per lane");
  if (threadIdx.x >= 32) {
    return;
  }
  const int i = threadIdx.x;
  if (i == 0) {
    *ok = 1;
  }
  __syncwarp();

  // A = L D L^T, L unit lower triangular in the strict lower triangle, D on the
  // diagonal.  Lane i owns row i.
  for (int k = 0; k < Dim; ++k) {
    const T dk = A[k * Dim + k];
    if (i == 0 && !(dk > T(0)) ) {
      *ok = 0;
    }
    if (i == 0 && !isfinite(dk)) {
      *ok = 0;
    }
    __syncwarp();
    if (*ok == 0) {
      return; // uniform across the warp: every lane read the same flag
    }
    if (i > k && i < Dim) {
      A[i * Dim + k] /= dk;
    }
    __syncwarp();
    if (i > k && i < Dim) {
      const T lik_dk = A[i * Dim + k] * dk;
      for (int j = k + 1; j <= i; ++j) {
        A[i * Dim + j] -= lik_dk * A[j * Dim + k];
      }
    }
    __syncwarp();
  }

  // Forward substitution, L y = b, column oriented so each step is a parallel
  // update of the remaining entries rather than a serial dot product.
  if (i < Dim) {
    x[i] = b[i];
  }
  __syncwarp();
  for (int k = 0; k < Dim; ++k) {
    const T xk = x[k];
    __syncwarp();
    if (i > k && i < Dim) {
      x[i] -= A[i * Dim + k] * xk;
    }
    __syncwarp();
  }
  // Diagonal solve, then back substitution against L^T.
  if (i < Dim) {
    x[i] /= A[i * Dim + i];
  }
  __syncwarp();
  for (int k = Dim - 1; k >= 0; --k) {
    const T xk = x[k];
    __syncwarp();
    if (i < k) {
      x[i] -= A[k * Dim + i] * xk;
    }
    __syncwarp();
  }

  if (i < Dim && !isfinite(x[i])) {
    *ok = 0;
  }
  __syncwarp();
}

/// Machine epsilon used by the column scaling, matching graph.hpp's use of
/// std::numeric_limits<double>::epsilon().  <limits> is not usable from device
/// code, so the two cases are spelled out.
template <typename T> __device__ inline T lm_epsilon() {
  return T(2.220446049250313e-16);
}
template <> __device__ inline float lm_epsilon<float>() { return 1.1920929e-7f; }

/// Stand-in for "worse than any real chi2", used when the linear solve fails.
/// Must stay finite so the isfinite() test still rejects the step.
template <typename T> __device__ inline T lm_huge() { return T(1.0e300); }
template <> __device__ inline float lm_huge<float>() { return 3.0e38f; }

/// Relinearise: recompute H, b, chi2 and the column scaling at `state`.
/// Block-cooperative; leaves the block synchronised on exit.
template <int Dim, typename T, typename Problem, typename State>
__device__ inline void fused_linearize(const Problem &problem,
                                       const State &state,
                                       FusedLMScratch<Dim, T> &sc) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  for (int i = tid; i < Dim * Dim; i += nthreads) {
    sc.H0[i] = T(0);
  }
  for (int i = tid; i < Dim; i += nthreads) {
    sc.b[i] = T(0);
  }
  if (tid == 0) {
    sc.chi2 = T(0);
  }
  __syncthreads();

  problem.linearize(state, sc.H0, sc.b, &sc.chi2);
  __syncthreads();

  if (tid < Dim) {
    const double diag = static_cast<double>(sc.H0[tid * Dim + tid]);
    sc.scale[tid] =
        static_cast<T>(1.0 / (static_cast<double>(lm_epsilon<T>()) +
                              sqrt(diag > 0.0 ? diag : 0.0)));
  }
  __syncthreads();

  for (int idx = tid; idx < Dim * Dim; idx += nthreads) {
    sc.H0[idx] *= sc.scale[idx / Dim] * sc.scale[idx % Dim];
  }
  for (int i = tid; i < Dim; i += nthreads) {
    sc.b[i] *= sc.scale[i];
  }
  __syncthreads();
}

/// Recompute only chi2 at `state`.  Block-cooperative; synchronised on exit.
template <int Dim, typename T, typename Problem, typename State>
__device__ inline void fused_chi2(const Problem &problem, const State &state,
                                  FusedLMScratch<Dim, T> &sc) {
  if (threadIdx.x == 0) {
    sc.new_chi2 = T(0);
  }
  __syncthreads();
  problem.chi2(state, &sc.new_chi2);
  __syncthreads();
}

/// Run the LM loop.  `state` and `backup` must live in shared memory; the whole
/// block must call this.  Returns with the block synchronised and `state`
/// holding the best iterate found.
template <int Dim, typename T, typename Problem, typename State>
__device__ inline void fused_levenberg_marquardt(const Problem &problem,
                                                 State &state, State &backup,
                                                 FusedLMScratch<Dim, T> &sc,
                                                 const FusedLMOptions &opt) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  if (tid == 0) {
    if (!opt.warm_start_damping) {
      sc.mu = static_cast<T>(opt.initial_damping);
    }
    sc.nu = T(2);
    sc.num_bad = 0;
    sc.run = true;
  }
  __syncthreads();

  fused_linearize<Dim, T>(problem, state, sc);

  for (int it = 0; it < opt.iterations; ++it) {
    const T initial_chi2 = sc.chi2;

    // Damp and solve.  One thread: the factorisation is tiny and serial.
    for (int i = tid; i < Dim * Dim; i += nthreads) {
      sc.H[i] = sc.H0[i];
    }
    __syncthreads();
    if (tid < Dim) {
      const double h = static_cast<double>(sc.H0[tid * Dim + tid]);
      const double mu = static_cast<double>(sc.mu);
      const double damped =
          opt.use_identity
              ? h + mu
              : h + mu * (h < 1.0e-6 ? 1.0e-6 : (h > 1.0e32 ? 1.0e32 : h));
      sc.H[tid * Dim + tid] = static_cast<T>(damped);
    }
    __syncthreads();

    dense_ldlt_solve<Dim, T>(sc.H, sc.b, sc.delta, &sc.solve_ok);
    __syncthreads();

    const bool solve_ok = sc.solve_ok != 0;
    if (tid == 0 && solve_ok) {
      for (int i = 0; i < Dim; ++i) {
        sc.dx[i] = sc.delta[i] * sc.scale[i];
      }
      backup = state;
      problem.update(state, sc.dx);
    }
    __syncthreads();

    if (solve_ok) {
      fused_chi2<Dim, T>(problem, state, sc);
    } else if (tid == 0) {
      sc.new_chi2 = T(0);
    }
    __syncthreads();

    if (tid == 0) {
      T new_chi2 = solve_ok ? sc.new_chi2 : lm_huge<T>();

      // rho = actual reduction / predicted reduction, with graphite's
      // 1e-3 floor on the denominator.
      T denom = T(1);
      if (solve_ok) {
        denom = T(0);
        for (int i = 0; i < Dim; ++i) {
          const T d = sc.delta[i];
          denom += d * (sc.mu * d + sc.b[i]);
        }
        denom += T(1.0e-3);
      }
      const T rho = (initial_chi2 - new_chi2) / denom;

      bool accepted = solve_ok && isfinite(new_chi2) && rho > T(0);
      if (accepted) {
        double alpha = 1.0 - pow(2.0 * static_cast<double>(rho) - 1.0, 3.0);
        alpha = alpha < 1.0 / 3.0 ? 1.0 / 3.0 : (alpha > 2.0 / 3.0 ? 2.0 / 3.0 : alpha);
        sc.mu *= static_cast<T>(alpha);
        sc.nu = T(2);
      } else {
        if (solve_ok) {
          state = backup;
        }
        sc.mu *= sc.nu;
        sc.nu *= T(2);
        new_chi2 = initial_chi2;
      }

      sc.chi2 = new_chi2;
      sc.step_accepted = accepted;

      if (!isfinite(sc.mu) || rho == T(0)) {
        sc.run = false;
      }
      if (accepted) {
        if ((initial_chi2 - new_chi2) * T(1.0e3) < initial_chi2) {
          sc.num_bad++;
        } else {
          sc.num_bad = 0;
        }
        if (sc.num_bad >= opt.max_negligible_steps) {
          sc.run = false;
        }
      }
    }
    __syncthreads();

    if (!sc.run) {
      break;
    }
    if (sc.step_accepted && it + 1 < opt.iterations) {
      fused_linearize<Dim, T>(problem, state, sc);
    }
  }
  __syncthreads();
}

} // namespace fused
} // namespace ORB_SLAM3
