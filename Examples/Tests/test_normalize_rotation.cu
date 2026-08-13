/**
 * Validation for gpu::NormalizeRotation (include/GPUPose.h).
 *
 * Reference is the CPU implementation in G2oTypes.h:
 *     Eigen::JacobiSVD<Matrix3> svd(R, ComputeFullU | ComputeFullV);
 *     return svd.matrixU() * svd.matrixV().transpose();
 *
 * Checks, over a batch of inputs ranging from exact rotations to badly
 * non-orthogonal / rank-deficient matrices:
 *   1. host gpu::NormalizeRotation  vs  Eigen JacobiSVD  (double and float)
 *   2. device gpu::NormalizeRotation vs the host result (same code path)
 *   3. orthogonality ||X^T X - I|| and det(X) of the output
 *   4. ExpSO3 / LogSO3 round trip, which is the main consumer
 *   5. the superseded single-step polar approximation, for reference
 *
 * Exit code is non-zero if any check exceeds its tolerance.
 */

#include <graphite/common.hpp>
#include <graphite/types.hpp>
#include <graphite/vertex.hpp>

#include "GPUPose.h"

#include <Eigen/Dense>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                               \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                        __FILE__, __LINE__);                                      \
            return 2;                                                             \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// The old implementation, kept here only as a baseline to compare against.
// ---------------------------------------------------------------------------
template <typename T>
hd_fn Eigen::Matrix<T, 3, 3> PolarStepReference(const Eigen::Matrix<T, 3, 3>& R) {
    const Eigen::Matrix<T, 3, 3> RTR = R.transpose() * R;
    const Eigen::Matrix<T, 3, 3> inv_sqrt =
        Eigen::Matrix<T, 3, 3>::Identity() * T(1.5) - RTR * T(0.5);
    Eigen::Matrix<T, 3, 3> result = R * inv_sqrt;
    if (result.determinant() < T(0)) result.col(2) = -result.col(2);
    return result;
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------
template <typename T>
__global__ void NormalizeKernel(const T* in, T* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Eigen::Matrix<T, 3, 3> R = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(in + 9 * i);
    Eigen::Map<Eigen::Matrix<T, 3, 3>>(out + 9 * i) = gpu::NormalizeRotation(R);
}

template <typename T>
__global__ void PolarKernel(const T* in, T* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Eigen::Matrix<T, 3, 3> R = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(in + 9 * i);
    Eigen::Map<Eigen::Matrix<T, 3, 3>>(out + 9 * i) = PolarStepReference(R);
}

// ---------------------------------------------------------------------------
// Test inputs
// ---------------------------------------------------------------------------
struct Sample {
    Eigen::Matrix3d R;
    std::string     group;
    bool            singular = false;  // U*V^T is not unique -> skip SVD compare
};

static Eigen::Matrix3d RandomRotation(std::mt19937& rng) {
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> u(-M_PI, M_PI);
    Eigen::Vector3d axis(n(rng), n(rng), n(rng));
    if (axis.norm() < 1e-9) axis = Eigen::Vector3d::UnitZ();
    axis.normalize();
    return Eigen::AngleAxisd(u(rng), axis).toRotationMatrix();
}

static std::vector<Sample> BuildSamples() {
    std::mt19937 rng(20260813);
    std::normal_distribution<double> n(0.0, 1.0);
    std::vector<Sample> s;

    // Exact rotations: must come back unchanged.
    for (int i = 0; i < 200; ++i) s.push_back({RandomRotation(rng), "exact rotation"});

    // Rotations perturbed by increasing amounts of noise. The largest ones are
    // where the single polar step falls apart.
    const double sigmas[] = {1e-12, 1e-8, 1e-4, 1e-2, 1e-1, 5e-1, 1.0};
    for (double sigma : sigmas) {
        for (int i = 0; i < 100; ++i) {
            Eigen::Matrix3d R = RandomRotation(rng);
            Eigen::Matrix3d N;
            for (int k = 0; k < 9; ++k) N(k) = n(rng);
            s.push_back({R + sigma * N, "rotation + noise " + std::to_string(sigma)});
        }
    }

    // Anisotropically scaled rotations: singular values far from 1.
    for (int i = 0; i < 100; ++i) {
        Eigen::Vector3d d(std::pow(10.0, n(rng)), std::pow(10.0, n(rng)), std::pow(10.0, n(rng)));
        s.push_back({RandomRotation(rng) * d.asDiagonal() * RandomRotation(rng), "scaled rotation"});
    }

    // Fully random matrices.
    for (int i = 0; i < 200; ++i) {
        Eigen::Matrix3d M;
        for (int k = 0; k < 9; ++k) M(k) = n(rng);
        s.push_back({M, "random matrix"});
    }

    // Reflections: det < 0. Both implementations must agree on the convention.
    for (int i = 0; i < 50; ++i) {
        Eigen::Matrix3d R = RandomRotation(rng);
        R.col(0) = -R.col(0);
        s.push_back({R, "reflection"});
    }

    // Near-degenerate: repeated singular values (U,V not unique but U*V^T is).
    for (int i = 0; i < 50; ++i) {
        Eigen::Vector3d d(1.0, 1.0, 1.0 + 1e-9 * n(rng));
        s.push_back({RandomRotation(rng) * d.asDiagonal() * RandomRotation(rng), "repeated sv"});
    }

    // Structured / edge cases.
    s.push_back({Eigen::Matrix3d::Identity(), "identity"});
    s.push_back({-Eigen::Matrix3d::Identity(), "-identity"});
    s.push_back({Eigen::Matrix3d::Identity() * 1e-8, "tiny identity"});
    s.push_back({Eigen::Matrix3d::Identity() * 1e8, "huge identity"});

    // Rank deficient: the polar factor is not unique, so only the structural
    // checks (orthogonality, finiteness) apply.
    {
        Eigen::Matrix3d M = RandomRotation(rng);
        M.col(2).setZero();
        s.push_back({M, "rank 2", true});

        Eigen::Matrix3d M2 = RandomRotation(rng);
        M2.col(1) = M2.col(0);
        s.push_back({M2, "rank 2 (dup col)", true});

        Eigen::Matrix3d M3 = Eigen::Matrix3d::Zero();
        M3.col(0) = Eigen::Vector3d(1.0, 2.0, 3.0);
        s.push_back({M3, "rank 1", true});

        s.push_back({Eigen::Matrix3d::Zero(), "zero", true});
    }

    // The DR mask applied in ImuCamPose::UpdateW before normalizing.
    for (int i = 0; i < 50; ++i) {
        Eigen::Matrix3d R = RandomRotation(rng);
        R(0, 2) = 0.0; R(1, 2) = 0.0; R(2, 0) = 0.0; R(2, 1) = 0.0;
        s.push_back({R, "UpdateW masked DR"});
    }

    return s;
}

// ---------------------------------------------------------------------------
// Per-group error accumulation
// ---------------------------------------------------------------------------
struct Stat {
    std::string group;
    int    count      = 0;
    double svd_err    = 0.0;   // vs Eigen JacobiSVD (skipped for singular)
    double polar_err  = 0.0;   // old implementation, vs Eigen JacobiSVD
    double orth_err   = 0.0;   // ||X^T X - I||_inf
    double det_err    = 0.0;   // ||det(X)| - 1|
    double device_err = 0.0;   // device vs host, same code
    double float_err  = 0.0;   // float: orthogonality and device-vs-host
};

static Stat& GroupOf(std::vector<Stat>& stats, const std::string& g) {
    for (Stat& s : stats)
        if (s.group == g) return s;
    stats.push_back(Stat{g});
    return stats.back();
}

int main() {
    const std::vector<Sample> samples = BuildSamples();
    const int n = static_cast<int>(samples.size());
    std::printf("gpu::NormalizeRotation validation over %d matrices\n\n", n);

    // Flatten inputs (Eigen is column-major, matching Eigen::Map on device).
    std::vector<double> h_in(9 * n);
    std::vector<float>  h_in_f(9 * n);
    for (int i = 0; i < n; ++i) {
        Eigen::Map<Eigen::Matrix3d>(h_in.data() + 9 * i)  = samples[i].R;
        Eigen::Map<Eigen::Matrix3f>(h_in_f.data() + 9 * i) = samples[i].R.cast<float>();
    }

    // --- device pass -------------------------------------------------------
    double *d_in = nullptr, *d_out = nullptr, *d_polar = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, 9 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, 9 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_polar, 9 * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), 9 * n * sizeof(double), cudaMemcpyHostToDevice));

    const int block = 128, grid = (n + block - 1) / block;
    NormalizeKernel<double><<<grid, block>>>(d_in, d_out, n);
    PolarKernel<double><<<grid, block>>>(d_in, d_polar, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_dev(9 * n);
    CUDA_CHECK(cudaMemcpy(h_dev.data(), d_out, 9 * n * sizeof(double), cudaMemcpyDeviceToHost));

    float *d_in_f = nullptr, *d_out_f = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in_f, 9 * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_f, 9 * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in_f, h_in_f.data(), 9 * n * sizeof(float), cudaMemcpyHostToDevice));
    NormalizeKernel<float><<<grid, block>>>(d_in_f, d_out_f, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_dev_f(9 * n);
    CUDA_CHECK(cudaMemcpy(h_dev_f.data(), d_out_f, 9 * n * sizeof(float), cudaMemcpyDeviceToHost));

    // --- host pass + comparison -------------------------------------------
    std::vector<Stat> stats;
    double worst_float = 0.0;
    bool   nonfinite   = false;

    for (int i = 0; i < n; ++i) {
        const Eigen::Matrix3d& R = samples[i].R;
        Stat& st = GroupOf(stats, samples[i].group);
        st.count++;

        const Eigen::Matrix3d gpu_host = gpu::NormalizeRotation(R);
        const Eigen::Matrix3d gpu_dev  = Eigen::Map<Eigen::Matrix3d>(h_dev.data() + 9 * i);
        const Eigen::Matrix3d polar    = PolarStepReference(R);

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix3d ref = svd.matrixU() * svd.matrixV().transpose();

        if (!gpu_host.allFinite() || !gpu_dev.allFinite()) nonfinite = true;

        if (!samples[i].singular) {
            st.svd_err   = std::max(st.svd_err, (gpu_host - ref).cwiseAbs().maxCoeff());
            st.polar_err = std::max(st.polar_err, (polar - ref).cwiseAbs().maxCoeff());
        }
        st.device_err = std::max(st.device_err, (gpu_host - gpu_dev).cwiseAbs().maxCoeff());
        st.orth_err   = std::max(st.orth_err,
                                 (gpu_host.transpose() * gpu_host - Eigen::Matrix3d::Identity())
                                     .cwiseAbs().maxCoeff());
        st.det_err    = std::max(st.det_err, std::abs(std::abs(gpu_host.determinant()) - 1.0));

        // float path: check the result is orthogonal in single precision too
        const Eigen::Matrix3f fdev = Eigen::Map<Eigen::Matrix3f>(h_dev_f.data() + 9 * i);
        const Eigen::Matrix3f fin(samples[i].R.cast<float>());
        const Eigen::Matrix3f fhost = gpu::NormalizeRotation(fin);
        st.float_err = std::max<double>(
            st.float_err,
            (fhost.transpose() * fhost - Eigen::Matrix3f::Identity()).cwiseAbs().maxCoeff());
        st.float_err = std::max<double>(st.float_err, (fhost - fdev).cwiseAbs().maxCoeff());
        worst_float = std::max(worst_float, st.float_err);
    }

    std::printf("%-28s %6s %11s %11s %11s %11s %11s %11s\n", "group", "n", "vs SVD",
                "old polar", "orth err", "det err", "dev-host", "float");
    std::printf("%s\n", std::string(108, '-').c_str());
    double worst_svd = 0.0, worst_orth = 0.0, worst_det = 0.0, worst_dev = 0.0;
    for (const Stat& s : stats) {
        std::printf("%-28s %6d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e\n", s.group.c_str(),
                    s.count, s.svd_err, s.polar_err, s.orth_err, s.det_err, s.device_err,
                    s.float_err);
        worst_svd  = std::max(worst_svd, s.svd_err);
        worst_orth = std::max(worst_orth, s.orth_err);
        worst_det  = std::max(worst_det, s.det_err);
        worst_dev  = std::max(worst_dev, s.device_err);
    }

    // --- ExpSO3 / LogSO3 round trip ---------------------------------------
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    double worst_exp = 0.0;
    for (int i = 0; i < 5000; ++i) {
        Eigen::Vector3d w(nd(rng), nd(rng), nd(rng));
        w.normalize();
        w *= (i % 2 == 0) ? 1e-7 * (i + 1) : 3.0 * double(i) / 5000.0;  // tiny .. ~pi
        const Eigen::Matrix3d Rw = gpu::ExpSO3(w[0], w[1], w[2]);
        const Eigen::Vector3d back = gpu::LogSO3(Rw);
        worst_exp = std::max(worst_exp, (back - w).cwiseAbs().maxCoeff());
        worst_orth = std::max(worst_orth,
                              (Rw.transpose() * Rw - Eigen::Matrix3d::Identity())
                                  .cwiseAbs().maxCoeff());
    }

    // --- throughput --------------------------------------------------------
    {
        const int m = 1 << 20;
        std::vector<double> big(9 * m);
        std::mt19937 r2(3);
        for (int i = 0; i < m; ++i)
            Eigen::Map<Eigen::Matrix3d>(big.data() + 9 * i) = RandomRotation(r2);
        double *bd_in = nullptr, *bd_out = nullptr;
        CUDA_CHECK(cudaMalloc(&bd_in, 9 * m * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&bd_out, 9 * m * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(bd_in, big.data(), 9 * m * sizeof(double), cudaMemcpyHostToDevice));
        const int g = (m + block - 1) / block;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        float ms_svd = 0.f, ms_polar = 0.f;

        NormalizeKernel<double><<<g, block>>>(bd_in, bd_out, m);  // warm up
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(t0);
        for (int r = 0; r < 10; ++r) NormalizeKernel<double><<<g, block>>>(bd_in, bd_out, m);
        cudaEventRecord(t1);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&ms_svd, t0, t1);

        cudaEventRecord(t0);
        for (int r = 0; r < 10; ++r) PolarKernel<double><<<g, block>>>(bd_in, bd_out, m);
        cudaEventRecord(t1);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&ms_polar, t0, t1);

        std::printf("\nthroughput over %d matrices x10: jacobi svd %.2f ms, old polar %.2f ms\n",
                    m, ms_svd, ms_polar);
        CUDA_CHECK(cudaFree(bd_in));
        CUDA_CHECK(cudaFree(bd_out));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_polar));
    CUDA_CHECK(cudaFree(d_in_f));
    CUDA_CHECK(cudaFree(d_out_f));

    // --- verdict -----------------------------------------------------------
    struct Check { const char* name; double value; double tol; };
    const Check checks[] = {
        {"max |gpu - Eigen JacobiSVD| (double)", worst_svd,  1e-12},
        {"max ||X^T X - I|| (double)",           worst_orth, 1e-12},
        {"max ||det X| - 1| (double)",           worst_det,  1e-12},
        {"max |device - host| (double)",         worst_dev,  1e-12},
        {"float: orthogonality + dev/host",      worst_float, 2e-5},
        {"LogSO3(ExpSO3(w)) - w",                worst_exp,  1e-9},
    };

    std::printf("\n");
    bool ok = !nonfinite;
    if (nonfinite) std::printf("FAIL  non-finite output produced\n");
    for (const Check& c : checks) {
        const bool pass = (c.value <= c.tol);
        ok = ok && pass;
        std::printf("%-4s %-38s %11.3e  (tol %.1e)\n", pass ? "ok" : "FAIL", c.name, c.value, c.tol);
    }
    std::printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
