/// @file chi2.hpp
#pragma once
#include <graphite/ops/common.hpp>

namespace graphite {

namespace ops {

template <typename T, typename P, size_t E>
__device__ T compute_chi2(const T *residuals, const P *pmat,
                          const size_t factor_id) {
  T r2[E] = {0};

#pragma unroll
  for (int i = 0; i < E; i++) {
#pragma unroll
    for (int j = 0; j < E; j++) {
      r2[i] += static_cast<T>(pmat[factor_id * E * E + i * E + j]) *
               residuals[factor_id * E + j];
    }
  }

  T value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) {
    value += r2[i] * residuals[factor_id * E + i];
  }

  return value;
}

template <typename T, typename S, size_t E, typename L>
__global__ void
compute_chi2_kernel(T *chi2, S *chi2_derivative, const T *residuals,
                    const size_t num_threads, const S *pmat, const L *loss) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }
  T raw_chi2 = compute_chi2<T, S, E>(residuals, pmat, idx);
  chi2[idx] = loss[idx].loss(raw_chi2);
  chi2_derivative[idx] = loss[idx].loss_derivative(raw_chi2);
}

template <typename T, typename S, typename F> void compute_chi2_async(F *f) {
  // Then for each vertex, we need to compute the error
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // At this point all necessary data should be on the GPU
  auto verts = f->get_vertices();

  constexpr auto error_dim = F::error_dim;
  const auto num_factors = f->active_count();

  const auto num_threads = num_factors;
  size_t threads_per_block = 256;
  size_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  thrust::fill(thrust::cuda::par_nosync.on(0), f->chi2_vec.begin(),
               f->chi2_vec.end(), static_cast<T>(0));
  compute_chi2_kernel<T, S, F::error_dim>
      <<<num_blocks, threads_per_block, 0, 0>>>(
          f->chi2_vec.data().get(), f->chi2_derivative.data().get(),
          f->residuals.data().get(), num_threads,
          f->precision_matrices.data().get(), f->loss.data().get());

  // cudaStreamSynchronize(0);
}

} // namespace ops

} // namespace graphite