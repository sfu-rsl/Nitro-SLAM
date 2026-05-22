/// @file hessian.hpp
#pragma once
#include <graphite/ops/common.hpp>

namespace graphite {

namespace ops {

template <typename T, typename S, size_t N, size_t E>
__global__ void compute_hessian_block_kernel(
    const size_t vi, const size_t vj, size_t dim_i, size_t dim_j,
    const size_t *active_factors, const size_t num_active_factors,
    const size_t *ids, const size_t *block_offsets, const uint8_t *vi_active,
    const uint8_t *vj_active, const size_t *hessian_offset_i,
    const size_t *hessian_offset_j, const S *jacobian_i, const S *jacobian_j,
    const S *precision, const S *chi2_derivative, S *hessian) {
  // TODO: simpify and optimize this kernel
  const auto idx = get_thread_id();

  const auto block_id = idx / (dim_i * dim_j);

  if (block_id >= num_active_factors) {
    return;
  }

  const auto factor_idx = active_factors[block_id];

  const size_t vi_id = ids[factor_idx * N + vi];
  const size_t vj_id = ids[factor_idx * N + vj];

  if (is_vertex_active(vi_active, vi_id) &&
      is_vertex_active(vj_active, vj_id)) {

    const size_t block_size = dim_i * dim_j;
    const size_t offset = idx % block_size;
    // Hessian block may be rectangular
    // output blocks are all column major

    const bool transposed = hessian_offset_i[vi_id] > hessian_offset_j[vj_id];

    auto ji = factor_idx * E * dim_i + jacobian_i;
    auto jj = factor_idx * E * dim_j + jacobian_j;
    const auto precision_offset = factor_idx * E * E;
    const auto p = precision + precision_offset;

    if (transposed) {
      cuda::std::swap(ji, jj);
      cuda::std::swap(dim_i, dim_j);
    }

    const size_t row = offset % dim_i;
    const size_t col = offset / dim_i;

    // Each thread computes one element of the Hessian block
    using highp = T;
    highp value = 0;

    // computes J_i^T * P * J_j

    const auto J = jj + col * E;
    const auto Jt = ji + row * E;
#pragma unroll
    for (int i = 0; i < E; i++) { // p row
      highp pj = 0;
#pragma unroll
      for (int j = 0; j < E; j++) { // p col
        pj += (highp)p[i * E + j] * (highp)J[j];
      }
      value += (highp)Jt[i] * pj;
    }

    value *= (highp)chi2_derivative[factor_idx];

    auto block = hessian + (block_offsets[block_id] + (row + col * dim_i));
    S lp_value = static_cast<S>(value);
    atomicAdd(block, lp_value);
  }
}

template <typename S, int D>
__global__ void augment_hessian_diagonal_kernel(S *diagonal_blocks,
                                                S *scalar_diagonal, const S mu,
                                                const uint8_t *active_state,
                                                const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto block_size = D * D;

  const auto vertex_id = idx;
  if (!is_vertex_active(active_state, vertex_id)) {
    return;
  }

  S *block = diagonal_blocks + vertex_id * block_size;
  for (size_t i = 0; i < D; i++) {

    // const double diag = static_cast<double>(block[i * D + i]);
    const double diag = static_cast<double>(scalar_diagonal[vertex_id * D + i]);
    const double new_diag =
        diag + static_cast<double>(mu) * std::clamp(diag, 1.0e-6, 1.0e32);
    // const double new_diag =
    //     diag + static_cast<double>(mu) * diag;
    block[i * D + i] = static_cast<S>(new_diag);
  }
}

template <typename T, typename S, typename V>
void augment_block_diagonal(V *v, InvP<T, S> *block_diagonal,
                            InvP<T, S> *scalar_diagonal, T mu,
                            cudaStream_t stream) {
  const size_t num_threads = v->count();
  const auto threads_per_block = 256;
  const auto num_blocks =
      (num_threads + threads_per_block - 1) / threads_per_block;

  augment_hessian_diagonal_kernel<InvP<T, S>, V::dim>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          block_diagonal, scalar_diagonal, (InvP<T, S>)mu,
          v->get_active_state(), num_threads);
}

template <typename T, typename S, int D>
__global__ void apply_block_jacobi_kernel(T *z, const T *r, S *block_diagonal,
                                          const size_t *hessian_ids,
                                          const uint8_t *active_state,
                                          const size_t num_threads) {
  const size_t idx = get_thread_id();
  const auto local_vertex_id = idx / D;

  if (idx >= num_threads || !is_vertex_active(active_state, local_vertex_id)) {
    return;
  }

  constexpr auto block_size = D * D;

  S *block = block_diagonal + local_vertex_id * block_size;
  const auto hessian_offset = hessian_ids[local_vertex_id];
  const auto offset = idx % D;
  const auto row = offset;

  T value = 0;
#pragma unroll
  for (size_t i = 0; i < D; i++) {
    value += (T)block[row + i * D] * r[hessian_offset + i];
  }
  z[hessian_offset + row] = value;
}

template <typename T, typename S, typename V>
void apply_block_jacobi(V *v, T *z, const T *r, InvP<T, S> *block_diagonal,
                        cudaStream_t stream) {
  const size_t num_parameters = v->count() * v->dimension();
  const size_t num_threads = num_parameters;
  const auto threads_per_block = 256;
  const auto num_blocks =
      (num_threads + threads_per_block - 1) / threads_per_block;

  apply_block_jacobi_kernel<T, InvP<T, S>, V::dim>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          z, r, block_diagonal, v->get_hessian_ids(), v->get_active_state(),
          num_threads);
}

template <typename highp, typename InvP, typename T, size_t I, size_t N,
          size_t E, size_t D>
__global__ void compute_hessian_diagonal_kernel(
    InvP *diagonal_blocks, const T *jacs, const size_t *active_ids,
    const size_t *ids, const uint8_t *active_state, const T *pmat,
    const T *chi2_derivative, const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;
  constexpr size_t block_size = D * D;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / block_size];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)
  // const size_t row = idx % D;
  // const size_t col = idx / D;

  const size_t offset = idx % block_size;
  const size_t row = offset % D;
  const size_t col = offset / D;

  // left[i]*pmat[i*E+j]*right[i] = h value
  // where i goes from 0 to E
  const T *Jt = jacs + jacobian_offset + row * E;
  const T *J = jacs + jacobian_offset + col * E;

  const T *precision_matrix = pmat + precision_offset;

  highp value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    highp pj = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      pj += (highp)precision_matrix[i * E + j] * (highp)J[j];
    }
    value += (highp)Jt[i] * pj;
    // value += Jt[i]*J[i];
  }

  // if (row == col) {
  //     value = 1;
  //     // printf("Thread %d, row: %d, col: %d, value: %f\n", idx, row, col,
  //     value);
  //     // printf("D=%d\n", D);
  // }
  // else {
  //     value = 0;
  // }
  value *= (highp)chi2_derivative[factor_id];

  // T* block = diagonal_blocks + local_id*block_size + (idx % block_size);
  // printf("Thread %d, Hessian offset: %u\n", idx, local_id);
  // if (value != 0) {
  //  printf("Thread %d, row: %d, col: %d, value: %f\n", idx, row, col, value);
  // }
  // if (D == 9 && local_id*block_size + row + col*D > 1701) {
  //     printf("Thread %d, vertex id: %u, row: %d, col: %d, value: %f, offset:
  //     %u \n", idx, local_id, row, col, value, local_id*block_size + row +
  //     col*D);

  // }
  // value = 5.0;
  // T* block = diagonal_blocks + local_id*block_size + row + col*D;
  InvP *block = diagonal_blocks + (local_id * block_size + row + col * D);
  // if (row == col) {
  //     // local_id = 5;
  //     // row = 6; col = 7;
  //     // value = 8.5;
  //     printf("Thread %d, vertex id: %llu, row: %llu, col: %llu, value: %f,
  //     offset: %u \n", idx, local_id, row, col, value, local_id*block_size +
  //     row + col*D);
  // }
  // if (row == 8 && col == 8) {
  //     printf("Number of threads: %d\n", num_threads);
  //     printf("Thread %d, vertex id: %u, row: %zu, col: %zu, value: %f\n",
  //     idx, local_id, row, col, value);
  // }
  // T* block = diagonal_blocks + row*D + col;
  InvP lp_value = static_cast<InvP>(value);
  atomicAdd(block, lp_value);
  // block[0] = value;
  // *block = value;
}

template <typename highp, typename InvP, typename T, size_t I, size_t N,
          size_t E, size_t D, typename VT, typename F>
__global__ void compute_hessian_diagonal_dynamic_kernel(
    InvP *diagonal_blocks, const size_t *active_ids, const size_t *ids,
    const size_t *hessian_ids, const VT args,
    const typename F::ObservationType *obs, const highp *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data,
    const uint8_t *active_state, const T *pmat, const T *chi2_derivative,
    const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;
  constexpr size_t block_size = D * D;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / block_size];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)

  const size_t offset = idx % block_size;
  const size_t row = offset % D;
  const size_t col = offset / D;

  highp jacobian[jacobian_size];

  compute_Jblock<highp, I, N, typename F::ObservationType, E, F, VT>(
      jacobian, factor_id, local_id, obs, constraint_data, ids, hessian_ids,
      args, std::make_index_sequence<N>{});

  const auto hessian_offset = hessian_ids[local_id];

  const highp *Jt = jacobian + row * E;
  const highp *J = jacobian + col * E;

  const T *precision_matrix = pmat + precision_offset;

  const highp *jscale = jacobian_scales + hessian_offset;

  highp value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    highp pj = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      pj += (highp)precision_matrix[i * E + j] * (highp)J[j] * jscale[col];
    }
    value += (highp)Jt[i] * jscale[row] * pj;
  }

  value *= (highp)chi2_derivative[factor_id];

  InvP *block = diagonal_blocks + (local_id * block_size + row + col * D);

  InvP lp_value = static_cast<InvP>(value);
  atomicAdd(block, lp_value);
}

template <typename T, typename S, typename F, std::size_t... Is>
void launch_kernel_block_diagonal(
    F *f, std::array<InvP<T, S> *, F::get_num_vertices()> &diagonal_blocks,
    std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
    const size_t num_factors, cudaStream_t stream, std::index_sequence<Is...>) {
  (([&] {
     constexpr size_t num_vertices = F::get_num_vertices();
     constexpr size_t dimension = F::get_vertex_sizes()[Is];
     const size_t num_threads = num_factors * dimension * dimension;
     // std::cout << "Launching block diagonal kernel" << std::endl;
     // std::cout << "Num threads: " << num_threads << std::endl;
     // std::cout << "dimension: " << dimension << std::endl;
     // std::cout << "num_factors: " << num_factors << std::endl;
     size_t threads_per_block = 256;
     size_t num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
     // std::endl; std::cout << "Checking residual ptr: " <<
     // f->residuals.data().get() << std::endl; std::cout << "Checking ids
     // ptr: " << f->device_ids.data().get() << std::endl;

     if (f->store_jacobians() || !is_analytical<F>()) {
       compute_hessian_diagonal_kernel<T, InvP<T, S>, S, Is, num_vertices,
                                       F::error_dim, dimension>
           <<<num_blocks, threads_per_block, 0, stream>>>(
               diagonal_blocks[Is], jacs[Is], f->active_indices.data().get(),
               f->device_ids.data().get(),
               f->vertex_descriptors[Is]->get_active_state(),
               f->precision_matrices.data().get(),
               f->chi2_derivative.data().get(), num_threads);
     } else {
       if constexpr (is_analytical<F>()) {
         compute_hessian_diagonal_dynamic_kernel<
             T, InvP<T, S>, S, Is, num_vertices, F::error_dim, dimension,
             typename F::VertexPointerPointerTuple, F>
             <<<num_blocks, threads_per_block, 0, stream>>>(
                 diagonal_blocks[Is], f->active_indices.data().get(),
                 f->device_ids.data().get(), hessian_ids[Is], f->get_vertices(),
                 f->device_obs.data().get(), jacobian_scales,
                 f->data.data().get(),
                 f->vertex_descriptors[Is]->get_active_state(),
                 f->precision_matrices.data().get(),
                 f->chi2_derivative.data().get(), num_threads);
       }
     }
   }()),
   ...);
}

template <typename T, typename S, typename F>
void compute_block_diagonal(
    F *f, std::array<InvP<T, S> *, F::get_num_vertices()> &diagonal_blocks,
    const T *jacobian_scales, cudaStream_t stream) {

  // Then for each vertex, we need to compute the error
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // At this point all necessary data should be on the GPU
  // std::array<T*, num_vertices> verts;
  auto verts = f->get_vertices();
  std::array<S *, num_vertices> jacs;
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    jacs[i] = f->jacobians[i].data.data().get();
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
  }

  const auto num_factors = f->active_count();

  launch_kernel_block_diagonal<T, S>(f, diagonal_blocks, hessian_ids, jacs,
                                     jacobian_scales, num_factors, stream,
                                     std::make_index_sequence<num_vertices>{});
}

template <typename highp, typename T, size_t I, size_t N, size_t E, size_t D>
__global__ void compute_hessian_scalar_diagonal_kernel(
    highp *diagonal, const T *jacs, const size_t *active_ids, const size_t *ids,
    const size_t *hessian_ids, const uint8_t *active_state, const T *pmat,
    const T *chi2_derivative, const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / D];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)
  const size_t row = idx % D;
  const size_t col = row;

  // left[i]*pmat[i*E+j]*right[i] = h value
  // where i goes from 0 to E
  const T *Jt = jacs + jacobian_offset + row * E;
  const T *J = jacs + jacobian_offset + col * E;

  const T *precision_matrix = pmat + precision_offset;

  highp value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    highp pj = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      pj += (highp)precision_matrix[i * E + j] * (highp)J[j];
    }
    value += (highp)Jt[i] * pj;
  }

  value *= (highp)chi2_derivative[factor_id];

  // T* block = diagonal_blocks+(local_id*block_size + row + col*D);
  const size_t hessian_offset = hessian_ids[local_id];
  highp *location = diagonal + hessian_offset + row;
  // T lp_value = static_cast<T>(value);
  atomicAdd(location, value);
}

template <typename highp, typename T, size_t I, size_t N, size_t E, size_t D,
          typename VT, typename F, bool use_scales>
__global__ void compute_hessian_scalar_diagonal_dynamic_kernel(
    highp *diagonal, const T *jacs, const size_t *active_ids, const size_t *ids,
    const size_t *hessian_ids, const VT args,
    const typename F::ObservationType *obs, const highp *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data,
    const uint8_t *active_state, const T *pmat, const T *chi2_derivative,
    const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / D];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)
  const size_t row = idx % D;
  const size_t col = row;

  // using G = std::conditional_t<is_low_precision<T>::value, highp, T>;
  // G jacobian[jacobian_size];
  highp jacobian[jacobian_size];

  compute_Jblock<highp, I, N, typename F::ObservationType, E, F, VT>(
      jacobian, factor_id, local_id, obs, constraint_data, ids, hessian_ids,
      args, std::make_index_sequence<N>{});
  const auto hessian_offset = hessian_ids[local_id];
  const highp *Jt = jacobian + row * E;
  const highp *J = jacobian + col * E;

  const T *precision_matrix = pmat + precision_offset;
  highp value = 0;
  if constexpr (use_scales) {
    const highp *jscale = jacobian_scales + hessian_offset;

#pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
      highp pj = 0;
#pragma unroll
      for (int j = 0; j < E; j++) { // pmat col
        pj += (highp)precision_matrix[i * E + j] * (highp)J[j] * jscale[col];
      }
      value += (highp)Jt[i] * jscale[row] * pj;
    }
  } else {

#pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
      highp pj = 0;
#pragma unroll
      for (int j = 0; j < E; j++) { // pmat col
        pj += (highp)precision_matrix[i * E + j] * (highp)J[j];
      }
      value += (highp)Jt[i] * pj;
    }
  }

  value *= (highp)chi2_derivative[factor_id];

  // T* block = diagonal_blocks+(local_id*block_size + row + col*D);
  highp *location = diagonal + hessian_offset + row;
  // T lp_value = static_cast<T>(value);
  atomicAdd(location, value);
}

template <typename T, typename S, typename F, std::size_t... Is>
void launch_kernel_hessian_scalar_diagonal(
    F *f, T *diagonal,
    std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
    const size_t num_factors, std::index_sequence<Is...>) {
  (([&] {
     constexpr size_t num_vertices = F::get_num_vertices();
     constexpr size_t dimension = F::get_vertex_sizes()[Is];
     const size_t num_threads = num_factors * dimension;

     size_t threads_per_block = 256;
     size_t num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     if (f->store_jacobians() || !is_analytical<F>()) {
       compute_hessian_scalar_diagonal_kernel<T, S, Is, num_vertices,
                                              F::error_dim, dimension>
           <<<num_blocks, threads_per_block>>>(
               diagonal, jacs[Is], f->active_indices.data().get(),
               f->device_ids.data().get(), hessian_ids[Is],
               f->vertex_descriptors[Is]->get_active_state(),
               f->precision_matrices.data().get(),
               f->chi2_derivative.data().get(), num_threads);
     } else {
       if constexpr (is_analytical<F>()) {
         if (jacobian_scales == nullptr) {
           compute_hessian_scalar_diagonal_dynamic_kernel<
               T, S, Is, num_vertices, F::error_dim, dimension,
               typename F::VertexPointerPointerTuple, F, false>
               <<<num_blocks, threads_per_block>>>(
                   diagonal, jacs[Is], f->active_indices.data().get(),
                   f->device_ids.data().get(), hessian_ids[Is],
                   f->get_vertices(), f->device_obs.data().get(), nullptr,
                   f->data.data().get(),
                   f->vertex_descriptors[Is]->get_active_state(),
                   f->precision_matrices.data().get(),
                   f->chi2_derivative.data().get(), num_threads);
         } else {
           compute_hessian_scalar_diagonal_dynamic_kernel<
               T, S, Is, num_vertices, F::error_dim, dimension,
               typename F::VertexPointerPointerTuple, F, true>
               <<<num_blocks, threads_per_block>>>(
                   diagonal, jacs[Is], f->active_indices.data().get(),
                   f->device_ids.data().get(), hessian_ids[Is],
                   f->get_vertices(), f->device_obs.data().get(),
                   jacobian_scales, f->data.data().get(),
                   f->vertex_descriptors[Is]->get_active_state(),
                   f->precision_matrices.data().get(),
                   f->chi2_derivative.data().get(), num_threads);
         }
       }
     }
   }()),
   ...);
}

template <typename T, typename S, typename F>
void compute_hessian_scalar_diagonal(F *f, T *diagonal,
                                     const T *jacobian_scales) {

  // Then for each vertex, we need to compute the error
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // At this point all necessary data should be on the GPU
  // std::array<T*, num_vertices> verts;
  auto verts = f->get_vertices();
  std::array<S *, num_vertices> jacs;
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    jacs[i] = f->jacobians[i].data.data().get();
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
  }

  const auto num_factors = f->active_count();

  launch_kernel_hessian_scalar_diagonal<T, S>(
      f, diagonal, hessian_ids, jacs, jacobian_scales, num_factors,
      std::make_index_sequence<num_vertices>{});
}

} // namespace ops

} // namespace graphite