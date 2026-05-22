/// @file linearize.hpp
#pragma once
#include <graphite/ops/common.hpp>

namespace graphite {
namespace ops {

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void compute_jacobian_kernel(
    const M *obs, T *error, S *jacs,
    const typename F::ConstraintDataType *constraint_data,
    const size_t *active_ids, const size_t *ids, const size_t num_threads,
    const VT args, const uint8_t *active_state, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  const auto factor_id = active_ids[idx];
  const size_t vertex_id = ids[factor_id * N + I];
  if (!is_vertex_active(active_state, vertex_id)) {
    return;
  }

  const M *local_obs = obs + factor_id;
  // T *local_error = error + factor_id * E;
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto vargs = cuda::std::make_tuple(
      (*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  constexpr size_t jacobian_size = E * vertex_sizes[I];

  S *j = jacs + factor_id * jacobian_size;
  using DataType = typename F::ConstraintDataType;
  using ObsType = typename F::ObservationType;

  // if constexpr (is_low_precision<S>::value) {
  if constexpr (!std::is_same<T, S>::value) {
    T jacobian[jacobian_size];

    if constexpr (std::is_same<Empty, ObsType>::value &&
                  std::is_same<Empty, DataType>::value) {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                         jacobian);
    } else if constexpr (std::is_same<Empty, DataType>::value) {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                         *local_obs, jacobian);
    } else if constexpr (std::is_same<Empty, ObsType>::value) {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                         *local_data, jacobian);
    } else {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                         *local_obs, *local_data, jacobian);
    }

#pragma unroll
    for (size_t i = 0; i < jacobian_size; ++i) {
      j[i] = jacobian[i];
    }
  } else {
    if constexpr (std::is_same<Empty, ObsType>::value &&
                  std::is_same<Empty, DataType>::value) {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))..., j);
    } else if constexpr (std::is_same<Empty, DataType>::value) {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                         *local_obs, j);
    } else if constexpr (std::is_same<Empty, ObsType>::value) {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                         *local_data, j);
    } else {
      F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                         *local_obs, *local_data, j);
    }
  }
}

template <typename T, typename S, typename F, typename VT, std::size_t... Is>
void launch_kernel_jacobians(
    F *f, std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    VT &verts, std::array<S *, F::get_num_vertices()> &jacs,
    const size_t num_factors, StreamPool &streams, std::index_sequence<Is...>) {
  (([&] {
     constexpr auto num_vertices = F::get_num_vertices();
     const auto num_threads = num_factors;
     size_t threads_per_block = 256;
     size_t num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     compute_jacobian_kernel<T, S, Is, num_vertices,
                             typename F::ObservationType, F::error_dim, F,
                             typename F::VertexPointerPointerTuple>
         <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
             f->device_obs.data().get(), f->residuals.data().get(), jacs[Is],
             f->data.data().get(), f->active_indices.data().get(),
             f->device_ids.data().get(), num_threads, verts,
             f->vertex_descriptors[Is]->get_active_state(),
             std::make_index_sequence<num_vertices>{});
   }()),
   ...);
}

template <typename T, typename S, typename F>
void compute_jacobians(F *f, StreamPool &streams) {
  if (!(f->store_jacobians() || !is_analytical<F>())) {
    return;
  }
  // Then for each vertex, we need to compute the error
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // At this point all necessary data should be on the GPU
  // std::array<T*, num_vertices> verts;
  auto verts = f->get_vertices();
  std::array<S *, num_vertices> jacs;
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    // verts[i] = f->vertex_descriptors[i]->vertices();
    jacs[i] = f->jacobians[i].data.data().get();
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();

    // Important: Must clear Jacobian storage
    thrust::fill(thrust::cuda::par_nosync.on(streams.select(i)),
                 f->jacobians[i].data.begin(), f->jacobians[i].data.end(),
                 static_cast<S>(0));
  }

  const auto num_factors = f->active_count();

  launch_kernel_jacobians<T, S>(f, hessian_ids, verts, jacs, num_factors,
                                streams,
                                std::make_index_sequence<num_vertices>{});
  streams.sync_n(num_vertices);
}

template <typename highp, typename T, size_t I, size_t N, size_t E, size_t D>
__global__ void
scale_jacobians_kernel(T *jacs, const highp *jacobian_scales,
                       const size_t *active_ids, const size_t *ids,
                       const size_t *hessian_ids, const uint8_t *active_state,
                       const size_t num_threads) {

  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;

  // This time, each factor will have D threads
  const size_t factor_id = active_ids[idx / D];
  const size_t col = idx % D;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }

  const auto jacobian_offset = factor_id * jacobian_size;
  const size_t hessian_offset = hessian_ids[local_id];
  const highp scale = jacobian_scales[hessian_offset + col];

  T *Jcol = jacs + jacobian_offset + col * E;
#pragma unroll
  for (size_t i = 0; i < E; i++) {

    const highp scaled_j = static_cast<highp>(Jcol[i]) * scale;

    Jcol[i] = static_cast<T>(scaled_j);
  }
}

template <typename T, typename S, typename F, std::size_t... Is>
void launch_kernel_scale_jacobians(
    F *f, T *jacobian_scales,
    std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    std::array<S *, F::get_num_vertices()> &jacs, const size_t num_factors,
    std::index_sequence<Is...>) {
  (([&] {
     constexpr size_t num_vertices = F::get_num_vertices();
     constexpr size_t dimension = F::get_vertex_sizes()[Is];
     const size_t num_threads = num_factors * dimension;

     size_t threads_per_block = 256;
     size_t num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     scale_jacobians_kernel<T, S, Is, num_vertices, F::error_dim, dimension>
         <<<num_blocks, threads_per_block>>>(
             jacs[Is], jacobian_scales, f->active_indices.data().get(),
             f->device_ids.data().get(), hessian_ids[Is],
             f->vertex_descriptors[Is]->get_active_state(), num_threads);
   }()),
   ...);
}

template <typename T, typename S, typename F>
void scale_jacobians(F *f, T *jacobian_scales) {

  if (!(f->store_jacobians() || !is_analytical<F>())) {
    return;
  }
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

  launch_kernel_scale_jacobians<T, S>(f, jacobian_scales, hessian_ids, jacs,
                                      num_factors,
                                      std::make_index_sequence<num_vertices>{});
}

// The output will be part of b with length of the vertex (where b = -J^T * r)
// Note the negative sign - different papers use different conventions
// TODO: Replace with generic J^T x r kernel?
// Note: The error vector is local to the factor
// Include precision matrix
template <typename T, typename S, size_t I, size_t N, size_t E, typename F,
          std::size_t... Is>
__global__ void compute_b_kernel(T *b, const T *error, const size_t *active_ids,
                                 const size_t *ids, const size_t *hessian_ids,
                                 const size_t num_threads, const S *jacs,
                                 const uint8_t *active_state, const S *pmat,
                                 const S *loss_derivative,
                                 std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  constexpr auto jacobian_size = vertex_sizes[I] * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / vertex_sizes[I]];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;
  const auto error_offset = factor_id * E;
  // constexpr auto col_offset = I*E; // for untransposed J
  const auto col_offset = (idx % vertex_sizes[I]) * E; // for untransposed J

  // Use loss kernel
  const T dL = loss_derivative[factor_id];

  T value = 0.0;
  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;
  T x2[E] = {0.0};

#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      // x2[i] += pmat[precision_offset + i + j*E] * error[error_offset + j]; //
      // col major
      x2[i] += dL * (T)pmat[precision_offset + i * E + j] *
               static_cast<T>(
                   error[error_offset + j]); // row major (use for faster access
                                             // on symmetrical matrix)
    }
  }

#pragma unroll
  for (int i = 0; i < E; i++) {
    value -= (T)jacs[jacobian_offset + col_offset + i] * x2[i];
  }

  const auto hessian_offset =
      hessian_ids[local_id]; // each vertex has a hessian_ids array

  // printf("Hessian offset: %u\n", hessian_offset);
  // printf("Adding b[%d] += %f\n", hessian_offset + (idx % vertex_sizes[I]),
  // value); printf("Thread %d, Hessian offset: %u\n", idx, hessian_offset);

  atomicAdd(&b[hessian_offset + (idx % vertex_sizes[I])], value);
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void
compute_b_dynamic_kernel(T *b, const T *error, const size_t *active_ids,
                         const size_t *ids, const size_t *hessian_ids,
                         const size_t num_threads, const VT args, const M *obs,
                         const T *jacobian_scales,
                         const typename F::ConstraintDataType *constraint_data,
                         const uint8_t *active_state, const S *pmat,
                         const S *loss_derivative, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  constexpr auto jacobian_size = vertex_sizes[I] * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / vertex_sizes[I]];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;
  const auto error_offset = factor_id * E;
  // constexpr auto col_offset = I*E; // for untransposed J
  const auto col_offset = (idx % vertex_sizes[I]) * E; // for untransposed J

  // Use loss kernel
  const T dL = loss_derivative[factor_id];

  T value = 0.0;
  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;
  T x2[E] = {0.0};

#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      // x2[i] += pmat[precision_offset + i + j*E] * error[error_offset + j]; //
      // col major
      x2[i] += dL * (T)pmat[precision_offset + i * E + j] *
               static_cast<T>(
                   error[error_offset + j]); // row major (use for faster access
                                             // on symmetrical matrix)
    }
  }

  // using G = std::conditional_t<is_low_precision<S>::value, T, S>;
  T jacobian[jacobian_size];

  compute_Jblock<T, I, N, M, E, F, VT>(jacobian, factor_id, local_id, obs,
                                       constraint_data, ids, hessian_ids, args,
                                       std::make_index_sequence<N>{});
  const auto hessian_offset = hessian_ids[local_id];
  const auto scale = jacobian_scales[hessian_offset + (idx % vertex_sizes[I])];
#pragma unroll
  for (int i = 0; i < E; i++) {
    value -= jacobian[col_offset + i] * x2[i];
  }
  value *= scale;

  atomicAdd(&b[hessian_offset + (idx % vertex_sizes[I])], value);
}

template <typename T, typename S, typename F, std::size_t... Is>
void launch_kernel_compute_b(
    F *f, T *b, std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
    const size_t num_factors, std::index_sequence<Is...>) {
  (([&] {
     constexpr auto num_vertices = F::get_num_vertices();
     const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
     // std::cout << "Launching compute b kernel" << std::endl;
     // std::cout << "Num threads: " << num_threads << std::endl;
     size_t threads_per_block = 256;
     size_t num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
     // std::endl; std::cout << "Checking residual ptr: " <<
     // f->residuals.data().get() << std::endl; std::cout << "Checking ids
     // ptr: " << f->device_ids.data().get() << std::endl;

     if (f->store_jacobians() || !is_analytical<F>()) {

       compute_b_kernel<T, S, Is, num_vertices, F::error_dim, F>
           <<<num_blocks, threads_per_block>>>(
               b, f->residuals.data().get(), f->active_indices.data().get(),
               f->device_ids.data().get(), hessian_ids[Is], num_threads,
               jacs[Is], f->vertex_descriptors[Is]->get_active_state(),
               f->precision_matrices.data().get(),
               f->chi2_derivative.data().get(),
               std::make_index_sequence<num_vertices>{});

     } else {
       // std::cout << "Launching compute b dynamic kernel" << std::endl;
       if constexpr (is_analytical<F>()) {
         compute_b_dynamic_kernel<T, S, Is, num_vertices,
                                  typename F::ObservationType, F::error_dim, F,
                                  typename F::VertexPointerPointerTuple>
             <<<num_blocks, threads_per_block>>>(
                 b, f->residuals.data().get(), f->active_indices.data().get(),
                 f->device_ids.data().get(), hessian_ids[Is], num_threads,
                 f->get_vertices(), f->device_obs.data().get(), jacobian_scales,
                 f->data.data().get(),
                 f->vertex_descriptors[Is]->get_active_state(),
                 f->precision_matrices.data().get(),
                 f->chi2_derivative.data().get(),
                 std::make_index_sequence<num_vertices>{});
       }
     }
   }()),
   ...);
}

template <typename T, typename S, typename F>
void compute_b_async(F *f, T *b, const T *jacobian_scales) {
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // std::array<T*, num_vertices> verts;
  auto verts = f->get_vertices();
  std::array<S *, num_vertices> jacs;
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    jacs[i] = f->jacobians[i].data.data().get();
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
  }

  constexpr auto error_dim = F::error_dim;
  const auto num_factors = f->active_count();
  // std::cout << "Computing b for " << num_factors << " factors" <<
  // std::endl;
  launch_kernel_compute_b<T, S>(f, b, hessian_ids, jacs, jacobian_scales,
                                num_factors,
                                std::make_index_sequence<num_vertices>{});
  // cudaStreamSynchronize(0);
}

} // namespace ops
} // namespace graphite
