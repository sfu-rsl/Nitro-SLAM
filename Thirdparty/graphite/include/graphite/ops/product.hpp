/// @file product.hpp
#pragma once
#include <graphite/ops/common.hpp>

namespace graphite {

namespace ops {

template <typename T, typename G, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__device__ void
compute_Jcol_ad(Dual<T, G> *error, const size_t col, const size_t factor_id,
                const size_t vertex_id, const M *obs,
                const typename F::ConstraintDataType *constraint_data,
                size_t *ids, const size_t *hessian_ids, VT args,
                std::index_sequence<Is...>) {

  constexpr auto vertex_sizes = F::get_vertex_sizes();

  const M *local_obs = obs + factor_id;
  // Dual<T, G> local_error[E];
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto v = cuda::std::make_tuple(std::array<Dual<T, G>, vertex_sizes[Is]>{}...);

  auto vargs =
      std::make_tuple((*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  auto copy_vertices = [&v, &vertex_sizes, &vargs](auto &&...ptrs) {
    ((std::tuple_element<Is, typename F::Traits::VertexDescriptors>::type::
          Traits::parameters(*std::get<Is>(vargs),
                             cuda::std::get<Is>(v).data())),
     ...);
  };

  std::apply(copy_vertices, vargs);

  cuda::std::get<I>(v)[col].dual = static_cast<G>(1);

  F::Traits::error(cuda::std::get<Is>(v).data()..., local_obs, error, vargs,
                   local_data);
}

// Compute J * x where the length of vector x matches the Hessian dimension
// Each Jacobian block needs to be accessed just once
// So we need E threads for each block (error dimension)
// In total we should hae E*num_factors threads?
template <typename T, typename S, size_t I, size_t N, size_t E, size_t D,
          typename F, std::size_t... Is>
__global__ void compute_Jv_kernel(T *y, const T *x, const size_t *active_ids,
                                  const size_t *ids, const size_t *hessian_ids,
                                  const size_t num_threads, const S *jacs,
                                  const uint8_t *active_state,
                                  std::index_sequence<Is...>) {

  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto jacobian_size = D * E;

  // Each J block is stored as E x d col major, where d is the vertex size
  const size_t factor_id = active_ids[idx / E];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  T value = 0;

  const auto hessian_offset =
      hessian_ids[local_id]; // each vertex has a hessian_ids array
  const auto row_offset = (idx % E);
  // Adding i*E skips to the next column
  // size_t residual_offset = 0; // need to pass this in
  // it's the offset into the r vector
  // #pragma unroll
  // for (int i = 0; i < d; i++) {
  //     value += jacs[jacobian_offset + row_offset + i*E] * x[hessian_offset +
  //     i];
  // }

  const S *jrow = jacs + jacobian_offset + row_offset;
  const T *x_start = x + hessian_offset;

#pragma unroll
  for (int i = 0; i < D; i++) {
    value += (T)(jrow[i * E] * (S)x_start[i]);
  }

  atomicAdd(&y[idx], value);
  // y[idx] += value; // avoid unless sure that atomicAdd is not needed
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void compute_Jv_dynamic_manual2(
    T *y, T *x, const M *obs, const T *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data,
    const size_t *active_ids, const size_t *ids, const size_t *hessian_ids,
    const size_t num_factors, VT args, const uint8_t *active_state,
    std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();
  constexpr size_t D = F::get_vertex_sizes()[I];

  const size_t factor_id = active_ids[idx / E];
  const size_t row_in_jacobian = idx % E;
  if (factor_id < num_factors) {

    const auto vertex_id = ids[factor_id * N + I];

    const auto hess_col = hessian_ids[vertex_id];
    const T *x_start = x + hess_col;

    // Each thread block stores a complete Jacobian row in shared memory
    if (is_vertex_active(active_state, vertex_id)) {
      // using G = std::conditional_t<is_low_precision<S>::value, T, S>;
      using G = T;
      // Dual<T, G> error[E];
      constexpr auto jacobian_size = E * D;
      G jacobian[jacobian_size];
      compute_Jblock<T, I, N, M, E, F, VT>(jacobian, factor_id, vertex_id, obs,
                                           constraint_data, ids, hessian_ids,
                                           args, std::make_index_sequence<N>{});

      T sum = 0.0;
#pragma unroll
      for (size_t i = 0; i < D; i++) {
        const T scaled_j = static_cast<T>(jacobian[row_in_jacobian + i * E]) *
                           jacobian_scales[hess_col + i];
        sum += static_cast<T>((S)scaled_j * (S)x_start[i]);
      }
      atomicAdd(&y[idx], sum);
    }
  }
}

template <typename T, typename S, typename F, std::size_t... Is>
void launch_kernel_compute_Jv(
    F *f, T *out, T *in,
    std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
    const size_t num_factors, StreamPool &streams, std::index_sequence<Is...>) {
  (([&] {
     constexpr auto num_vertices = F::get_num_vertices();
     constexpr auto vertex_sizes = F::get_vertex_sizes();
     if (f->store_jacobians() || !is_analytical<F>()) {
       const auto num_threads = num_factors * F::error_dim;

       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;
       compute_Jv_kernel<T, S, Is, num_vertices, F::error_dim,
                         f->get_vertex_sizes()[Is], F>
           <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
               out, in, f->active_indices.data().get(),
               f->device_ids.data().get(), hessian_ids[Is], num_threads,
               jacs[Is], f->vertex_descriptors[Is]->get_active_state(),
               std::make_index_sequence<num_vertices>{});
     } else {
       constexpr auto num_vertices = F::get_num_vertices();
       constexpr auto vertex_sizes = F::get_vertex_sizes();
       const auto num_threads = num_factors * F::error_dim;

       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;
       constexpr size_t E = F::error_dim;

       if constexpr (is_analytical<F>()) {

         compute_Jv_dynamic_manual2<T, S, Is, num_vertices,
                                    typename F::ObservationType, E, F,
                                    typename F::VertexPointerPointerTuple>
             <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
                 out, in, f->device_obs.data().get(), jacobian_scales,
                 f->data.data().get(), f->active_indices.data().get(),
                 f->device_ids.data().get(), hessian_ids[Is], num_factors,
                 f->get_vertices(),
                 f->vertex_descriptors[Is]->get_active_state(),
                 std::make_index_sequence<num_vertices>{});
       }
     }
   }()),
   ...);
}

template <typename T, typename S, typename F>
void compute_Jv(F *f, T *out, T *in, const T *jacobian_scales,
                StreamPool &streams) {
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // std::array<T*, num_vertices> verts;
  auto verts = f->get_vertices();
  std::array<S *, num_vertices> jacs;
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    // verts[i] = f->vertex_descriptors[i]->x();
    jacs[i] = f->jacobians[i].data.data().get();
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
  }

  const auto num_factors = f->active_count();

  launch_kernel_compute_Jv<T, S>(f, out, in, hessian_ids, jacs, jacobian_scales,
                                 num_factors, streams,
                                 std::make_index_sequence<num_vertices>{});
  streams.sync_n(num_vertices);
}

// Compute J^T * x where x is the size of the residual vector
// Each Jacobian block needs to be accessed just once
// For each block, we need d threads where d is the vertex size
// We need to load the x vector location for the corresponding block row of J
// So this assumes that the x vector has the same layout as the residual vector
// for this factor (rather than a global residual vector) The aggregate output
// will be H x len(x) where H is hessian dimension
// Compute J^T * P * x where P is the precision matrix
template <typename T, typename S, size_t I, size_t N, size_t E, size_t D,
          typename F, std::size_t... Is>
__global__ void compute_JtPv_kernel(T *y, const T *x, const size_t *active_ids,
                                    const size_t *ids,
                                    const size_t *hessian_ids,
                                    const size_t num_threads, const S *jacs,
                                    const uint8_t *active_state, const S *pmat,
                                    const S *chi2_derivative,
                                    const std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto jacobian_size = D * E;

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
  const auto error_offset = factor_id * E;
  const auto col_offset = (idx % D) * E; // for untransposed J

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Use loss kernel
  // const auto dL = chi2_derivative[factor_id];

  // T x2[E] = {0};
  // T value = 0;

  const S *jcol = jacs + jacobian_offset + col_offset;

  const S *precision_matrix = pmat + precision_offset;
  const T *x_start = x + error_offset;

  T value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    const auto p_row = precision_matrix + i * E;
    S x2 = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      x2 += p_row[j] * (S)x_start[j];
    }
    value += (T)(jcol[i] * x2);
  }

  value *= (T)chi2_derivative[factor_id];

  const auto hessian_offset =
      hessian_ids[local_id]; // each vertex has a hessian_ids array

  atomicAdd(&y[hessian_offset + (idx % D)], value);
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          size_t D, typename F, typename VT, std::size_t... Is>
__global__ void compute_JtPv_dynamic_kernel(
    T *y, const T *x, const size_t *active_ids, const size_t *ids,
    const size_t *hessian_ids, const size_t num_threads, const VT args,
    const M *obs, const T *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data,
    const uint8_t *active_state, const S *pmat, const S *chi2_derivative,
    const std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto jacobian_size = D * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / D];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (!is_vertex_active(active_state, local_id)) {
    return;
  }
  const auto error_offset = factor_id * E;
  const auto col_offset = (idx % D) * E; // for untransposed J

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  using G = T;
  G jacobian[jacobian_size];

  compute_Jblock<T, I, N, M, E, F, VT>(jacobian, factor_id, local_id, obs,
                                       constraint_data, ids, hessian_ids, args,
                                       std::make_index_sequence<N>{});

  const auto hessian_offset = hessian_ids[local_id];
  const auto scale = jacobian_scales[hessian_offset + (idx % D)];

  const G *jcol = jacobian + col_offset;

  const S *precision_matrix = pmat + precision_offset;
  const T *x_start = x + error_offset;

  T value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    const auto p_row = precision_matrix + i * E;
    S x2 = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      x2 += p_row[j] * (S)x_start[j];
    }
    value += (T)((S)jcol[i] * x2);
  }

  value *= (T)chi2_derivative[factor_id] * scale;

  atomicAdd(&y[hessian_offset + (idx % D)], value);
}

template <typename T, typename S, typename F, std::size_t... Is>
void launch_kernel_compute_JtPv(
    F *f, T *out, T *in,
    std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
    const size_t num_factors, StreamPool &streams, std::index_sequence<Is...>) {
  (([&] {
     constexpr auto num_vertices = F::get_num_vertices();
     const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
     // std::cout << "Launching compute Jtv kernel" << std::endl;
     // std::cout << "Num threads: " << num_threads << std::endl;
     size_t threads_per_block = 256;
     size_t num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
     // std::endl; std::cout << "Checking residual ptr: " <<
     // f->residuals.data().get() << std::endl; std::cout << "Checking ids
     // ptr: " << f->device_ids.data().get() << std::endl;
     if (f->store_jacobians() || !is_analytical<F>()) {
       compute_JtPv_kernel<T, S, Is, num_vertices, F::error_dim,
                           f->get_vertex_sizes()[Is], F>
           <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
               out, in, f->active_indices.data().get(),
               f->device_ids.data().get(), hessian_ids[Is], num_threads,
               jacs[Is], f->vertex_descriptors[Is]->get_active_state(),
               f->precision_matrices.data().get(),
               f->chi2_derivative.data().get(),
               std::make_index_sequence<num_vertices>{});
     } else {
       if constexpr (is_analytical<F>()) {
         compute_JtPv_dynamic_kernel<T, S, Is, num_vertices,
                                     typename F::ObservationType, F::error_dim,
                                     f->get_vertex_sizes()[Is], F,
                                     typename F::VertexPointerPointerTuple>
             <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
                 out, in, f->active_indices.data().get(),
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
void compute_Jtv(F *f, T *out, T *in, const T *jacobian_scales,
                 StreamPool &streams) {
  constexpr auto num_vertices = f->get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // std::array<T*, num_vertices> verts;
  auto verts = f->get_vertices();
  std::array<S *, num_vertices> jacs;
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    // verts[i] = f->vertex_descriptors[i]->x();
    jacs[i] = f->jacobians[i].data.data().get();
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
  }

  const auto num_factors = f->active_count();

  launch_kernel_compute_JtPv<T, S>(f, out, in, hessian_ids, jacs,
                                   jacobian_scales, num_factors, streams,
                                   std::make_index_sequence<num_vertices>{});
  streams.sync_n(num_vertices);
}

} // namespace ops

} // namespace graphite