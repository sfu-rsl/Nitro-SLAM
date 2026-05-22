/// @file update.hpp
#pragma once
#include <graphite/ops/common.hpp>

namespace graphite {

namespace ops {

template <typename T, typename S, typename Descriptor, typename V>
__global__ void
apply_update_kernel(V **vertices, const T *delta_x, const T *jacobian_scales,
                    const size_t *hessian_ids, const uint8_t *active_state,
                    const size_t num_threads) {
  const size_t vertex_id = get_thread_id();

  if (vertex_id >= num_threads || !is_vertex_active(active_state, vertex_id)) {
    return;
  }

  const T *delta = delta_x + hessian_ids[vertex_id];
  const T *scales = jacobian_scales + hessian_ids[vertex_id];

  std::array<T, Descriptor::dim> scaled_delta;
#pragma unroll
  for (size_t i = 0; i < Descriptor::dim; i++) {
    scaled_delta[i] = delta[i] * scales[i];
  }

  // vertices[vertex_id]->update(scaled_delta.data());
  Descriptor::Traits::update(*vertices[vertex_id], scaled_delta.data());
}

template <typename T, typename S, typename V>
void apply_update(V *v, const T *delta_x, T *jacobian_scales,
                  cudaStream_t stream) {
  const size_t num_threads = v->count();
  const auto threads_per_block = 256;
  const auto num_blocks =
      (num_threads + threads_per_block - 1) / threads_per_block;

  apply_update_kernel<T, S, V, typename V::VertexType>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          v->vertices(), delta_x, jacobian_scales, v->get_hessian_ids(),
          v->get_active_state(), num_threads);
}

} // namespace ops

} // namespace graphite