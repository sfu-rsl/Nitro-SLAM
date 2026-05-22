/// @file active.hpp
#pragma once
#include <graphite/ops/common.hpp>

namespace graphite {

/**
 * @namespace graphite::ops
 * @brief Operations for Graphite
 */
namespace ops {

// For flagging active vertices of a factor
// For an active factor, the MSB of a vertex's active state is set to 1
template <size_t I, size_t N>
__global__ void
flag_active_vertices_kernel(const size_t *ids, uint8_t *v_active,
                            const uint8_t *f_active, const size_t factor_count,
                            const uint8_t level) {
  const size_t factor_id = get_thread_id();

  if (factor_id >= factor_count) {
    return;
  }

  if (is_factor_active(f_active[factor_id], level)) {
    // TODO: Investigate if this can cause a bug - do we need to use atomicOr?
    const auto vertex_id = ids[factor_id * N + I];
    v_active[vertex_id] |= 0x80; // Set MSB to 1
  }
}

template <typename F, std::size_t... Is>
void launch_kernel_flag_active(F *f, const uint8_t level,
                               std::index_sequence<Is...>) {

  auto vd = f->vertex_descriptors;
  const uint8_t *f_active = f->device_active.data().get();
  const size_t *device_ids = f->device_ids.data().get();
  const size_t num_factors = f->internal_count();

  (([&] {
     uint8_t *v_active = vd[Is]->get_active_state();
     const size_t num_threads = num_factors;
     const auto threads_per_block = 256;
     const auto num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     flag_active_vertices_kernel<Is, F::get_num_vertices()>
         <<<num_blocks, threads_per_block>>>(device_ids, v_active, f_active,
                                             num_factors, level);
   }()),
   ...);
}

template <typename F> void flag_active_vertices(F *f, const uint8_t level) {
  launch_kernel_flag_active(f, level,
                            std::make_index_sequence<F::get_num_vertices()>{});
}

} // namespace ops

} // namespace graphite