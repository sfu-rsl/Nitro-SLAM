/// @file active.hpp
#pragma once
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace graphite {

// For factors, ignores the MSB
__host__ __device__ __forceinline__ bool
is_factor_active(const uint8_t active_val, const uint8_t level) {
  constexpr uint8_t NOT_MSB = 0x7F; // 01111111
  return (active_val & NOT_MSB) <= level && ((active_val & 0x80) == 0);
}

// For vertex
__host__ __device__ __forceinline__ bool
is_vertex_active(const uint8_t *active_state, const size_t vertex_id) {
  return !(active_state[vertex_id] > 0);
}

// Returns the number of active constraints and fills the active_indices vector
inline size_t
build_active_indices(const thrust::device_vector<uint8_t> &active,
                     thrust::device_vector<size_t> &active_indices,
                     const size_t count, const uint8_t level) {

  // Count active constraints
  const size_t active_count =
      thrust::count_if(thrust::device, active.begin(), active.end(),
                       [level] __device__(const uint8_t a) {
                         return is_factor_active(a, level);
                       });

  // Resize active indices to the number of active constraints
  active_indices.clear();
  active_indices.resize(active_count);

  // Fill active indices with the indices of active constraints
  thrust::copy_if(thrust::device, thrust::make_counting_iterator<size_t>(0),
                  thrust::make_counting_iterator<size_t>(count), active.begin(),
                  active_indices.begin(), [level] __device__(const uint8_t a) {
                    return is_factor_active(a, level);
                  });
  return active_count;
}

} // namespace graphite