/// @file common.hpp
#pragma once
#include <graphite/common.hpp>
#include <graphite/differentiation.hpp>
#include <graphite/stream.hpp>
#include <graphite/types.hpp>

namespace graphite {

__device__ __forceinline__ size_t get_thread_id() {
  return static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
         static_cast<size_t>(threadIdx.x);
}

template <typename T, size_t I, size_t N, typename M, size_t E, typename F,
          typename VT, std::size_t... Is>
__device__ void compute_Jblock(
    T *jacobian, const size_t factor_id, const size_t vertex_id, const M *obs,
    const typename F::ConstraintDataType *constraint_data, const size_t *ids,
    const size_t *hessian_ids, VT args, std::index_sequence<Is...>) {

  constexpr auto vertex_sizes = F::get_vertex_sizes();

  const M *local_obs = obs + factor_id;
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto vargs = cuda::std::make_tuple(
      (*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  using DataType = typename F::ConstraintDataType;
  using ObsType = typename F::ObservationType;

  if constexpr (std::is_empty<ObsType>::value &&
                std::is_empty<DataType>::value) {
    F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                       jacobian);
  } else if constexpr (std::is_empty<DataType>::value) {
    F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                       *local_obs, jacobian);
  } else if constexpr (std::is_empty<ObsType>::value) {
    F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                       *local_data, jacobian);
  } else {
    F::Traits::template jacobian<T, I>((*cuda::std::get<Is>(vargs))...,
                                       *local_obs, *local_data, jacobian);
  }
}

} // namespace graphite