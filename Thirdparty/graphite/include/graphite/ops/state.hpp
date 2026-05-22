/// @file active.hpp
#pragma once
#include <graphite/ops/common.hpp>

namespace graphite {

template <typename, typename = void>
struct has_type_alias_State : std::false_type {};

template <typename T>
struct has_type_alias_State<T, std::void_t<typename T::State>>
    : std::true_type {};

template <typename T, typename Fallback, typename = void> struct get_State_or {
  using type = Fallback;
};

// Specialization: use T::State if it exists
template <typename T, typename Fallback>
struct get_State_or<T, Fallback, std::void_t<typename T::State>> {
  using type = typename T::State;
};

// Helper alias
template <typename T, typename Fallback>
using get_State_or_t = typename get_State_or<T, Fallback>::type;

namespace ops {

template <typename VertexType, typename State, typename Traits, typename T>
__global__ void backup_state_kernel(VertexType **vertices, State *dst,
                                    const uint8_t *active_state,
                                    const size_t num_vertices) {

  const size_t vertex_id = get_thread_id();

  if (vertex_id >= num_vertices || !is_vertex_active(active_state, vertex_id))
    return;
  if constexpr (has_type_alias_State<Traits>::value) {
    dst[vertex_id] = Traits::get_state(*vertices[vertex_id]);
  } else {
    dst[vertex_id] = *vertices[vertex_id];
  }
}

template <typename VertexType, typename State, typename Traits, typename T>
__global__ void set_state_kernel(VertexType **vertices, const State *src,
                                 const uint8_t *active_state,
                                 const size_t num_vertices) {

  const size_t vertex_id = get_thread_id();

  if (vertex_id >= num_vertices || !is_vertex_active(active_state, vertex_id))
    return;

  if constexpr (has_type_alias_State<Traits>::value) {
    Traits::set_state(*vertices[vertex_id], src[vertex_id]);
  } else {
    *vertices[vertex_id] = src[vertex_id];
  }
}

} // namespace ops

} // namespace graphite