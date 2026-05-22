#pragma once

#include <graphite/vertex.hpp>

namespace test_types {

struct Vec2 {
  float x;
  float y;
};

struct Vec2Traits {
  static constexpr size_t dimension = 2;
  using Vertex = Vec2;

  template <typename P>
  d_fn static void parameters(const Vertex &vertex, P *parameters) {
    parameters[0] = static_cast<P>(vertex.x);
    parameters[1] = static_cast<P>(vertex.y);
  }

  d_fn static void update(Vertex &vertex, const float *delta) {
    vertex.x += delta[0];
    vertex.y += delta[1];
  }
};

struct Vec2StateTraits {
  static constexpr size_t dimension = 2;
  using Vertex = Vec2;
  using State = float;

  template <typename P>
  d_fn static void parameters(const Vertex &vertex, P *parameters) {
    parameters[0] = static_cast<P>(vertex.x);
    parameters[1] = static_cast<P>(vertex.y);
  }

  d_fn static void update(Vertex &vertex, const float *delta) {
    vertex.x += delta[0];
    vertex.y += delta[1];
  }

  d_fn static State get_state(const Vertex &vertex) { return vertex.x; }

  d_fn static void set_state(Vertex &vertex, const State &state) {
    vertex.x = state;
  }
};

using Vec2Descriptor = graphite::VertexDescriptor<float, float, Vec2Traits>;
using Vec2StateDescriptor =
    graphite::VertexDescriptor<float, float, Vec2StateTraits>;

} // namespace test_types
