/// @file preconditioner.hpp
#pragma once
#include <graphite/factor.hpp>
#include <graphite/vertex.hpp>

namespace graphite {

template <typename T, typename S> class Preconditioner {
public:
  virtual void update_structure(Graph<T, S> *graph, StreamPool &streams) = 0;

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) = 0;

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  StreamPool &streams) = 0;

  virtual void apply(Graph<T, S> *graph, T *z, const T *r,
                     StreamPool &streams) = 0;
};

} // namespace graphite