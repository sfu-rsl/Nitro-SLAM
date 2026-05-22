/// @file solver.hpp
#pragma once
#include <graphite/common.hpp>
#include <graphite/graph.hpp>
#include <graphite/utils.hpp>

namespace graphite {

/**
 * @brief Linear solver interface. Implement this for your own linear solvers.
 */
template <typename T, typename S> class Solver {
public:
  virtual ~Solver() = default;

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  StreamPool &streams) = 0;

  virtual void update_structure(Graph<T, S> *graph, StreamPool &streams) = 0;

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) = 0;

  virtual bool solve(Graph<T, S> *graph, T *delta_x, StreamPool &streams) = 0;
};

} // namespace graphite