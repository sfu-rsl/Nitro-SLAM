/// @file eigen.hpp
#pragma once
#include <graphite/hessian.hpp>
#include <graphite/solver/eigen_solver_interface.hpp>
#include <graphite/solver/solver.hpp>

namespace graphite {

template <typename T, typename S> class EigenLDLTSolver : public Solver<T, S> {
private:
  EigenLDLTWrapper<S> solver;

  Eigen::SparseMatrix<S, Eigen::ColMajor> matrix;

  using Index = typename Eigen::SparseMatrix<S, Eigen::ColMajor>::StorageIndex;

  Hessian<T, S> H;
  CSCMatrix<S, Index> d_matrix;

  thrust::host_vector<S> h_x;
  thrust::host_vector<S> h_b;

  void fill_matrix_structure() {
    const auto dim = d_matrix.d_pointers.size() - 1;
    matrix.resize(dim, dim);
    matrix.resizeNonZeros(d_matrix.d_values.size());

    auto h_ptrs = matrix.outerIndexPtr();
    auto h_indices = matrix.innerIndexPtr();

    thrust::copy(d_matrix.d_pointers.begin(), d_matrix.d_pointers.end(),
                 h_ptrs);
    thrust::copy(d_matrix.d_indices.begin(), d_matrix.d_indices.end(),
                 h_indices);

    h_x.resize(dim);
    h_b.resize(dim);
  }

  void fill_matrix_values() {
    auto h_values = matrix.valuePtr();
    thrust::copy(d_matrix.d_values.begin(), d_matrix.d_values.end(), h_values);
  }

public:
  EigenLDLTSolver() : solver() {}

  virtual void update_structure(Graph<T, S> *graph,
                                StreamPool &streams) override {
    H.build_structure(graph, streams);
    H.build_csc_structure(graph, d_matrix);
    fill_matrix_structure(); // for CPU matrix
    solver.analyze_pattern(matrix);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    H.update_values(graph, streams);
    H.update_csc_values(graph, d_matrix);
    fill_matrix_values(); // for CPU matrix
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  StreamPool &streams) override {
    H.apply_damping(graph, damping_factor, streams);
    H.update_csc_values(graph, d_matrix);
    fill_matrix_values(); // TODO: Use a more lightweight method to just update
                          // diagonal
  }

  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {

    auto dim = graph->get_hessian_dimension();

    if (!solver.factorize(matrix)) {
      std::cerr << "LDLT matrix decomposition failed!";
      return false;
    }

    thrust::fill(thrust::device, x, x + dim, static_cast<T>(0.0));

    thrust::copy(graph->get_b().begin(), graph->get_b().end(), h_b.data());
    thrust::device_ptr<T> d_x(
        x); // If you don't wrap the pointer, thrust breaks on older toolkits
    thrust::copy(d_x, d_x + dim, h_x.data());

    auto map_b = VecMap<S>(h_b.data(), dim, 1);
    auto map_x = VecMap<S>(h_x.data(), dim, 1);
    if (!solver.solve(map_b, map_x)) {
      std::cerr << "LDLT solve failed!";
      return false;
    }

    thrust::copy(h_x.begin(), h_x.end(), d_x);

    return true;
  }
};

} // namespace graphite