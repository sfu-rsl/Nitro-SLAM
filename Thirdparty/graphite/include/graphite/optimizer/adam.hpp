/// @file adam.hpp
#pragma once
#include <graphite/graph.hpp>
#include <graphite/ops/vector.hpp>
#include <graphite/solver/solver.hpp>
#include <iomanip>

namespace graphite {

namespace optimizer {

template <typename T, typename S> class AdamOptions {
public:
  AdamOptions()
      : iterations(100), learning_rate(1e-3), beta1(0.9), beta2(0.999),
        epsilon(1e-8), optimization_level(0), verbose(false),
        stop_flag(nullptr), streams(nullptr) {}

  size_t iterations;
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  uint8_t optimization_level;
  bool verbose;
  bool *stop_flag;
  StreamPool *streams;

  bool validate() const {
    if (streams == nullptr) {
      if (verbose) {
        std::cerr << "Adam options invalid: streams is null" << std::endl;
      }
      return false;
    }

    return true;
  }
};

/**
 * @brief Adam optimization algorithm
 *
 * @tparam T Scalar type
 * @tparam S Scalar type
 * @param graph Graph to optimize
 * @param options Options for the optimization
 * @return true if optimization completed successfully, false otherwise
 */
template <typename T, typename S>
bool adam(Graph<T, S> *graph, AdamOptions<T, S> *options) {

  // Initialize something for all iterations
  auto start = std::chrono::steady_clock::now();

  if (!options->validate()) {
    if (options->verbose) {
      std::cerr << "Adam options invalid" << std::endl;
    }
    return false;
  }

  auto streams = options->streams;

  if (!graph->initialize_optimization(options->optimization_level)) {
    return false;
  }

  if (!graph->build_structure()) {
    return false;
  }

  thrust::device_vector<T> delta_x(graph->get_hessian_dimension());

  bool run = true;

  double time =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  // Print iteration table headers
  if (options->verbose) {
    std::cout << std::setprecision(12) << std::setw(18) << "Iteration"
              << std::setw(24) << "Initial Chi2" << std::setw(24)
              << "Current Chi2" << std::setw(24) << std::setw(24) << "Time"
              << std::setw(24) << "Total Time" << std::endl;
    std::cout
        << "---------------------------------------------------------------"
           "---------------------------------------------------------------"
           "------------"
        << std::endl;
  }

  const T lr = options->learning_rate;
  const T beta1 = options->beta1;
  const T beta2 = options->beta2;
  const T epsilon = options->epsilon;

  const auto dim = delta_x.size();
  thrust::device_vector<T> m(dim, T(0.0));
  thrust::device_vector<T> v(dim, T(0.0));
  thrust::device_vector<T> g(dim, T(0.0));

  const auto num_iterations = options->iterations;
  for (size_t i = 0; i < num_iterations && run; i++) {

    start = std::chrono::steady_clock::now();
    graph->linearize(*streams);
    T chi2 = graph->chi2();
    thrust::copy(thrust::device, graph->get_b().begin(), graph->get_b().end(),
                 g.begin());
    ops::compute_adam_step_async(0, g.size(), g.data().get(),
                                 delta_x.data().get(), m.data().get(),
                                 v.data().get(), lr, beta1, beta2, epsilon, i);
    cudaStreamSynchronize(0);
    graph->apply_update(delta_x.data().get(), *streams);

    // Try step
    graph->compute_error();
    T new_chi2 = graph->chi2();

    double iteration_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    time += iteration_time;
    if (options->verbose) {
      std::cout << std::setprecision(12) << std::setw(18) << i << std::setw(24)
                << chi2 << std::setw(24) << new_chi2 << std::setw(24)
                << iteration_time << std::setw(24) << time << std::endl;
    }

    if (options->stop_flag && *(options->stop_flag)) {
      std::cout << "Stopping optimization due to stop flag" << std::endl;
      break;
    }
  }

  return run;
}

} // namespace optimizer

} // namespace graphite
