/// @file levenberg_marquardt.hpp
#pragma once
#include <graphite/graph.hpp>
#include <graphite/solver/solver.hpp>
#include <iomanip>

/**
 * @namespace graphite
 * @brief The top-level namespace for Graphite
 */
namespace graphite {

/**
 * @namespace graphite::optimizer
 * @brief Optimization algorithms for Graphite
 */
namespace optimizer {

template <typename T, typename S>
T compute_rho(Graph<T, S> *graph, thrust::device_vector<T> &delta_x,
              const T chi2, const T new_chi2, const T mu,
              const bool step_is_good) {
  // Compute rho
  //  TODO: Don't store these in the graph
  auto &b = graph->get_b();
  T num = (chi2 - new_chi2);
  T denom = 1.0;
  if (step_is_good) {

    const auto n = delta_x.size();
    const auto bb = b.data().get();
    const auto dx = delta_x.data().get();

    denom = thrust::transform_reduce(
        thrust::device, thrust::make_counting_iterator<std::size_t>(0),
        thrust::make_counting_iterator<std::size_t>(n),
        [dx, bb, mu] __host__ __device__(const std::size_t i) {
          T x = dx[i];
          return x * (mu * x + bb[i]);
        },
        T{0}, thrust::plus<T>{});

    denom += 1.0e-3;
  }

  return num / (denom);
}

/**
 * @brief Levenberg-Marquardt options
 */
template <typename T, typename S> class LevenbergMarquardtOptions {
public:
  LevenbergMarquardtOptions()
      : solver(nullptr), iterations(10), initial_damping(1e-4),
        optimization_level(0), verbose(false), stop_flag(nullptr),
        streams(nullptr) {}

  /// Pointer to the linear solver to use for the optimization
  Solver<T, S> *solver;
  /// Maximum number of iterations to run the optimization for
  size_t iterations;
  /// Initial damping factor (lambda) value
  double initial_damping;
  /// Constraints with level <= optimization_level will be optimized
  uint8_t optimization_level;
  /// If true, print verbose output during optimization
  bool verbose;
  /// Pointer to a stop flag which can be used to terminate the optimization
  bool *stop_flag;
  /// Pointer to a pool of streams
  StreamPool *streams;

  /**
   * @brief Validate the options
   * @return true if the options are valid, false otherwise
   */
  bool validate() const {
    if (solver == nullptr) {
      if (verbose) {
        std::cerr << "Levenberg-Marquardt options invalid: solver is null"
                  << std::endl;
      }
      return false;
    }
    if (streams == nullptr) {
      if (verbose) {
        std::cerr << "Levenberg-Marquardt options invalid: streams is null"
                  << std::endl;
      }
      return false;
    }

    return true;
  }
};

/**
 * @brief Levenberg-Marquardt optimization algorithm
 *
 * @tparam T Scalar type
 * @tparam S Scalar type
 * @param graph Graph to optimize
 * @param options Options for the optimization
 * @return true if optimization completed successfully, false otherwise
 */
template <typename T, typename S>
bool levenberg_marquardt(Graph<T, S> *graph,
                         LevenbergMarquardtOptions<T, S> *options) {

  // Initialize something for all iterations
  auto start = std::chrono::steady_clock::now();

  if (!options->validate()) {
    if (options->verbose) {
      std::cerr << "Levenberg-Marquardt options invalid" << std::endl;
    }
    return false;
  }

  T mu = static_cast<T>(options->initial_damping);
  T nu = 2;

  auto solver = options->solver;
  auto streams = options->streams;

  if (!graph->initialize_optimization(options->optimization_level)) {
    return false;
  }

  if (!graph->build_structure()) {
    return false;
  }

  solver->update_structure(graph, *streams);

  graph->linearize(*streams);

  // Initialize solver values
  solver->update_values(graph, *streams);
  T chi2 = graph->chi2();

  thrust::device_vector<T> delta_x(graph->get_hessian_dimension());

  bool run = true;

  double time =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  // Print iteration table headers
  if (options->verbose) {
    std::cout << std::setprecision(12) << std::setw(18) << "Iteration"
              << std::setw(24) << "Initial Chi2" << std::setw(24)
              << "Current Chi2" << std::setw(24) << "Lambda" << std::setw(24)
              << "Time" << std::setw(24) << "Total Time" << std::endl;
    std::cout
        << "---------------------------------------------------------------"
           "---------------------------------------------------------------"
           "------------"
        << std::endl;
  }

  const auto num_iterations = options->iterations;
  for (size_t i = 0; i < num_iterations && run; i++) {

    start = std::chrono::steady_clock::now();

    solver->set_damping_factor(graph, static_cast<T>(mu), *streams);
    bool solve_ok = solver->solve(graph, delta_x.data().get(), *streams);

    graph->backup_parameters();
    graph->apply_update(delta_x.data().get(), *streams);

    // Try step
    graph->compute_error();
    T new_chi2 = graph->chi2();

    if (!solve_ok) {
      new_chi2 = std::numeric_limits<T>::max();
    }

    T rho = compute_rho(graph, delta_x, chi2, new_chi2, mu, solve_ok);

    if (solve_ok && std::isfinite(new_chi2) && rho > 0) {
      // update hyperparameters
      double alpha = 1.0 - pow(2.0 * rho - 1.0, 3);
      alpha = std::max(std::min(alpha, 2.0 / 3.0), 1.0 / 3.0);
      mu *= static_cast<T>(alpha);
      nu = 2;
      // Relinearize since step is accepted
      graph->linearize(*streams);
      solver->update_values(graph, *streams);
      // std::cout << "Good step" << std::endl;
      // std::cout << "rho: " << rho << std::endl;
    } else {
      graph->revert_parameters();
      graph->compute_error();
      graph->chi2();
      // update hyperparameters
      mu *= nu;
      nu *= 2;
      // std::cout << "Bad step" << std::endl;
      // std::cout << "rho: " << rho << std::endl;
      // std::cout << "Previous chi2: " << chi2 << std::endl;
      // std::cout << "Current chi2: " << new_chi2 << std::endl;
      new_chi2 = chi2;
    }

    double iteration_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    time += iteration_time;
    if (options->verbose) {
      std::cout << std::setprecision(12) << std::setw(18) << i << std::setw(24)
                << chi2 << std::setw(24) << new_chi2 << std::setw(24) << mu
                << std::setw(24) << iteration_time << std::setw(24) << time
                << std::endl;
    }
    chi2 = new_chi2;

    if (!std::isfinite(mu)) {
      std::cout << "Damping factor is infinite, terminating optimization"
                << std::endl;
      run = false;
    }

    if (rho == 0) {
      std::cout << "Rho is zero, terminating optimization" << std::endl;
      break;
    }

    if (options->stop_flag && *(options->stop_flag)) {
      std::cout << "Stopping optimization due to stop flag" << std::endl;
      break;
    }
  }

  return run;
}

/**
 * @brief Levenberg-Marquardt with similar early termination stopping criteria
 * to ORB-SLAM.
 *
 * @tparam T Scalar type
 * @tparam S Scalar type
 * @param graph Graph to optimize
 * @param options Options for the optimization
 * @return true if optimization completed successfully, false otherwise
 */

template <typename T, typename S>
bool levenberg_marquardt2(Graph<T, S> *graph,
                          LevenbergMarquardtOptions<T, S> *options) {

  // Initialize something for all iterations
  auto start = std::chrono::steady_clock::now();

  if (!options->validate()) {
    if (options->verbose) {
      std::cerr << "Levenberg-Marquardt options invalid" << std::endl;
    }
    return false;
  }

  T mu = static_cast<T>(options->initial_damping);
  T nu = 2;
  int num_bad = 0;

  auto solver = options->solver;
  auto streams = options->streams;

  if (!graph->initialize_optimization(options->optimization_level)) {
    return false;
  }

  if (!graph->build_structure()) {
    return false;
  }
  solver->update_structure(graph, *streams);

  graph->linearize(*streams);
  solver->update_values(graph, *streams);
  T chi2 = graph->chi2();

  thrust::device_vector<T> delta_x(graph->get_hessian_dimension());

  bool run = true;

  double time =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  // Print iteration table headers
  if (options->verbose) {
    std::cout << std::setprecision(12) << std::setw(18) << "Iteration"
              << std::setw(24) << "Initial Chi2" << std::setw(24)
              << "Current Chi2" << std::setw(24) << "Lambda" << std::setw(24)
              << "Time" << std::setw(24) << "Total Time" << std::endl;
    std::cout
        << "---------------------------------------------------------------"
           "---------------------------------------------------------------"
           "------------"
        << std::endl;
  }

  const auto num_iterations = options->iterations;
  for (size_t i = 0; i < num_iterations && run; i++) {

    start = std::chrono::steady_clock::now();
    T initial_chi2 = chi2;
    T end_chi2 = initial_chi2;

    solver->set_damping_factor(graph, static_cast<T>(mu), *streams);

    bool solve_ok = solver->solve(graph, delta_x.data().get(), *streams);

    graph->backup_parameters();

    graph->apply_update(delta_x.data().get(), *streams);

    // Try step
    graph->compute_error();

    T new_chi2 = graph->chi2();

    if (!solve_ok) {
      new_chi2 = std::numeric_limits<T>::max();
    }

    T rho = compute_rho(graph, delta_x, chi2, new_chi2, mu, solve_ok);

    bool step_accepted = false;
    if (solve_ok && std::isfinite(new_chi2) && rho > 0) {
      // update hyperparameters
      double alpha = 1.0 - pow(2.0 * rho - 1.0, 3);
      alpha = std::max(std::min(alpha, 2.0 / 3.0), 1.0 / 3.0);
      mu *= static_cast<T>(alpha);
      nu = 2;
      // Relinearize since step is accepted
      graph->linearize(*streams);

      solver->update_values(graph, *streams);

      // H and b are valid
      // residuals should be valid
      // chi2 should be valid

      end_chi2 = new_chi2;
      step_accepted = true;
      // std::cout << "Good step" << std::endl;
      // std::cout << "rho: " << rho << std::endl;
    } else {
      graph->revert_parameters();

      // At this point, what is valid?
      // - Linear system (H and b) are still valid
      // - chi2 and derivatives are invalid (so the implicit Hessian is invalid,
      // i.e. PCG will be invalid)
      // - However chi2 depends on the residuals, which are also invalid
      // So we need to recompute the error, and the chi2

      graph->compute_error();
      graph->chi2();
      // hyperparameters
      mu *= nu;
      nu *= 2;
      // std::cout << "Bad step" << std::endl;
      new_chi2 = initial_chi2; // chi2 computation may be non-deterministic
    }

    double iteration_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    time += iteration_time;
    if (options->verbose) {
      std::cout << std::setprecision(4) << std::setw(10) << i << std::setw(16)
                << chi2 << std::setw(16) << new_chi2 << std::setw(16) << mu
                << std::setw(16) << iteration_time << std::setw(16) << time
                << std::endl;
    }
    chi2 = new_chi2;

    if (!std::isfinite(mu)) {
      std::cout << "Damping factor is infinite, terminating optimization"
                << std::endl;
      run = false;
    }

    if (rho == 0) {
      std::cout << "Rho is zero, terminating optimization" << std::endl;
      break;
    }

    if (options->stop_flag && *(options->stop_flag)) {
      std::cout << "Stopping optimization due to stop flag" << std::endl;
      break;
    }

    if (step_accepted) {
      if (((initial_chi2 - end_chi2) * 1.0e3) < initial_chi2) {
        num_bad++;
      } else {
        num_bad = 0;
      }

      if (num_bad >= 3) {
        break;
      }
    }
  }

  return run;
}

} // namespace optimizer
} // namespace graphite