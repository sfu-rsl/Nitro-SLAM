/** @file circle.cu
  A simple example which fits points to a circle.
*/
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <graphite/optimizer/levenberg_marquardt.hpp>
#include <graphite/preconditioner/identity.hpp>
#include <graphite/solver/pcg.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace graphite {

// Point definition
template <typename T> using Point = Eigen::Matrix<T, 2, 1>;

// Traits for Point
template <typename T, typename S> struct PointTraits {
  static constexpr size_t dimension = 2;
  using Vertex = Point<T>;

  template <typename P>
  d_fn static void parameters(const Vertex &vertex, P *parameters) {
    Eigen::Map<Eigen::Matrix<P, dimension, 1>> params_map(parameters);
    params_map = vertex.template cast<P>();
  }

  d_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};

template <typename T, typename S>
using PointDescriptor = VertexDescriptor<T, S, PointTraits<T, S>>;

// Factor traits for the circle constraint
template <typename T, typename S> struct CircleFactorTraits {
  static constexpr size_t dimension = 1;
  using VertexDescriptors = std::tuple<PointDescriptor<T, S>>;
  using Observation = T;
  using Data = Empty;
  using Loss = DefaultLoss<T, dimension>;
  // using Differentiation = DifferentiationMode::Auto;
  using Differentiation = DifferentiationMode::Manual;

  template <typename D>
  d_fn static void error(const D *point, const T &obs, D *error) {
    const auto x = point[0];
    const auto y = point[1];
    const auto r = obs;
    error[0] = x * x + y * y - r * r;
  }

  template <typename J, size_t I>
  d_fn static void jacobian(const Point<T> &point, const T &obs, J *jacobian) {
    if constexpr (I == 0) {
      const auto x = point(0);
      const auto y = point(1);
      jacobian[0] = 2 * x;
      jacobian[1] = 2 * y;
    }
  }
};

template <typename T, typename S>
using CircleFactor = FactorDescriptor<T, S, CircleFactorTraits<T, S>>;

} // namespace graphite

int main(void) {

  using namespace graphite;

  // Initialize CUDA
  cudaSetDevice(0);

  // Create graph
  using FP = double;
  using SP = double;
  Graph<FP, SP> graph;

  const size_t num_vertices = 5;

  // Create vertices
  auto point_desc = PointDescriptor<FP, SP>();
  point_desc.reserve(num_vertices);
  graph.add_descriptor(&point_desc);

  FP center[2] = {0.0, 0.0};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FP> dist(0.0, 2 * M_PI);

  const FP radius = 4.0;
  const FP sigma = 0.3;

  std::normal_distribution<FP> n1(0.0, sigma);
  std::normal_distribution<FP> n2(0.0, sigma);

  managed_vector<Point<FP>> pts(num_vertices); // addresses must not change
  constexpr auto id_offset = 10;               // user provides arbitrary ids

  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    FP angle = dist(gen);
    FP point[2] = {center[0] + radius * cos(angle),
                   center[1] + radius * sin(angle)};
    point[0] += n1(gen);
    point[1] += n2(gen);
    pts[vertex_id] = Point<FP>(point[0], point[1]);
    std::cout << "Adding point " << vertex_id << "=(" << point[0] << ", "
              << point[1] << ") with radius="
              << sqrt(point[0] * point[0] + point[1] * point[1]) << std::endl;
    point_desc.add_vertex(vertex_id + id_offset, &pts[vertex_id]);
  }
  // Create edges
  auto factor_desc = CircleFactor<FP, SP>(&point_desc);
  factor_desc.reserve(num_vertices);
  graph.add_descriptor(&factor_desc);

  const auto loss = DefaultLoss<FP, 1>();
  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    factor_desc.add_factor({vertex_id + id_offset}, radius, nullptr, Empty(),
                           loss);
  }

  // Set the last vertex as fixed
  point_desc.set_fixed(num_vertices - 1 + id_offset, true);

  // Disable third constraint for point 2
  factor_desc.set_active(2, 0x1);

  // Configure solver
  graphite::IdentityPreconditioner<FP, SP> preconditioner;
  graphite::PCGSolver<FP, SP> solver(50, 1e-20, 10.0, &preconditioner);

  // Optimize
  constexpr size_t iterations = 100;
  std::cout << "Graph built with " << num_vertices << " vertices and "
            << factor_desc.internal_count() << " factors." << std::endl;
  std::cout << "Optimizing!" << std::endl;

  StreamPool streams(1);
  constexpr uint8_t optimization_level = 0;

  optimizer::LevenbergMarquardtOptions<FP, SP> options;
  options.solver = &solver;
  options.initial_damping = 1e-6;
  options.iterations = iterations;
  options.optimization_level = optimization_level;
  options.verbose = true;
  options.streams = &streams;

  auto start = std::chrono::steady_clock::now();
  optimizer::levenberg_marquardt<FP, SP>(&graph, &options);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Optimization took " << elapsed.count() << " seconds."
            << std::endl;

  // Read back optimized values
  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    const auto point = point_desc.get_vertex(vertex_id + id_offset);
    const auto &p = *point;
    auto x = p(0);
    auto y = p(1);
    std::cout << "Optimized point " << vertex_id << "=(" << x << ", " << y
              << ") with radius=" << sqrt(x * x + y * y) << std::endl;
  }

  std::cout << "points 2 and 4 should remain unchanged." << std::endl;

  return 0;
}
