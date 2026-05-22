#include "common_types.hpp"
#include <graphite/factor.hpp>
#include <gtest/gtest.h>

namespace {
using test_types::Vec2;
using test_types::Vec2Descriptor;

template <typename DiffMode,
          template <typename, int> class LossT = graphite::DefaultLoss>
struct UnaryFactorTraits {
  static constexpr size_t dimension = 1;
  using VertexDescriptors = std::tuple<Vec2Descriptor>;
  using Observation = float;
  using Data = graphite::Empty;
  using Loss = LossT<float, dimension>;
  using Differentiation = DiffMode;

  template <typename D>
  d_fn static void error(const D *vertex, const Observation &obs, D *residual) {
    residual[0] = vertex[0] - obs;
  }

  template <typename D, size_t I>
  d_fn static void jacobian(const Vec2 &, const Observation &, D *jacobian) {
    jacobian[0] = static_cast<D>(1);
    jacobian[1] = static_cast<D>(0);
  }
};

template <typename DiffMode,
          template <typename, int> class LossT = graphite::DefaultLoss>
struct CoupledUnaryFactorTraits {
  static constexpr size_t dimension = 1;
  using VertexDescriptors = std::tuple<Vec2Descriptor>;
  using Observation = float;
  using Data = graphite::Empty;
  using Loss = LossT<float, dimension>;
  using Differentiation = DiffMode;

  template <typename D>
  d_fn static void error(const D *vertex, const Observation &obs, D *residual) {
    residual[0] =
        static_cast<D>(2) * vertex[0] + static_cast<D>(3) * vertex[1] - obs;
  }

  template <typename D, size_t I>
  d_fn static void jacobian(const Vec2 &, const Observation &, D *jacobian) {
    jacobian[0] = static_cast<D>(2);
    jacobian[1] = static_cast<D>(3);
  }
};

template <typename DiffMode,
          template <typename, int> class LossT = graphite::DefaultLoss>
struct BinaryFactorTraits {
  static constexpr size_t dimension = 1;
  using VertexDescriptors = std::tuple<Vec2Descriptor, Vec2Descriptor>;
  using Observation = float;
  using Data = graphite::Empty;
  using Loss = LossT<float, dimension>;
  using Differentiation = DiffMode;

  template <typename D>
  d_fn static void error(const D *v0, const D *v1, const Observation &obs,
                         D *residual) {
    residual[0] = v0[0] + static_cast<D>(2) * v0[1] +
                  static_cast<D>(3) * v1[0] + static_cast<D>(4) * v1[1] - obs;
  }

  template <typename D, size_t I>
  d_fn static void jacobian(const Vec2 &, const Vec2 &, const Observation &,
                            D *jacobian) {
    if constexpr (I == 0) {
      jacobian[0] = static_cast<D>(1);
      jacobian[1] = static_cast<D>(2);
    } else {
      jacobian[0] = static_cast<D>(3);
      jacobian[1] = static_cast<D>(4);
    }
  }
};

template <typename DiffMode,
          template <typename, int> class LossT = graphite::DefaultLoss>
struct Residual2FactorTraits {
  static constexpr size_t dimension = 2;
  using VertexDescriptors = std::tuple<Vec2Descriptor>;
  using Observation = Vec2;
  using Data = graphite::Empty;
  using Loss = LossT<float, dimension>;
  using Differentiation = DiffMode;

  template <typename D>
  d_fn static void error(const D *vertex, const Observation &obs, D *residual) {
    residual[0] = vertex[0] - static_cast<D>(obs.x);
    residual[1] = vertex[1] - static_cast<D>(obs.y);
  }

  template <typename D, size_t I>
  d_fn static void jacobian(const Vec2 &, const Observation &, D *jacobian) {
    // 2x2 identity in column-major storage for E=2, D=2
    jacobian[0] = static_cast<D>(1);
    jacobian[1] = static_cast<D>(0);
    jacobian[2] = static_cast<D>(0);
    jacobian[3] = static_cast<D>(1);
  }
};

using AutoFactor = graphite::FactorDescriptor<
    float, float, UnaryFactorTraits<graphite::DifferentiationMode::Auto>>;
using ManualFactor = graphite::FactorDescriptor<
    float, float, UnaryFactorTraits<graphite::DifferentiationMode::Manual>>;
using CoupledManualFactor = graphite::FactorDescriptor<
    float, float,
    CoupledUnaryFactorTraits<graphite::DifferentiationMode::Manual>>;
using BinaryManualFactor = graphite::FactorDescriptor<
    float, float, BinaryFactorTraits<graphite::DifferentiationMode::Manual>>;
using Residual2ManualFactor = graphite::FactorDescriptor<
    float, float, Residual2FactorTraits<graphite::DifferentiationMode::Manual>>;
using ManualHuberFactor = graphite::FactorDescriptor<
    float, float,
    UnaryFactorTraits<graphite::DifferentiationMode::Manual,
                      graphite::HuberLoss>>;

TEST(FactorDescriptorTests, UseAutodiffReflectsDifferentiationMode) {
  Vec2Descriptor vertex_desc;

  AutoFactor auto_factor(&vertex_desc);
  ManualFactor manual_factor(&vertex_desc);

  EXPECT_TRUE(auto_factor.use_autodiff());
  EXPECT_FALSE(manual_factor.use_autodiff());

  EXPECT_FALSE(auto_factor.supports_dynamic_jacobians());
  EXPECT_TRUE(manual_factor.supports_dynamic_jacobians());
}

TEST(FactorDescriptorTests, ComputeError) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.to_device();

  ManualFactor factor(&vertex_desc);
  factor.add_factor({10}, 2.5f);

  factor.initialize_device_ids(0);
  factor.to_device();

  factor.compute_error();

  thrust::host_vector<float> residuals = factor.residuals;
  ASSERT_EQ(residuals.size(), 1u);
  EXPECT_FLOAT_EQ(residuals[0], 7.0f - 2.5f);
}

TEST(FactorDescriptorTests, AddFactor) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{1.0f, 0.0f});
  vertices.push_back(Vec2{2.0f, 0.0f});
  vertices.push_back(Vec2{3.0f, 0.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.add_vertex(30, v0 + 2, false);

  ManualFactor factor(&vertex_desc);
  const auto f0 = factor.add_factor({10}, 1.5f);
  const auto f1 = factor.add_factor({20}, 2.5f);
  const auto f2 = factor.add_factor({30}, 3.5f);

  EXPECT_EQ(factor.internal_count(), 3u);
  EXPECT_EQ(factor.device_obs.size(), 3u);
  EXPECT_FLOAT_EQ(factor.device_obs[0], 1.5f);
  EXPECT_FLOAT_EQ(factor.device_obs[1], 2.5f);
  EXPECT_FLOAT_EQ(factor.device_obs[2], 3.5f);

  EXPECT_EQ(factor.get_vertex_ids(f0)[0], 10u);
  EXPECT_EQ(factor.get_vertex_ids(f1)[0], 20u);
  EXPECT_EQ(factor.get_vertex_ids(f2)[0], 30u);
}

TEST(FactorDescriptorTests, RemoveFactorFromBeginning) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{1.0f, 0.0f});
  vertices.push_back(Vec2{2.0f, 0.0f});
  vertices.push_back(Vec2{3.0f, 0.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.add_vertex(30, v0 + 2, false);

  ManualFactor factor(&vertex_desc);
  const auto f0 = factor.add_factor({10}, 1.5f);
  const auto f1 = factor.add_factor({20}, 2.5f);
  const auto f2 = factor.add_factor({30}, 3.5f);

  factor.to_device();

  factor.remove_factor(f0);

  EXPECT_EQ(factor.internal_count(), 2u);
  EXPECT_EQ(factor.get_vertex_ids(f1)[0], 20u);
  EXPECT_EQ(factor.get_vertex_ids(f2)[0], 30u);
}

TEST(FactorDescriptorTests, RemoveFactorFromMiddle) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{1.0f, 0.0f});
  vertices.push_back(Vec2{2.0f, 0.0f});
  vertices.push_back(Vec2{3.0f, 0.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.add_vertex(30, v0 + 2, false);

  ManualFactor factor(&vertex_desc);
  const auto f0 = factor.add_factor({10}, 1.5f);
  const auto f1 = factor.add_factor({20}, 2.5f);
  const auto f2 = factor.add_factor({30}, 3.5f);

  factor.to_device();

  factor.remove_factor(f1);

  EXPECT_EQ(factor.internal_count(), 2u);
  EXPECT_EQ(factor.get_vertex_ids(f0)[0], 10u);
  EXPECT_EQ(factor.get_vertex_ids(f2)[0], 30u);
}

TEST(FactorDescriptorTests, RemoveFactorFromEnd) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{1.0f, 0.0f});
  vertices.push_back(Vec2{2.0f, 0.0f});
  vertices.push_back(Vec2{3.0f, 0.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.add_vertex(30, v0 + 2, false);

  ManualFactor factor(&vertex_desc);
  const auto f0 = factor.add_factor({10}, 1.5f);
  const auto f1 = factor.add_factor({20}, 2.5f);
  const auto f2 = factor.add_factor({30}, 3.5f);

  factor.to_device();

  factor.remove_factor(f2);

  EXPECT_EQ(factor.internal_count(), 2u);
  EXPECT_EQ(factor.get_vertex_ids(f0)[0], 10u);
  EXPECT_EQ(factor.get_vertex_ids(f1)[0], 20u);
}

TEST(FactorDescriptorTests, RemoveAllFactors) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{1.0f, 0.0f});
  vertices.push_back(Vec2{2.0f, 0.0f});
  vertices.push_back(Vec2{3.0f, 0.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.add_vertex(30, v0 + 2, false);

  ManualFactor factor(&vertex_desc);
  const auto f0 = factor.add_factor({10}, 1.5f);
  const auto f1 = factor.add_factor({20}, 2.5f);
  const auto f2 = factor.add_factor({30}, 3.5f);

  factor.to_device();

  factor.remove_factor(f0);
  factor.remove_factor(f1);
  factor.remove_factor(f2);

  EXPECT_EQ(factor.internal_count(), 0u);

  factor.initialize_device_ids(0);
  EXPECT_EQ(factor.active_count(), 0u);
  EXPECT_TRUE(factor.host_ids.empty());
  EXPECT_TRUE(factor.device_ids.empty());
}

TEST(FactorDescriptorTests, ComputeErrorAutodiff) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.to_device();

  AutoFactor factor(&vertex_desc);
  factor.add_factor({10}, 2.5f);

  factor.initialize_device_ids(0);
  EXPECT_EQ(factor.active_count(), 1u);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_error_autodiff(streams);

  thrust::host_vector<float> residuals = factor.residuals;
  ASSERT_EQ(residuals.size(), 1u);
  EXPECT_FLOAT_EQ(residuals[0], 7.0f - 2.5f);

  thrust::host_vector<float> jacobian = factor.jacobians[0].data;
  ASSERT_EQ(jacobian.size(), 2u);
  EXPECT_FLOAT_EQ(jacobian[0], 1.0f);
  EXPECT_FLOAT_EQ(jacobian[1], 0.0f);
}

TEST(FactorDescriptorTests, FlagActiveVerticesAsync) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertices.push_back(Vec2{9.0f, 1.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.to_device();

  ManualFactor factor(&vertex_desc);
  const auto f0 = factor.add_factor({10}, 2.5f);
  const auto f1 = factor.add_factor({20}, 3.5f);
  factor.set_active(f1, 1); // inactive at optimization level 0

  factor.initialize_device_ids(0);
  ASSERT_EQ(factor.active_count(), 1u);

  factor.flag_active_vertices_async(0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto local0 = vertex_desc.get_global_map().at(10);
  const auto local1 = vertex_desc.get_global_map().at(20);
  const uint8_t *active_state = vertex_desc.get_active_state();

  EXPECT_EQ(active_state[local0] & 0x80, 0x80);
  EXPECT_EQ(active_state[local1] & 0x80, 0x00);

  // Ensure level-gated activity still works when selecting higher level.
  factor.flag_active_vertices_async(1);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(active_state[local1] & 0x80, 0x80);
  (void)f0;
}

TEST(FactorDescriptorTests, ComputeJacobians) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.to_device();

  ManualFactor factor(&vertex_desc);
  factor.add_factor({10}, 2.5f);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_jacobians(streams);

  thrust::host_vector<float> jacobian = factor.jacobians[0].data;
  ASSERT_EQ(jacobian.size(), 2u);
  EXPECT_FLOAT_EQ(jacobian[0], 1.0f);
  EXPECT_FLOAT_EQ(jacobian[1], 0.0f);
}

TEST(FactorDescriptorTests, ScaleJacobiansAsync) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.to_device();

  CoupledManualFactor factor(&vertex_desc);
  factor.add_factor({10}, 2.5f);
  factor.add_factor({10}, 2.5f);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_jacobians(streams);

  thrust::host_vector<float> jacobian_before = factor.jacobians[0].data;
  ASSERT_EQ(jacobian_before.size(), 4u);
  EXPECT_FLOAT_EQ(jacobian_before[0], 2.0f);
  EXPECT_FLOAT_EQ(jacobian_before[1], 3.0f);
  EXPECT_FLOAT_EQ(jacobian_before[2], 2.0f);
  EXPECT_FLOAT_EQ(jacobian_before[3], 3.0f);

  graphite::managed_vector<float> jacobian_scales;
  jacobian_scales.resize(2);
  jacobian_scales[0] = 2.0f;
  jacobian_scales[1] = 3.0f;

  factor.scale_jacobians_async(jacobian_scales.data().get());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<float> jacobian_after = factor.jacobians[0].data;
  ASSERT_EQ(jacobian_after.size(), 4u);
  EXPECT_FLOAT_EQ(jacobian_after[0], 4.0f);
  EXPECT_FLOAT_EQ(jacobian_after[1], 9.0f);
  EXPECT_FLOAT_EQ(jacobian_after[2], 4.0f);
  EXPECT_FLOAT_EQ(jacobian_after[3], 9.0f);
}

TEST(FactorDescriptorTests, ComputeB) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.to_device();

  ManualFactor factor(&vertex_desc);
  factor.add_factor({10}, 2.5f);
  factor.add_factor({10}, 2.5f);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_error();
  factor.compute_jacobians(streams);

  factor.compute_error();
  factor.chi2(); // populates chi2_derivative via configured loss

  graphite::managed_vector<float> b;
  graphite::managed_vector<float> jacobian_scales;
  b.resize(2);
  jacobian_scales.resize(2);
  const float b0_init = 3.0f;
  const float b1_init = -7.0f;
  b[0] = b0_init;
  b[1] = b1_init;
  jacobian_scales[0] = 1.0f;
  jacobian_scales[1] = 1.0f;

  factor.compute_b_async(b.data().get(), jacobian_scales.data().get());
  factor.compute_b_async(b.data().get(), jacobian_scales.data().get());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Two identical factors per call; two calls total => 4x single-factor update.
  EXPECT_FLOAT_EQ(b[0], b0_init - 4.0f * (7.0f - 2.5f));
  EXPECT_FLOAT_EQ(b[1], b1_init);
}

TEST(FactorDescriptorTests, ComputeBHuberLoss) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.to_device();

  ManualHuberFactor factor(&vertex_desc);
  const graphite::HuberLoss<float, 1> huber(1.0f);
  factor.add_factor({10}, 2.5f, nullptr, graphite::Empty{}, huber);
  factor.add_factor({10}, 2.5f, nullptr, graphite::Empty{}, huber);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_error();
  factor.compute_jacobians(streams);
  factor.chi2(); // populates chi2_derivative from configured loss function

  graphite::managed_vector<float> b;
  graphite::managed_vector<float> jacobian_scales;
  b.resize(2);
  jacobian_scales.resize(2);
  const float b0_init = 5.0f;
  const float b1_init = -11.0f;
  b[0] = b0_init;
  b[1] = b1_init;
  jacobian_scales[0] = 1.0f;
  jacobian_scales[1] = 1.0f;

  factor.compute_b_async(b.data().get(), jacobian_scales.data().get());
  factor.compute_b_async(b.data().get(), jacobian_scales.data().get());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // With Huber delta=1 and residual r=4.5, each factor contributes -1.
  // Two factors per call and two calls total => -4 from initial value.
  EXPECT_FLOAT_EQ(b[0], b0_init - 4.0f);
  EXPECT_FLOAT_EQ(b[1], b1_init);
}

TEST(FactorDescriptorTests, ComputeHessianBlockDiagonal) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.to_device();

  CoupledManualFactor factor(&vertex_desc);
  factor.add_factor({10}, 2.5f);
  factor.add_factor({10}, 2.5f);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_jacobians(streams);

  factor.compute_error();
  factor.chi2(); // populates chi2_derivative via configured loss

  graphite::managed_vector<float> jacobian_scales;
  jacobian_scales.resize(2);
  jacobian_scales[0] = 1.0f;
  jacobian_scales[1] = 1.0f;

  std::unordered_map<graphite::BaseVertexDescriptor<float, float> *,
                     thrust::device_vector<float>>
      block_diagonals;
  block_diagonals[&vertex_desc].resize(4, 0.0f);

  factor.compute_hessian_block_diagonal_async(block_diagonals,
                                              jacobian_scales.data().get(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<float> h_block = block_diagonals[&vertex_desc];
  ASSERT_EQ(h_block.size(), 4u);
  // Column-major 2x2 block for one vertex: [h00, h10, h01, h11].
  // Two factors, J=[2,3], P=I => 2 * (J^T J) = [[8, 12], [12, 18]].
  EXPECT_FLOAT_EQ(h_block[0], 8.0f);
  EXPECT_FLOAT_EQ(h_block[1], 12.0f);
  EXPECT_FLOAT_EQ(h_block[2], 12.0f);
  EXPECT_FLOAT_EQ(h_block[3], 18.0f);
}

TEST(FactorDescriptorTests, ComputeHessianScalarDiagonal) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.to_device();

  CoupledManualFactor factor(&vertex_desc);
  factor.add_factor({10}, 2.5f);
  factor.add_factor({10}, 2.5f);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_jacobians(streams);

  factor.compute_error();
  factor.chi2(); // populates chi2_derivative via configured loss

  graphite::managed_vector<float> diagonal;
  graphite::managed_vector<float> jacobian_scales;
  diagonal.resize(2);
  jacobian_scales.resize(2);
  diagonal[0] = 0.0f;
  diagonal[1] = 0.0f;
  jacobian_scales[0] = 1.0f;
  jacobian_scales[1] = 1.0f;

  factor.compute_hessian_scalar_diagonal_async(diagonal.data().get(),
                                               jacobian_scales.data().get());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // One vertex (dim=2), two factors, J=[2,3] and P=I => diagonal [8, 18].
  EXPECT_FLOAT_EQ(diagonal[0], 8.0f);
  EXPECT_FLOAT_EQ(diagonal[1], 18.0f);
}

TEST(FactorDescriptorTests, ComputeJvHuberLoss) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.to_device();

  ManualHuberFactor factor(&vertex_desc);
  const graphite::HuberLoss<float, 1> huber(1.0f);
  const auto f0 =
      factor.add_factor({10}, 2.5f, nullptr, graphite::Empty{}, huber);
  const auto f1 =
      factor.add_factor({10}, 2.5f, nullptr, graphite::Empty{}, huber);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_jacobians(streams);

  graphite::managed_vector<float> out;
  graphite::managed_vector<float> in;
  graphite::managed_vector<float> jacobian_scales;
  out.resize(2);
  in.resize(2);
  jacobian_scales.resize(2);

  out[0] = 0.0f;
  out[1] = 0.0f;
  in[0] = 3.0f;
  in[1] = 5.0f;
  jacobian_scales[0] = 1.0f;
  jacobian_scales[1] = 1.0f;

  factor.compute_Jv(out.data().get(), in.data().get(),
                    jacobian_scales.data().get(), streams);

  // J = [1, 0], so each factor output is J*x = in[0].
  EXPECT_FLOAT_EQ(out[0], in[0]);
  EXPECT_FLOAT_EQ(out[1], in[0]);

  // Fixed vertex: output should be unchanged.
  vertex_desc.set_fixed(10, true);
  out[0] = 17.0f;
  out[1] = 23.0f;
  factor.compute_Jv(out.data().get(), in.data().get(),
                    jacobian_scales.data().get(), streams);
  EXPECT_FLOAT_EQ(out[0], 17.0f);
  EXPECT_FLOAT_EQ(out[1], 23.0f);

  // Inactive (non-fixed) vertex via MSB flag: output should be unchanged.
  vertex_desc.set_fixed(10, false);
  vertex_desc.get_active_state()[0] = 0x80;
  out[0] = 29.0f;
  out[1] = 31.0f;
  factor.compute_Jv(out.data().get(), in.data().get(),
                    jacobian_scales.data().get(), streams);
  EXPECT_FLOAT_EQ(out[0], 29.0f);
  EXPECT_FLOAT_EQ(out[1], 31.0f);

  // Restore active vertex state.
  vertex_desc.get_active_state()[0] = 0;

  // Inactive factors: output should be unchanged.
  factor.set_active(f0, 1);
  factor.set_active(f1, 1);
  factor.initialize_device_ids(0);
  EXPECT_EQ(factor.active_count(), 0u);
  out[0] = 37.0f;
  out[1] = 41.0f;
  factor.compute_Jv(out.data().get(), in.data().get(),
                    jacobian_scales.data().get(), streams);
  EXPECT_FLOAT_EQ(out[0], 37.0f);
  EXPECT_FLOAT_EQ(out[1], 41.0f);
}

TEST(FactorDescriptorTests, ComputeJtvHuberLoss) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.to_device();

  ManualHuberFactor factor(&vertex_desc);
  const graphite::HuberLoss<float, 1> huber(1.0f);
  const auto f0 =
      factor.add_factor({10}, 2.5f, nullptr, graphite::Empty{}, huber);
  const auto f1 =
      factor.add_factor({10}, 2.5f, nullptr, graphite::Empty{}, huber);

  factor.initialize_device_ids(0);
  EXPECT_EQ(factor.active_count(), 2u);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(1);
  factor.compute_error();
  factor.compute_jacobians(streams);
  factor.chi2(); // populates chi2_derivative using Huber loss derivative

  graphite::managed_vector<float> out;
  graphite::managed_vector<float> in;
  graphite::managed_vector<float> jacobian_scales;
  out.resize(2);
  in.resize(2);
  jacobian_scales.resize(2);

  out[0] = 0.0f;
  out[1] = 0.0f;
  in[0] = 9.0f;
  in[1] = 9.0f;
  jacobian_scales[0] = 1.0f;
  jacobian_scales[1] = 1.0f;

  factor.compute_Jtv(out.data().get(), in.data().get(),
                     jacobian_scales.data().get(), streams);

  // For each factor: J^T * P * x = [in_i, 0].
  // Huber delta=1 with residual 4.5 gives dL = 1/4.5, so each contributes
  // [9*(1/4.5), 0] = [2, 0]. Two factors => [4, 0].
  EXPECT_FLOAT_EQ(out[0], 4.0f);
  EXPECT_FLOAT_EQ(out[1], 0.0f);

  // Fixed vertex: output should be unchanged.
  vertex_desc.set_fixed(10, true);
  out[0] = 43.0f;
  out[1] = 47.0f;
  factor.compute_Jtv(out.data().get(), in.data().get(),
                     jacobian_scales.data().get(), streams);
  EXPECT_FLOAT_EQ(out[0], 43.0f);
  EXPECT_FLOAT_EQ(out[1], 47.0f);

  // Inactive (non-fixed) vertex via MSB flag: output should be unchanged.
  vertex_desc.set_fixed(10, false);
  vertex_desc.get_active_state()[0] = 0x80;
  out[0] = 53.0f;
  out[1] = 59.0f;
  factor.compute_Jtv(out.data().get(), in.data().get(),
                     jacobian_scales.data().get(), streams);
  EXPECT_FLOAT_EQ(out[0], 53.0f);
  EXPECT_FLOAT_EQ(out[1], 59.0f);

  // Restore active vertex state.
  vertex_desc.get_active_state()[0] = 0;

  // Inactive factors: output should be unchanged.
  factor.set_active(f0, 1);
  factor.set_active(f1, 1);
  factor.initialize_device_ids(0);
  EXPECT_EQ(factor.active_count(), 0u);
  out[0] = 61.0f;
  out[1] = 67.0f;
  factor.compute_Jtv(out.data().get(), in.data().get(),
                     jacobian_scales.data().get(), streams);
  EXPECT_FLOAT_EQ(out[0], 61.0f);
  EXPECT_FLOAT_EQ(out[1], 67.0f);
}

TEST(FactorDescriptorTests, Chi2HuberLoss) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 0.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.to_device();

  ManualHuberFactor factor(&vertex_desc);
  const graphite::HuberLoss<float, 1> huber(1.0f);
  const auto f0 =
      factor.add_factor({10}, 2.5f, nullptr, graphite::Empty{}, huber);
  const auto f1 =
      factor.add_factor({10}, 6.5f, nullptr, graphite::Empty{}, huber);

  factor.initialize_device_ids(0);
  EXPECT_EQ(factor.active_count(), 2u);
  factor.to_device();

  factor.compute_error();

  // residuals are 4.5 and 0.5, with delta=1.0:
  // rho(4.5^2) = 2*4.5 - 1 = 8, rho(0.5^2) = 0.25
  const float total_chi2 = factor.chi2();
  EXPECT_FLOAT_EQ(total_chi2, 8.25f);
  EXPECT_FLOAT_EQ(factor.chi2(f0), 8.0f);
  EXPECT_FLOAT_EQ(factor.chi2(f1), 0.25f);
}

TEST(FactorDescriptorTests, GetDefaultPrecisionMatrix) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.push_back(Vec2{7.0f, 5.0f});
  vertex_desc.add_vertex(10, vertices.data().get(), false);
  vertex_desc.to_device();

  Residual2ManualFactor factor(&vertex_desc);
  factor.add_factor({10}, test_types::Vec2{1.0f, 2.0f});

  ASSERT_EQ(factor.precision_matrices.size(), 4u);
  EXPECT_FLOAT_EQ(factor.precision_matrices[0], 1.0f);
  EXPECT_FLOAT_EQ(factor.precision_matrices[1], 0.0f);
  EXPECT_FLOAT_EQ(factor.precision_matrices[2], 0.0f);
  EXPECT_FLOAT_EQ(factor.precision_matrices[3], 1.0f);
}

TEST(FactorDescriptorTests, ClearResetsStorage) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{7.0f, 5.0f});
  vertices.push_back(Vec2{11.0f, 13.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.to_device();

  BinaryManualFactor factor(&vertex_desc, &vertex_desc);
  factor.add_factor({10, 20}, 2.5f);
  factor.add_factor({20, 10}, 3.5f);

  factor.initialize_device_ids(0);
  factor.to_device();
  factor.initialize_jacobian_storage();

  graphite::StreamPool streams(2);
  factor.compute_error();
  factor.compute_jacobians(streams);
  factor.chi2();

  ASSERT_GT(factor.internal_count(), 0u);
  ASSERT_GT(factor.active_count(), 0u);
  ASSERT_FALSE(factor.host_ids.empty());
  ASSERT_FALSE(factor.device_ids.empty());
  ASSERT_GT(factor.device_obs.size(), 0u);
  ASSERT_FALSE(factor.residuals.empty());

  factor.clear();

  EXPECT_EQ(factor.internal_count(), 0u);
  EXPECT_EQ(factor.active_count(), 0u);
  EXPECT_TRUE(factor.host_ids.empty());
  EXPECT_TRUE(factor.device_ids.empty());
  EXPECT_EQ(factor.device_obs.size(), 0u);
  EXPECT_TRUE(factor.residuals.empty());
  EXPECT_EQ(factor.precision_matrices.size(), 0u);
  EXPECT_EQ(factor.data.size(), 0u);
  EXPECT_EQ(factor.chi2_vec.size(), 0u);
  EXPECT_TRUE(factor.chi2_derivative.empty());
  EXPECT_EQ(factor.loss.size(), 0u);
  EXPECT_TRUE(factor.active.empty());
  EXPECT_TRUE(factor.device_active.empty());
  EXPECT_TRUE(factor.active_indices.empty());
  EXPECT_TRUE(factor.jacobians[0].data.empty());
  EXPECT_TRUE(factor.jacobians[1].data.empty());
}

TEST(FactorDescriptorTests, ComputeHessian) {
  Vec2Descriptor vertex_desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{7.0f, 5.0f});
  vertices.push_back(Vec2{11.0f, 13.0f});
  Vec2 *v0 = vertices.data().get();
  vertex_desc.add_vertex(10, v0, false);
  vertex_desc.add_vertex(20, v0 + 1, false);
  vertex_desc.set_hessian_column(10, 0, 0);
  vertex_desc.set_hessian_column(20, 2, 1);
  vertex_desc.to_device();

  CoupledManualFactor unary_factor(&vertex_desc);
  BinaryManualFactor binary_factor(&vertex_desc, &vertex_desc);

  unary_factor.add_factor({10}, 2.5f);
  unary_factor.add_factor({20}, 3.5f);
  binary_factor.add_factor({10, 20}, 4.5f);

  unary_factor.initialize_device_ids(0);
  binary_factor.initialize_device_ids(0);
  EXPECT_EQ(unary_factor.active_count(), 2u);
  EXPECT_EQ(binary_factor.active_count(), 1u);

  unary_factor.to_device();
  binary_factor.to_device();
  unary_factor.initialize_jacobian_storage();
  binary_factor.initialize_jacobian_storage();

  graphite::StreamPool streams(2);
  unary_factor.compute_jacobians(streams);
  binary_factor.compute_jacobians(streams);

  unary_factor.compute_error();
  binary_factor.compute_error();
  unary_factor.chi2();  // populates chi2_derivative via configured loss
  binary_factor.chi2(); // populates chi2_derivative via configured loss

  thrust::device_vector<graphite::BlockCoordinates> block_coords;
  unary_factor.get_hessian_block_coordinates(block_coords);
  binary_factor.get_hessian_block_coordinates(block_coords);

  thrust::host_vector<graphite::BlockCoordinates> h_block_coords = block_coords;
  ASSERT_EQ(h_block_coords.size(), 5u);
  size_t n00 = 0;
  size_t n01 = 0;
  size_t n11 = 0;
  for (const auto &coord : h_block_coords) {
    if (coord.row == 0u && coord.col == 0u) {
      n00++;
    } else if (coord.row == 0u && coord.col == 1u) {
      n01++;
    } else if (coord.row == 1u && coord.col == 1u) {
      n11++;
    }
  }
  EXPECT_EQ(n00, 2u);
  EXPECT_EQ(n01, 1u);
  EXPECT_EQ(n11, 2u);

  std::unordered_map<graphite::BlockCoordinates, size_t> block_indices;
  block_indices[graphite::BlockCoordinates{0, 0}] = 0;
  block_indices[graphite::BlockCoordinates{0, 1}] = 4;
  block_indices[graphite::BlockCoordinates{1, 1}] = 8;

  thrust::device_vector<float> d_hessian(12, 0.0f);
  std::vector<size_t> h_block_offsets(5, static_cast<size_t>(-1));

  size_t mul_count = 0;
  mul_count += unary_factor.setup_hessian_computation(
      block_indices, d_hessian, h_block_offsets.data() + mul_count, streams);
  mul_count += binary_factor.setup_hessian_computation(
      block_indices, d_hessian, h_block_offsets.data() + mul_count, streams);

  ASSERT_EQ(mul_count, 5u);
  EXPECT_EQ(h_block_offsets[0], 0u);
  EXPECT_EQ(h_block_offsets[1], 8u);
  EXPECT_EQ(h_block_offsets[2], 0u);
  EXPECT_EQ(h_block_offsets[3], 4u);
  EXPECT_EQ(h_block_offsets[4], 8u);

  thrust::device_vector<size_t> d_block_offsets = h_block_offsets;

  thrust::fill(d_hessian.begin(), d_hessian.end(), 0.0f);

  size_t exec_mul_count = 0;
  exec_mul_count += unary_factor.execute_hessian_computation(
      block_indices, d_hessian, d_block_offsets.data().get() + exec_mul_count,
      streams);
  exec_mul_count += binary_factor.execute_hessian_computation(
      block_indices, d_hessian, d_block_offsets.data().get() + exec_mul_count,
      streams);

  ASSERT_EQ(exec_mul_count, 5u);

  thrust::host_vector<float> hessian = d_hessian;
  ASSERT_EQ(hessian.size(), 12u);
  // block(0,0): unary(v0) + binary(v0,v1)
  EXPECT_FLOAT_EQ(hessian[0], 5.0f);
  EXPECT_FLOAT_EQ(hessian[1], 8.0f);
  EXPECT_FLOAT_EQ(hessian[2], 8.0f);
  EXPECT_FLOAT_EQ(hessian[3], 13.0f);
  // block(0,1): binary(v0,v1)
  EXPECT_FLOAT_EQ(hessian[4], 3.0f);
  EXPECT_FLOAT_EQ(hessian[5], 6.0f);
  EXPECT_FLOAT_EQ(hessian[6], 4.0f);
  EXPECT_FLOAT_EQ(hessian[7], 8.0f);
  // block(1,1): unary(v1) + binary(v0,v1)
  EXPECT_FLOAT_EQ(hessian[8], 13.0f);
  EXPECT_FLOAT_EQ(hessian[9], 18.0f);
  EXPECT_FLOAT_EQ(hessian[10], 18.0f);
  EXPECT_FLOAT_EQ(hessian[11], 25.0f);
}

} // namespace
