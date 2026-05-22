#include "common_types.hpp"
#include <gtest/gtest.h>

namespace {
using test_types::Vec2;
using test_types::Vec2Descriptor;
using test_types::Vec2StateDescriptor;

TEST(VertexDescriptorTests, DimensionMatchesTraits) {
  Vec2Descriptor desc;
  Vec2StateDescriptor state_desc;

  EXPECT_EQ(desc.dimension(), 2u);
  EXPECT_EQ(state_desc.dimension(), 2u);
}

TEST(VertexDescriptorTests, FixedActiveState) {
  Vec2Descriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{0.0f, 0.0f});
  vertices.push_back(Vec2{1.0f, 1.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;

  desc.add_vertex(10, v0, false);
  desc.add_vertex(20, v1, true);

  EXPECT_TRUE(desc.exists(10));
  EXPECT_TRUE(desc.exists(20));
  EXPECT_FALSE(desc.exists(30));

  EXPECT_FALSE(desc.is_fixed(10));
  EXPECT_TRUE(desc.is_active(10));

  EXPECT_TRUE(desc.is_fixed(20));
  EXPECT_FALSE(desc.is_active(20));

  desc.set_fixed(10, true);
  desc.set_fixed(20, false);

  EXPECT_TRUE(desc.is_fixed(10));
  EXPECT_FALSE(desc.is_active(10));

  EXPECT_FALSE(desc.is_fixed(20));
  EXPECT_TRUE(desc.is_active(20));

  EXPECT_EQ(desc.get_vertex(10), v0);
  EXPECT_EQ(desc.get_vertex(20), v1);
}

TEST(VertexDescriptorTests, ReplaceVertex) {
  Vec2Descriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{1.0f, 1.0f});
  vertices.push_back(Vec2{2.0f, 2.0f});
  vertices.push_back(Vec2{9.0f, 9.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;
  Vec2 *replacement = v0 + 2;

  desc.add_vertex(10, v0, false);
  desc.add_vertex(20, v1, true);

  desc.replace_vertex(20, replacement);

  auto *updated = desc.get_vertex(20);
  ASSERT_EQ(updated, replacement);
  EXPECT_FLOAT_EQ(updated->x, 9.0f);
  EXPECT_FLOAT_EQ(updated->y, 9.0f);
}

TEST(VertexDescriptorTests, ApplyUpdate) {
  Vec2Descriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{1.0f, 2.0f});
  vertices.push_back(Vec2{10.0f, 20.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;

  desc.add_vertex(10, v0, false); // active
  desc.add_vertex(20, v1, true);  // fixed/inactive

  desc.set_hessian_column(10, 0, 0);
  desc.set_hessian_column(20, 2, 1);
  desc.to_device();

  graphite::managed_vector<float> delta_x;
  graphite::managed_vector<float> jacobian_scales;
  delta_x.resize(4);
  jacobian_scales.resize(4);

  delta_x[0] = 4.0f;
  delta_x[1] = -2.0f;
  delta_x[2] = 100.0f;
  delta_x[3] = 200.0f;

  jacobian_scales[0] = 0.5f;
  jacobian_scales[1] = 2.0f;
  jacobian_scales[2] = 1.0f;
  jacobian_scales[3] = 1.0f;

  desc.apply_update_async(delta_x.data().get(), jacobian_scales.data().get(),
                          0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Active vertex receives scaled update.
  EXPECT_FLOAT_EQ(v0->x, 1.0f + delta_x[0] * jacobian_scales[0]);
  EXPECT_FLOAT_EQ(v0->y, 2.0f + delta_x[1] * jacobian_scales[1]);

  // Fixed/inactive vertex is not updated.
  EXPECT_FLOAT_EQ(v1->x, 10.0f);
  EXPECT_FLOAT_EQ(v1->y, 20.0f);
}

TEST(VertexDescriptorTests, AugmentBlockDiagonal) {
  Vec2Descriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{0.0f, 0.0f});
  vertices.push_back(Vec2{0.0f, 0.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;

  desc.add_vertex(10, v0, false); // active
  desc.add_vertex(20, v1, true);  // fixed/inactive

  graphite::managed_vector<float> block_diagonal;
  graphite::managed_vector<float> scalar_diagonal;
  block_diagonal.resize(8);  // 2 vertices * (2x2 blocks)
  scalar_diagonal.resize(4); // 2 vertices * dim(2)

  for (size_t i = 0; i < block_diagonal.size(); ++i) {
    block_diagonal[i] = -1.0f;
  }

  scalar_diagonal[0] = 2.0f;
  scalar_diagonal[1] = 4.0f;
  scalar_diagonal[2] = 8.0f;
  scalar_diagonal[3] = 16.0f;

  const float mu = 0.5f;
  desc.augment_block_diagonal_async(block_diagonal.data().get(),
                                    scalar_diagonal.data().get(), mu, 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Active vertex (block 0): only diagonal entries are updated.
  EXPECT_FLOAT_EQ(block_diagonal[0],
                  scalar_diagonal[0] + mu * scalar_diagonal[0]);
  EXPECT_FLOAT_EQ(block_diagonal[1], -1.0f);
  EXPECT_FLOAT_EQ(block_diagonal[2], -1.0f);
  EXPECT_FLOAT_EQ(block_diagonal[3],
                  scalar_diagonal[1] + mu * scalar_diagonal[1]);

  // Fixed/inactive vertex (block 1): untouched.
  EXPECT_FLOAT_EQ(block_diagonal[4], -1.0f);
  EXPECT_FLOAT_EQ(block_diagonal[5], -1.0f);
  EXPECT_FLOAT_EQ(block_diagonal[6], -1.0f);
  EXPECT_FLOAT_EQ(block_diagonal[7], -1.0f);
}

TEST(VertexDescriptorTests, ApplyBlockJacobi) {
  Vec2Descriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{0.0f, 0.0f});
  vertices.push_back(Vec2{0.0f, 0.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;

  desc.add_vertex(10, v0, false); // active
  desc.add_vertex(20, v1, true);  // fixed/inactive

  desc.set_hessian_column(10, 0, 0);
  desc.set_hessian_column(20, 2, 1);
  desc.to_device();

  graphite::managed_vector<float> z;
  graphite::managed_vector<float> r;
  graphite::managed_vector<float> block_diagonal;
  z.resize(4);
  r.resize(4);
  block_diagonal.resize(8); // 2 vertices * (2x2 blocks)

  // Output before applying preconditioner.
  for (size_t i = 0; i < z.size(); ++i) {
    z[i] = -5.0f;
  }

  // r for active vertex (offset 0), fixed vertex (offset 2).
  r[0] = 11.0f;
  r[1] = 13.0f;
  r[2] = 17.0f;
  r[3] = 19.0f;

  // Block 0 (active) in column-major: [2 5; 3 7]
  block_diagonal[0] = 2.0f;
  block_diagonal[1] = 3.0f;
  block_diagonal[2] = 5.0f;
  block_diagonal[3] = 7.0f;

  // Block 1 (fixed) should not be used due to inactive state.
  block_diagonal[4] = 101.0f;
  block_diagonal[5] = 103.0f;
  block_diagonal[6] = 107.0f;
  block_diagonal[7] = 109.0f;

  desc.apply_block_jacobi(z.data().get(), r.data().get(),
                          block_diagonal.data().get(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // z = block * r for active vertex.
  EXPECT_FLOAT_EQ(z[0], block_diagonal[0] * r[0] + block_diagonal[2] * r[1]);
  EXPECT_FLOAT_EQ(z[1], block_diagonal[1] * r[0] + block_diagonal[3] * r[1]);

  // Fixed/inactive vertex output is untouched.
  EXPECT_FLOAT_EQ(z[2], -5.0f);
  EXPECT_FLOAT_EQ(z[3], -5.0f);
}

void assert_remove_vertex_case(const size_t id_to_remove) {
  Vec2Descriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{0.0f, 0.0f});
  vertices.push_back(Vec2{1.0f, 1.0f});
  vertices.push_back(Vec2{2.0f, 2.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;
  Vec2 *v2 = v0 + 2;

  desc.add_vertex(10, v0, false);
  desc.add_vertex(20, v1, false);
  desc.add_vertex(30, v2, true);

  ASSERT_EQ(desc.count(), 3u);
  ASSERT_TRUE(desc.exists(10));
  ASSERT_TRUE(desc.exists(20));
  ASSERT_TRUE(desc.exists(30));

  desc.remove_vertex(id_to_remove);

  EXPECT_EQ(desc.count(), 2u);
  EXPECT_EQ(desc.exists(10), id_to_remove != 10);
  EXPECT_EQ(desc.exists(20), id_to_remove != 20);
  EXPECT_EQ(desc.exists(30), id_to_remove != 30);

  if (id_to_remove != 10) {
    EXPECT_EQ(desc.get_vertex(10), v0);
    EXPECT_FALSE(desc.is_fixed(10));
    EXPECT_TRUE(desc.is_active(10));
  }

  if (id_to_remove != 20) {
    EXPECT_EQ(desc.get_vertex(20), v1);
    EXPECT_FALSE(desc.is_fixed(20));
    EXPECT_TRUE(desc.is_active(20));
  }

  if (id_to_remove != 30) {
    EXPECT_EQ(desc.get_vertex(30), v2);
    EXPECT_TRUE(desc.is_fixed(30));
    EXPECT_FALSE(desc.is_active(30));
  }

  const auto &map = desc.get_global_map();
  ASSERT_EQ(map.size(), 2u);
  if (id_to_remove != 10) {
    EXPECT_LT(map.at(10), desc.count());
  }
  if (id_to_remove != 20) {
    EXPECT_LT(map.at(20), desc.count());
  }
  if (id_to_remove != 30) {
    EXPECT_LT(map.at(30), desc.count());
  }
}

TEST(VertexDescriptorTests, RemoveVertexFromStart) {
  assert_remove_vertex_case(10);
}

TEST(VertexDescriptorTests, RemoveVertexFromMiddle) {
  assert_remove_vertex_case(20);
}

TEST(VertexDescriptorTests, RemoveVertexFromEnd) {
  assert_remove_vertex_case(30);
}

TEST(VertexDescriptorTests, BackupRestoreState) {
  Vec2StateDescriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(3);
  vertices.push_back(Vec2{1.0f, 2.0f});
  vertices.push_back(Vec2{3.0f, 4.0f});
  vertices.push_back(Vec2{5.0f, 6.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;
  Vec2 *v2 = v0 + 2;

  desc.add_vertex(10, v0, false); // active
  desc.add_vertex(20, v1, true);  // fixed/inactive
  desc.add_vertex(30, v2, false); // active

  desc.to_device();
  desc.backup_parameters_async();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Mutate both fields.
  v0->x = 11.0f;
  v0->y = 12.0f;

  v1->x = 13.0f;
  v1->y = 14.0f;

  v2->x = 15.0f;
  v2->y = 16.0f;

  desc.restore_parameters_async();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Active vertices restore only State (x). y stays modified.
  EXPECT_FLOAT_EQ(v0->x, 1.0f);
  EXPECT_FLOAT_EQ(v0->y, 12.0f);
  EXPECT_FLOAT_EQ(v2->x, 5.0f);
  EXPECT_FLOAT_EQ(v2->y, 16.0f);

  // Fixed/inactive vertex is not restored at all.
  EXPECT_FLOAT_EQ(v1->x, 13.0f);
  EXPECT_FLOAT_EQ(v1->y, 14.0f);
}

TEST(VertexDescriptorTests, Clear) {
  Vec2Descriptor desc;
  graphite::managed_vector<Vec2> vertices;
  vertices.reserve(2);
  vertices.push_back(Vec2{7.0f, 8.0f});
  vertices.push_back(Vec2{9.0f, 10.0f});

  Vec2 *v0 = vertices.data().get();
  Vec2 *v1 = v0 + 1;

  desc.add_vertex(10, v0, false);
  desc.add_vertex(20, v1, true);

  ASSERT_EQ(desc.count(), 2u);
  ASSERT_FALSE(desc.get_global_map().empty());
  ASSERT_TRUE(desc.exists(10));
  ASSERT_TRUE(desc.exists(20));

  desc.clear();

  EXPECT_EQ(desc.count(), 0u);
  EXPECT_TRUE(desc.get_global_map().empty());
  EXPECT_FALSE(desc.exists(10));
  EXPECT_FALSE(desc.exists(20));
}

} // namespace
