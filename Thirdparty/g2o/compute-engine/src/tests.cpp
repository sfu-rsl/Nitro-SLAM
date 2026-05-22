#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>
#include <random>
#include <solver/backend.hpp>
#include <solver/pcg.hpp>
#include <solver/sparse_block_matrix.hpp>
#include <solver/utility.hpp>

using namespace compute;
ComputeEngine engine;

std::shared_ptr<SyncObject> create_sync_object() {
  return std::make_shared<SyncObject>(engine.create_op_sequence(),
                                      engine.create_op_sequence());
}

TEST(SparseBlockMatrix, Reserve) {
  // ComputeEngine engine;
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);

  matrix->resize({0, 6, 15}, {0, 3, 7});
  matrix->reserve_block(0, 0);
  matrix->reserve_block(1, 1);
  matrix->allocate_memory();

  BlockDim dim1{6, 3};
  auto b1 = matrix->get_block_ptr(0, 0);
  ASSERT_NE(b1, nullptr);
  ASSERT_EQ(matrix->get_block_dim(0, 0), dim1);

  BlockDim dim2{9, 4};
  auto b2 = matrix->get_block_ptr(1, 1);
  ASSERT_NE(b2, nullptr);
  ASSERT_EQ(matrix->get_block_dim(1, 1), dim2);

  ASSERT_EQ(matrix->get_block_ptr(0, 1), nullptr);
  ASSERT_EQ(matrix->get_block_ptr(1, 0), nullptr);
}

TEST(SparseBlockMatrix, SubgroupMultiplyAdd1) {
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize({0, 3, 6}, {0, 3, 6});
    matrix->reserve_block(0, 0);
    matrix->reserve_block(1, 1);
    matrix->allocate_memory();

    auto b1 = matrix->get_block_ptr(0, 0);
    auto m1 = Eigen::Map<Eigen::Matrix<double, 3, 3>>(b1);
    m1.setIdentity();

    auto b2 = matrix->get_block_ptr(1, 1);
    auto m2 = Eigen::Map<Eigen::Matrix<double, 3, 3>>(b2);
    m2.setIdentity();
  }
  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize({0, 3, 6}, {0, 3, 6});
    right->reserve_block(0, 0);
    right->reserve_block(1, 1);
    right->allocate_memory();

    auto b1 = right->get_block_ptr(0, 0);
    auto m1 = Eigen::Map<Eigen::Matrix<double, 3, 3>>(b1);
    m1.setIdentity();
    m1 *= 2.0;

    auto b2 = right->get_block_ptr(1, 1);
    auto m2 = Eigen::Map<Eigen::Matrix<double, 3, 3>>(b2);
    m2.setIdentity();
    m2 *= 2.0;
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest->resize(matrix->get_rows(), right->get_cols());
  // Reserve destination pattern (must be manually determined)
  dest->reserve_block(0, 0);
  dest->reserve_block(1, 1);
  dest->allocate_memory();
  dest->zero_memory();
  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_multiply_add(mult, dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  auto n = 1;
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), matrix->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_cols());

  BlockDim dim1{3, 3};
  auto b1 = dest->get_block_ptr(0, 0);
  ASSERT_NE(b1, nullptr);
  ASSERT_EQ(dest->get_block_dim(0, 0), dim1);

  BlockDim dim2{3, 3};
  auto b2 = dest->get_block_ptr(1, 1);
  ASSERT_NE(b2, nullptr);
  ASSERT_EQ(dest->get_block_dim(1, 1), dim2);

  auto m1 = Eigen::Map<Eigen::Matrix<double, 3, 3>>(b1);
  auto m2 = Eigen::Map<Eigen::Matrix<double, 3, 3>>(b2);
  // std::cout << "m1:\n" << m1 << std::endl;
  ASSERT_TRUE(m1.isApprox(m1.Identity() * 2.0 * n));
  // std::cout << "m2:\n" << m2 << std::endl;
  ASSERT_TRUE(m2.isApprox(m2.Identity() * 2.0 * n));
  dest->zero_memory();
  ASSERT_TRUE(m1.isZero());
  ASSERT_TRUE(m2.isZero());
}

TEST(SparseBlockMatrix, SubgroupMultiplyAdd2) {
  const int sz = 1000;
  const int d = 3;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setRandom();
    }
  }
  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      right->reserve_block(i, i);
    }
    right->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = right->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
      m1 *= 2.0;
    }
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    dest->resize(matrix->get_rows(), right->get_cols());
    // Reserve destination pattern (must be manually determined)
    for (int i = 0; i < sz; i++) {
      dest->reserve_block(i, i);
    }
    dest->allocate_memory();
    dest->zero_memory();
  }

  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_multiply_add(mult, dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), matrix->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_cols());

  for (int i = 0; i < sz; i++) {
    auto b1 = dest->get_block_ptr(i, i);
    ASSERT_NE(b1, nullptr);
    ASSERT_EQ(dest->get_block_dim(0, 0), BlockDim(d, d));
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    auto m0 =
        Eigen::Map<Eigen::Matrix<double, d, d>>(matrix->get_block_ptr(i, i));
    ASSERT_EQ(m1, m0 * 2.0);
  }

  // repeat and check if it doubled
  mult->execute();
  for (int i = 0; i < sz; i++) {
    auto b1 = dest->get_block_ptr(i, i);
    ASSERT_NE(b1, nullptr);
    ASSERT_EQ(dest->get_block_dim(0, 0), BlockDim(d, d));
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    auto m0 =
        Eigen::Map<Eigen::Matrix<double, d, d>>(matrix->get_block_ptr(i, i));
    ASSERT_EQ(m1, m0 * 4.0);
  }

  dest->zero_memory();
  for (int i = 0; i < sz; i++) {
    auto b1 = dest->get_block_ptr(i, i);
    ASSERT_NE(b1, nullptr);
    ASSERT_EQ(dest->get_block_dim(0, 0), BlockDim(d, d));
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    ASSERT_TRUE(m1.isZero());
  }
}

TEST(SparseBlockMatrix, SubgroupMultiplyAdd3) {
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize({0, 2, 4}, {0, 2, 4, 7});

    // block row 0
    matrix->reserve_block(0, 0);
    matrix->reserve_block(0, 1);
    matrix->reserve_block(0, 2);

    // block row 1
    matrix->reserve_block(1, 0);
    matrix->reserve_block(1, 1);
    matrix->reserve_block(1, 2);

    matrix->allocate_memory();

    for (int row = 0; row < 2; row++) {
      auto a1 = matrix->get_block_map(row, 0);
      a1 << 1, 2, 3, 4;

      auto a2 = matrix->get_block_map(row, 1);
      a2.setOnes();

      auto a3 = matrix->get_block_map(row, 2);
      a3.setOnes();
      a3 *= 2.0;
    }
  }

  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize({0, 2, 4, 7}, {0, 2});
    right->reserve_block(0, 0);
    right->reserve_block(1, 0);
    right->reserve_block(2, 0);
    right->allocate_memory();

    auto b1 = right->get_block_map(0, 0);
    b1.setIdentity();

    auto b2 = right->get_block_map(1, 0);
    b2.setIdentity();
    b2 *= 2.0;

    auto b3 = right->get_block_map(2, 0);
    b3.setOnes();
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest->resize(matrix->get_rows(), right->get_cols());
  // Reserve destination pattern (must be manually determined)
  dest->reserve_block(0, 0);
  dest->reserve_block(1, 0);
  dest->allocate_memory();
  dest->zero_memory();
  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_multiply_add(mult, dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), matrix->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_cols());

  Eigen::Matrix2d expected;
  expected << 9, 10, 11, 12;

  Eigen::Matrix2d c1 = dest->get_block_map(0, 0);
  Eigen::Matrix2d c2 = dest->get_block_map(1, 0);

  EXPECT_EQ(c1, expected);
  EXPECT_EQ(c2, expected);

  auto a = matrix->to_csc().toDense();
  auto b = right->to_csc().toDense();
  auto c = dest->to_csc().toDense();

  // std::cout << "A:\n" << a << std::endl;
  // std::cout << "B:\n" << b << std::endl;
  // std::cout << "C:\n" << c << std::endl;

  std::cout << "Expected A*B:\n" << a * b << std::endl;
  std::cout << "Actual A*B:\n" << c << std::endl;
}

TEST(SparseBlockMatrix, SubgroupMultiplySub3) {
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize({0, 2, 4}, {0, 2, 4, 7});

    // block row 0
    matrix->reserve_block(0, 0);
    matrix->reserve_block(0, 1);
    matrix->reserve_block(0, 2);

    // block row 1
    matrix->reserve_block(1, 0);
    matrix->reserve_block(1, 1);
    matrix->reserve_block(1, 2);

    matrix->allocate_memory();

    for (int row = 0; row < 2; row++) {
      auto a1 = matrix->get_block_map(row, 0);
      a1 << 1, 2, 3, 4;

      auto a2 = matrix->get_block_map(row, 1);
      a2.setOnes();

      auto a3 = matrix->get_block_map(row, 2);
      a3.setOnes();
      a3 *= 2.0;
    }
  }

  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize({0, 2, 4, 7}, {0, 2});
    right->reserve_block(0, 0);
    right->reserve_block(1, 0);
    right->reserve_block(2, 0);
    right->allocate_memory();

    auto b1 = right->get_block_map(0, 0);
    b1.setIdentity();

    auto b2 = right->get_block_map(1, 0);
    b2.setIdentity();
    b2 *= 2.0;

    auto b3 = right->get_block_map(2, 0);
    b3.setOnes();
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest->resize(matrix->get_rows(), right->get_cols());
  // Reserve destination pattern (must be manually determined)
  dest->reserve_block(0, 0);
  dest->reserve_block(1, 0);
  dest->allocate_memory();
  dest->zero_memory();
  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_multiply_add(mult, dest, right, false);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), matrix->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_cols());

  Eigen::Matrix2d expected;
  expected << 9, 10, 11, 12;

  Eigen::Matrix2d c1 = dest->get_block_map(0, 0);
  Eigen::Matrix2d c2 = dest->get_block_map(1, 0);

  EXPECT_EQ(c1, -expected);
  EXPECT_EQ(c2, -expected);

  auto a = matrix->to_csc().toDense();
  auto b = right->to_csc().toDense();
  auto c = dest->to_csc().toDense();

  // std::cout << "A:\n" << a << std::endl;
  // std::cout << "B:\n" << b << std::endl;
  // std::cout << "C:\n" << c << std::endl;

  std::cout << "Expected -A*B:\n" << -a * b << std::endl;
  std::cout << "Actual -A*B:\n" << c << std::endl;
}

TEST(SparseBlockMatrix, BlockDiagonalMultiplication) {
  const int sz = 1000;
  const int d = 3;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setRandom();
    }
  }
  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      right->reserve_block(i, i);
    }
    right->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = right->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
      m1 *= 2.0;
    }
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    dest->resize(matrix->get_rows(), right->get_cols());
    // Reserve destination pattern (must be manually determined)
    for (int i = 0; i < sz; i++) {
      dest->reserve_block(i, i);
    }
    dest->allocate_memory();
    dest->zero_memory();
  }

  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_block_diagonal_multiply_add(mult, dest, right);
  // auto mult = matrix->multiply_add(dest, right, false);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), matrix->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_cols());

  for (int i = 0; i < sz; i++) {
    auto b1 = dest->get_block_ptr(i, i);
    ASSERT_NE(b1, nullptr);
    ASSERT_EQ(dest->get_block_dim(0, 0), BlockDim(d, d));
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    auto m0 =
        Eigen::Map<Eigen::Matrix<double, d, d>>(matrix->get_block_ptr(i, i));
    ASSERT_EQ(m1, m0 * 2.0);
  }
}

TEST(SparseBlockMatrix, SubgroupTransposedMultiplyAdd) {
  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize({0, 2, 4}, {0, 2, 4, 7});

    // block row 0
    right->reserve_block(0, 0);
    right->reserve_block(0, 1);
    right->reserve_block(0, 2);

    // block row 1
    right->reserve_block(1, 0);
    right->reserve_block(1, 1);
    right->reserve_block(1, 2);

    right->allocate_memory();

    for (int row = 0; row < 2; row++) {
      auto a1 = right->get_block_map(row, 0);
      a1 << 1, 2, 3, 4;

      auto a2 = right->get_block_map(row, 1);
      a2.setOnes();

      auto a3 = right->get_block_map(row, 2);
      a3.setOnes();
      a3 *= 2.0;
    }
  }

  // Create left matrix
  auto left = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    left->resize({0, 2}, {0, 2, 4, 7});
    left->reserve_block(0, 0);
    left->reserve_block(0, 1);
    left->reserve_block(0, 2);
    left->allocate_memory();

    auto b1 = left->get_block_map(0, 0);
    b1.setIdentity();

    auto b2 = left->get_block_map(0, 1);
    b2.setIdentity();
    b2 *= 2.0;

    auto b3 = left->get_block_map(0, 2);
    b3.setOnes();
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest->resize(left->get_rows(), right->get_rows());
  // Reserve destination pattern (must be manually determined)
  dest->reserve_block(0, 0);
  dest->reserve_block(0, 1);
  dest->allocate_memory();
  dest->zero_memory();
  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  // auto mult = left->subgroup_transposed_multiply_add(dest, right);
  left->subgroup_transposed_multiply_add(mult, dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), left->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_rows());

  Eigen::Matrix2d expected;
  expected << 9, 10, 11, 12;
  expected.transposeInPlace();

  Eigen::Matrix2d c1 = dest->get_block_map(0, 0);
  Eigen::Matrix2d c2 = dest->get_block_map(0, 1);

  EXPECT_EQ(c1, expected);
  EXPECT_EQ(c2, expected);

  // std::cout << "c1:\n" << c1 << std::endl;
  // std::cout << "c2:\n" << c2 << std::endl;

  auto a = left->to_csc().toDense();
  auto b = right->to_csc().toDense();
  auto c = dest->to_csc().toDense();

  // std::cout << "A:\n" << a << std::endl;
  // std::cout << "B:\n" << b << std::endl;
  // std::cout << "C:\n" << c << std::endl;

  std::cout << "Expected A*B^T:\n" << a * b.transpose() << std::endl;
  std::cout << "Actual A*B:\n" << c << std::endl;
}

TEST(SparseBlockMatrix, SubgroupTransposedMultiplyAdd2) {
  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize({0, 2, 4}, {0, 2, 4, 7});

    // block row 0
    right->reserve_block(0, 0);
    right->reserve_block(0, 1);
    right->reserve_block(0, 2);

    // block row 1
    right->reserve_block(1, 0);
    right->reserve_block(1, 1);
    right->reserve_block(1, 2);

    right->allocate_memory();

    for (int row = 0; row < 2; row++) {
      auto a1 = right->get_block_map(row, 0);
      a1 << 1, 2, 3, 4;

      auto a2 = right->get_block_map(row, 1);
      a2.setOnes();

      auto a3 = right->get_block_map(row, 2);
      a3.setOnes();
      a3 *= 2.0;
    }
  }

  // Create left matrix
  auto left = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    left->resize({0, 2}, {0, 2, 4, 7});
    left->reserve_block(0, 0);
    left->reserve_block(0, 1);
    left->reserve_block(0, 2);
    left->allocate_memory();

    auto b1 = left->get_block_map(0, 0);
    b1.setIdentity();

    auto b2 = left->get_block_map(0, 1);
    b2.setIdentity();
    b2 *= 2.0;

    auto b3 = left->get_block_map(0, 2);
    b3.setOnes();
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest->resize(left->get_rows(), right->get_rows());
  // Reserve destination pattern (must be manually determined)
  dest->reserve_block(0, 0);
  dest->reserve_block(0, 1);
  dest->allocate_memory();
  dest->zero_memory();
  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto seq = engine.create_op_sequence();
  left->subgroup_transposed_multiply_add2(seq, dest, right);
  // auto mult = left->subgroup_transposed_multiply_add(dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  // mult->execute();
  seq->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), left->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_rows());

  Eigen::Matrix2d expected;
  expected << 9, 10, 11, 12;
  expected.transposeInPlace();

  Eigen::Matrix2d c1 = dest->get_block_map(0, 0);
  Eigen::Matrix2d c2 = dest->get_block_map(0, 1);

  EXPECT_EQ(c1, expected);
  EXPECT_EQ(c2, expected);

  // std::cout << "c1:\n" << c1 << std::endl;
  // std::cout << "c2:\n" << c2 << std::endl;

  auto a = left->to_csc().toDense();
  auto b = right->to_csc().toDense();
  auto c = dest->to_csc().toDense();

  // std::cout << "A:\n" << a << std::endl;
  // std::cout << "B:\n" << b << std::endl;
  // std::cout << "C:\n" << c << std::endl;

  std::cout << "Expected A*B^T:\n" << a * b.transpose() << std::endl;
  std::cout << "Actual A*B:\n" << c << std::endl;
}

TEST(SparseBlockMatrix, SubgroupTransposedMultiplyAdd3) {
  // Create right matrix
  auto right = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    right->resize({0, 2, 4}, {0, 2, 4, 7});

    // block row 0
    right->reserve_block(0, 0);
    right->reserve_block(0, 1);
    right->reserve_block(0, 2);

    // block row 1
    right->reserve_block(1, 0);
    right->reserve_block(1, 1);
    right->reserve_block(1, 2);

    right->allocate_memory();

    for (int row = 0; row < 2; row++) {
      auto a1 = right->get_block_map(row, 0);
      a1 << 1, 2, 3, 4;

      auto a2 = right->get_block_map(row, 1);
      a2.setOnes();

      auto a3 = right->get_block_map(row, 2);
      a3.setOnes();
      a3 *= 2.0;
    }
  }

  // Create left matrix
  auto left = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    left->resize({0, 2}, {0, 2, 4, 7});
    left->reserve_block(0, 0);
    left->reserve_block(0, 1);
    left->reserve_block(0, 2);
    left->allocate_memory();

    auto b1 = left->get_block_map(0, 0);
    b1.setIdentity();

    auto b2 = left->get_block_map(0, 1);
    b2.setIdentity();
    b2 *= 2.0;

    auto b3 = left->get_block_map(0, 2);
    b3.setOnes();
  }

  // Create destination matrix
  auto dest = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest->resize(left->get_rows(), right->get_rows());
  // Reserve destination pattern (must be manually determined)
  dest->reserve_block(0, 0);
  dest->reserve_block(0, 1);
  dest->allocate_memory();
  dest->zero_memory();
  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto seq = engine.create_op_sequence();
  left->subgroup_transposed_multiply_add3(seq, dest, right);
  // auto mult = left->subgroup_transposed_multiply_add(dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  // mult->execute();
  seq->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count()); Check
  // destination has correct structure and values
  ASSERT_EQ(dest->get_rows(), left->get_rows());
  ASSERT_EQ(dest->get_cols(), right->get_rows());

  Eigen::Matrix2d expected;
  expected << 9, 10, 11, 12;
  expected.transposeInPlace();

  Eigen::Matrix2d c1 = dest->get_block_map(0, 0);
  Eigen::Matrix2d c2 = dest->get_block_map(0, 1);

  EXPECT_EQ(c1, expected);
  EXPECT_EQ(c2, expected);

  // std::cout << "c1:\n" << c1 << std::endl;
  // std::cout << "c2:\n" << c2 << std::endl;

  auto a = left->to_csc().toDense();
  auto b = right->to_csc().toDense();
  auto c = dest->to_csc().toDense();

  // std::cout << "A:\n" << a << std::endl;
  // std::cout << "B:\n" << b << std::endl;
  // std::cout << "C:\n" << c << std::endl;

  std::cout << "Expected A*B^T:\n" << a * b.transpose() << std::endl;
  std::cout << "Actual A*B:\n" << c << std::endl;
}

// TODO: Add more tests where matrix is not square
TEST(SparseBlockMatrix, MultiplyVecAdd1) {
  const int sz = 1000;
  const int d = 3;
  const double scale = 5.0;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
      m1 *= scale;
    }
  }
  // Create right vector
  auto right = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto v = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
    v.setRandom();
  }

  // Create destination vector
  auto dest = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto pDest = dest->map();
    std::fill(pDest, pDest + sz * d, 0.0);
  }

  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_multiply_vec_add(mult, dest, right);
  // auto mult = matrix->subgroup_multiply_vec_add(dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count());

  auto v0 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
  auto v1 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(dest->map());
  // std::cout << "v0*s:\n" << v0*scale << std::endl;
  // std::cout << "v1:\n" << v1 << std::endl;
  ASSERT_TRUE(v1.isApprox(v0 * scale));
}

// TODO: Add more tests where matrix is not square
TEST(SparseBlockMatrix, BlockDiagonalMultiplyVecAdd) {
  const int sz = 1000;
  const int d = 3;
  const double scale = 2.0;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }

  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
      m1 *= scale;
    }
  }
  // Create right vector
  auto right = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto v = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
    v.setRandom();
  }

  // Create destination vector
  auto dest = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto pDest = dest->map();
    std::fill(pDest, pDest + sz * d, 0.0);
  }

  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_block_diagonal_multiply_vec_add(mult, dest, right);
  // auto mult = matrix->subgroup_block_diagonal_multiply_vec_add(dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count());

  auto v0 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
  auto v1 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(dest->map());
  // std::cout << "v0*s:\n" << v0*scale << std::endl;
  // std::cout << "v1:\n" << v1 << std::endl;
  ASSERT_TRUE(v1.isApprox(v0 * scale));
}

TEST(SparseBlockMatrix, RightMultiplyVecAdd1) {
  const int sz = 1000;
  const int d = 3;
  const double scale = 5.0;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
      m1 *= scale;
    }
  }
  // Create right vector
  auto right = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto v = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
    v.setRandom();
  }

  // Create destination vector
  auto dest = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto pDest = dest->map();
    std::fill(pDest, pDest + sz * d, 0.0);
  }

  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_right_multiply_vec_add(mult, dest, right);
  // auto mult = matrix->subgroup_right_multiply_vec_add(dest, right);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count());

  auto v0 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
  auto v1 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(dest->map());
  // std::cout << "v0*s:\n" << v0*scale << std::endl;
  // std::cout << "v1:\n" << v1 << std::endl;
  ASSERT_TRUE(v1.isApprox(v0 * scale));
}

TEST(SparseBlockMatrix, MultiplyVecSub) {
  const int sz = 1000;
  const int d = 3;
  const double scale = 5.0;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
      m1 *= scale;
    }
  }
  // Create right vector
  auto right = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto v = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
    v.setRandom();
  }

  // Create destination vector
  auto dest = engine.create_buffer<double>(nullptr, sz * d, BufferType::Host);
  {
    auto pDest = dest->map();
    std::fill(pDest, pDest + sz * d, 0.0);
  }

  // Apply multiplication
  auto trec = std::chrono::high_resolution_clock::now();
  auto mult = engine.create_op_sequence();
  matrix->subgroup_multiply_vec_add(mult, dest, right, false);
  // auto mult = matrix->subgroup_multiply_vec_add(dest, right, false);
  auto ta = std::chrono::high_resolution_clock::now();
  mult->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Rec: {}\nExec: {}\n", std::chrono::duration<double>(ta -
  // trec).count(), std::chrono::duration<double>(tb - ta).count());

  auto v0 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(right->map());
  auto v1 = Eigen::Map<Eigen::Matrix<double, sz * d, 1>>(dest->map());
  // std::cout << "v0*s:\n" << v0*scale << std::endl;
  // std::cout << "v1:\n" << v1 << std::endl;
  ASSERT_TRUE(v1.isApprox(v0 * -scale));
}
/*
TEST(SparseBlockMatrix, BlockDiagonalInverse)
{
    const int sz = 10000;
    const int d = 3;
    const double scale = 5.0;
    std::vector<BlockIndex> dims;
    dims.resize(sz + 1);
    BlockIndex dim = 0;
    for (BlockIndex i = 0; i < dims.size(); i++)
    {
        dims[i] = dim;
        dim += d;
    }
    // Create original matrix
    auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
    {

        matrix->resize(dims, dims);
        for (int i = 0; i < sz; i++)
        {
            matrix->reserve_block(i, i);
        }
        matrix->allocate_memory();

        for (int i = 0; i < sz; i++)
        {
            auto b1 = matrix->get_block_ptr(i, i);
            auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
            m1.setIdentity();
            m1 *= scale;
        }
    }

    SparseBlockMatrix<double> matrix_inv(engine);
    matrix_inv.take_structure_from(matrix);
    matrix_inv.allocate_memory();

    // Invert
    auto ta = std::chrono::high_resolution_clock::now();
    matrix->block_diagonal_inversion(matrix_inv);
    auto tb = std::chrono::high_resolution_clock::now();
    // fmt::print("Exec: {}\n", std::chrono::duration<double>(tb - ta).count());

    // Check
    for (int i = 0; i < sz; i++)
    {
        auto b1 = matrix_inv.get_block_ptr(i, i);
        auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
        ASSERT_EQ(m1, m1.Identity() * 1.0 / scale);
    }
}
*/
TEST(SparseBlockMatrix, MatrixInversionOp) {
  const int sz = 10000;
  const int d = 3;
  const double scale = 5.0;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
      m1 *= scale;
    }
  }

  auto matrix_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  matrix_inv->take_structure_from(matrix);
  matrix_inv->allocate_memory();

  // Invert
  auto op = engine.create_op_sequence();
  matrix->create_inversion_op(op, matrix_inv);
  auto ta = std::chrono::high_resolution_clock::now();
  op->execute();
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Exec: {}\n", std::chrono::duration<double>(tb - ta).count());

  // Check
  for (int i = 0; i < sz; i++) {
    auto b1 = matrix_inv->get_block_ptr(i, i);
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    ASSERT_EQ(m1, m1.Identity() * 1.0 / scale);
  }
}

TEST(SparseBlockMatrix, SetLambda) {
  const int sz = 10000;
  const int d = 3;
  const double lambda = 4.0;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory(BufferType::DeviceCached);

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
    }
  }

  auto set_seq = engine.create_op_sequence();
  auto restore_seq = engine.create_op_sequence();

  auto lambda_buffer =
      engine.create_buffer<double>(nullptr, 1, BufferType::DeviceCached);
  *(lambda_buffer->map()) = lambda;

  set_seq->sync_device<double>({matrix->get_buffer(), lambda_buffer}, {}, true);

  matrix->set_lambda(lambda_buffer, set_seq, restore_seq);

  set_seq->sync_local<double>({matrix->get_buffer(), lambda_buffer}, {}, false);
  restore_seq->sync_local<double>({matrix->get_buffer(), lambda_buffer}, {},
                                  false);

  // Check set
  set_seq->execute();
  for (int i = 0; i < sz; i++) {
    auto b1 = matrix->get_block_ptr(i, i);
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    ASSERT_EQ(m1, m1.Identity() * (lambda + 1.0));
  }

  // Check restore
  restore_seq->execute();
  for (int i = 0; i < sz; i++) {
    auto b2 = matrix->get_block_ptr(i, i);
    auto m2 = Eigen::Map<Eigen::Matrix<double, d, d>>(b2);
    ASSERT_EQ(m2, m2.Identity());
  }
}

// Matrix read test
TEST(SparseBlockMatrix, SetLambda2) {
  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt");
  // hpp->zero_memory();

  auto dest1 = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest1->take_structure_from(hpp);
  dest1->allocate_memory();

  auto lambda_check = [](MatPtr<double> src, MatPtr<double> dest,
                         double lambda_val) {
    for (size_t i = 0; i < src->num_rows(); i++) {
      auto msrc = src->get_block_map(i, i);
      auto mdest = dest->get_block_map(i, i);
      // auto src_dim = src.get_block_dim(i, i);
      // std::cout <<  "Dim: " << src_dim.first << ", " << src_dim.second <<
      // std::endl; std::cout << msrc << " \nvs\n" << mdest << "\n\n";

      if (mdest !=
          (msrc + mdest.Identity(mdest.rows(), mdest.cols()) * lambda_val)) {
        return false;
      }
    }
    return true;
  };

  auto seq1 = engine.create_op_sequence();
  auto seq2 = engine.create_op_sequence();

  auto lambda = engine.create_buffer<double>(nullptr, 1, BufferType::Host);
  lambda->map()[0] = 5.0;

  hpp->copy_blocks_into(seq1, dest1);
  seq1->insert_cc_barrier();
  dest1->set_lambda(lambda, seq1, seq2);
  std::cout << "Setting lambda!\n";
  seq1->execute();
  ASSERT_TRUE(lambda_check(hpp, dest1, 5.0));

  // restore lambda
  std::cout << "Restoring diagonal!\n";
  seq2->execute();
  ASSERT_TRUE(lambda_check(hpp, dest1, 0.0));
}

TEST(SparseBlockMatrix, BlockToCSC) {
  const int sz = 5;
  const int d = 3;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
    }
  }

  matrix->sort_col_indices();

  auto csc = matrix->to_csc();

  for (int i = 0; i < sz; i++) {
    auto b1 = matrix->get_block_ptr(i, i);
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    ASSERT_TRUE(m1.isIdentity());
  }

  Eigen::SparseMatrix<double> target(sz * d, sz * d);
  target.setIdentity();

  ASSERT_EQ(target.rows(), csc.rows());
  ASSERT_EQ(target.cols(), csc.cols());
  ASSERT_EQ(csc.nonZeros(), d * d * sz);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), csc.coeff(i, j));
    }
  }

  // upper
  Eigen::SparseMatrix<double> upper = matrix->to_csc(true);
  Eigen::SparseMatrix<double> target_tri =
      target.triangularView<Eigen::Upper>();

  ASSERT_EQ(upper.nonZeros(), sz * d * (d + 1) / 2);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target_tri.coeff(i, j), upper.coeff(i, j));
    }
  }

  // fill existing
  Eigen::SparseMatrix<double> filled(matrix->num_scalar_rows(),
                                     matrix->num_scalar_cols());

  matrix->fill_csc(filled);

  ASSERT_EQ(target.rows(), filled.rows());
  ASSERT_EQ(target.cols(), filled.cols());
  ASSERT_EQ(filled.nonZeros(), d * d * sz);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), filled.coeff(i, j));
    }
  }

  // fill existing (upper triangular)
  Eigen::SparseMatrix<double> filled_upper(matrix->num_scalar_rows(),
                                           matrix->num_scalar_cols());
  matrix->fill_csc(filled_upper, true);

  ASSERT_EQ(target.rows(), filled_upper.rows());
  ASSERT_EQ(target.cols(), filled_upper.cols());
  ASSERT_EQ(filled_upper.nonZeros(), sz * d * (d + 1) / 2);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), filled_upper.coeff(i, j));
    }
  }

  // fill values only
  Eigen::SparseMatrix<double, Eigen::ColMajor> filled_values =
      matrix->to_csc() * 5.0;

  // std::cout << "filled values (before):\n" << filled_values << std::endl;
  matrix->fill_csc_values(filled_values.valuePtr(), false);
  // std::cout << "filled values (after):\n" << filled_values << std::endl;

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), filled_values.coeff(i, j));
    }
  }

  // fill values (upper triangular)
  Eigen::SparseMatrix<double, Eigen::ColMajor> fv_tri =
      matrix->to_csc(true) * 5.0;

  // std::cout << "fv_tri nnz:\n" << fv_tri.nonZeros() << std::endl;
  // std::cout << "fv_tri (before):\n" << fv_tri << std::endl;
  matrix->fill_csc_values(fv_tri.valuePtr(), true);
  // std::cout << "fv_tri (after):\n" << fv_tri << std::endl;

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), fv_tri.coeff(i, j));
    }
  }
}

TEST(SparseBlockMatrix, BlockToCSC2) {
  const int sz = 5;
  const int d = 3;
  std::vector<BlockIndex> dims;
  dims.resize(sz + 1);
  BlockIndex dim = 0;
  for (BlockIndex i = 0; i < dims.size(); i++) {
    dims[i] = dim;
    dim += d;
  }
  // Create original matrix
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  {
    matrix->resize(dims, dims);
    for (int i = 0; i < sz; i++) {
      matrix->reserve_block(i, i);
    }
    matrix->allocate_memory();

    for (int i = 0; i < sz; i++) {
      auto b1 = matrix->get_block_ptr(i, i);
      auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
      m1.setIdentity();
    }
  }

  matrix->sort_col_indices();

  auto csc = matrix->to_csc2();

  for (int i = 0; i < sz; i++) {
    auto b1 = matrix->get_block_ptr(i, i);
    auto m1 = Eigen::Map<Eigen::Matrix<double, d, d>>(b1);
    ASSERT_TRUE(m1.isIdentity());
  }

  Eigen::SparseMatrix<double> target(sz * d, sz * d);
  target.setIdentity();

  ASSERT_EQ(target.rows(), csc.rows());
  ASSERT_EQ(target.cols(), csc.cols());
  ASSERT_EQ(csc.nonZeros(), d * d * sz);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), csc.coeff(i, j));
    }
  }

  // upper
  Eigen::SparseMatrix<double> upper = matrix->to_csc2(true);
  Eigen::SparseMatrix<double> target_tri =
      target.triangularView<Eigen::Upper>();

  // ASSERT_EQ(upper.nonZeros(), sz * d * (d + 1) / 2);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target_tri.coeff(i, j), upper.coeff(i, j));
    }
  }

  // fill existing
  Eigen::SparseMatrix<double> filled(matrix->num_scalar_rows(),
                                     matrix->num_scalar_cols());

  matrix->fill_csc2(filled);

  ASSERT_EQ(target.rows(), filled.rows());
  ASSERT_EQ(target.cols(), filled.cols());
  // ASSERT_EQ(filled.nonZeros(), d * d * sz);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), filled.coeff(i, j));
    }
  }

  // fill existing (upper triangular)
  Eigen::SparseMatrix<double> filled_upper(matrix->num_scalar_rows(),
                                           matrix->num_scalar_cols());
  matrix->fill_csc2(filled_upper, true);

  ASSERT_EQ(target.rows(), filled_upper.rows());
  ASSERT_EQ(target.cols(), filled_upper.cols());
  // ASSERT_EQ(filled_upper.nonZeros(), sz * d * (d + 1) / 2);

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), filled_upper.coeff(i, j));
    }
  }

  // fill values only
  Eigen::SparseMatrix<double, Eigen::ColMajor> filled_values =
      matrix->to_csc2() * 5.0;

  // std::cout << "filled values (before):\n" << filled_values << std::endl;
  matrix->fill_csc_values2(filled_values.valuePtr(), false);
  // std::cout << "filled values (after):\n" << filled_values << std::endl;

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), filled_values.coeff(i, j));
    }
  }

  // fill values (upper triangular)
  Eigen::SparseMatrix<double, Eigen::ColMajor> fv_tri =
      matrix->to_csc2(true) * 5.0;

  // std::cout << "fv_tri nnz:\n" << fv_tri.nonZeros() << std::endl;
  // std::cout << "fv_tri (before):\n" << fv_tri << std::endl;
  matrix->fill_csc_values2(fv_tri.valuePtr(), true);
  // std::cout << "fv_tri (after):\n" << fv_tri << std::endl;

  for (auto i = 0; i < sz * d; i++) {
    for (auto j = 0; j < sz * d; j++) {
      ASSERT_EQ(target.coeff(i, j), fv_tri.coeff(i, j));
    }
  }
}

template <typename T>
void check_specs(ComputeEngine& engine) {
  auto buf = engine.create_buffer<T>(nullptr, 4096, BufferType::Host);
  ASSERT_EQ(buf->element_size(), sizeof(T));
  ASSERT_EQ(buf->size(), 4096);
  ASSERT_EQ(buf->mem_size(), sizeof(T) * 4096);
}

TEST(GPUBuffer, ElementSize) {
  check_specs<double>(engine);
  check_specs<float>(engine);
  check_specs<int32_t>(engine);
  check_specs<uint>(engine);
  check_specs<bool>(engine);
}

// TEST(SolverMisc, Malloc)
// {
//     auto ta = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < 1000; i++) {
//     auto op = (SBMPushConstants*)malloc(sizeof(SBMPushConstants));
//     op->start = 0;
//     op->n = 1;
//     // free(op);
//     }
//     auto tb = std::chrono::high_resolution_clock::now();
//     std::cout << "malloc: " << std::chrono::duration<double>(tb-ta).count()
//     << std::endl;
// }

TEST(SolverMisc, ColRowMult) {
  // auto a = Eigen::Matrix3d::Random();
  // auto b = Eigen::Matrix3d::Random();
  Eigen::Matrix3d b = Eigen::Matrix3d::Random();
  Eigen::Matrix3d a = Eigen::Matrix3d::Random();
  // auto b = a*2.0;
  // b(1, 1) = 5;

  Eigen::Matrix3d c = a * b;

  Eigen::Matrix3d d = Eigen::Matrix3d::Zero();

  for (int col = 0; col < 3; col++) {
    for (int av = 0; av < 3; av++) {
      for (int bv = 0; bv < 3; bv++) {
        // // fmt::print("{},{} x {},{}\n", av, col, col, bv);
        d(av, bv) += a(av, col) * b(col, bv);
      }
    }
  }

  // for (int row_a = 0; row_a < 3; row_a++) {
  //     for (int col_b = 0; col_b < 3; col_b++) {
  //         for (int v = 0; v < 3; v++) {
  //             d(row_a, col_b) += a(row_a, v)*b(v, col_b);
  //         }
  //     }
  // }
  // std::cout << "c:\n" << c << std::endl;
  // std::cout << "d:\n" << d << std::endl;
  ASSERT_TRUE(c.isApprox(d));
}

// Matrix read test
TEST(SparseBlockMatrix, LoadFromFile) {
  auto matrix = std::make_shared<SparseBlockMatrix<double>>(engine);
  matrix->loadFromFile("data/small.txt");

  auto m0 = matrix->get_block_map(0, 0);
  ASSERT_TRUE(m0.isIdentity());

  Eigen::Matrix3d m1_expected;
  m1_expected << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::Matrix3d m1 = matrix->get_block_map(1, 1);
  ASSERT_EQ(m1, m1_expected);

  auto csc = matrix->to_csc();
  std::cout << csc << std::endl;
}

// Matrix read test
TEST(SparseBlockMatrix, SchurTest) {
  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt");

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt");

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt");
  hpl->sort_by_index();

  auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur_expected->loadFromFile("data/hschur.txt");

  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->take_structure_from(hschur_expected);
  hschur->sort_by_index();
  hschur->allocate_memory();
  hschur->zero_memory();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory();
  hll_inv->zero_memory();

  auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl_inv_hll->take_structure_from(hpl);
  hpl_inv_hll->allocate_memory();
  hpl_inv_hll->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  auto calc_HplinvHll = engine.create_op_sequence();
  hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll,
                                            hll_inv);

  auto calc_Hschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false,
                                                true, true);
  // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur,
  // hpl, false, 0, true, true);

  // do calculations
  inversion_op->execute();
  hpp->copy_blocks_into(hschur);
  calc_HplinvHll->execute();
  calc_Hschur->execute();

  // check result
  auto hschur_csc = hschur->to_csc();
  auto hschur_expected_csc = hschur_expected->to_csc();

  ASSERT_TRUE(hschur_csc.isApprox(hschur_expected_csc));

  // load bschur data
  auto bp = loadFromFile<double>(engine, "data/bp.txt");
  auto bl = loadFromFile<double>(engine, "data/bl.txt");
  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt");
  // auto bschur = engine.create_buffer<double>(nullptr,
  // bschur_expected->size(), compute::BufferType::Host);
  auto bschur = bp;

  // calculate bschur
  auto calc_bschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_multiply_vec_add(calc_bschur, bschur, bl, false);
  // auto calc_bschur = hpl_inv_hll->subgroup_multiply_vec_add(bschur, bl,
  // false);
  calc_bschur->execute();
  // check
  auto bschur_map =
      Eigen::Map<Eigen::MatrixXd>(bschur->map(), bschur->size(), 1);
  auto bschur_expected_map = Eigen::Map<Eigen::MatrixXd>(
      bschur_expected->map(), bschur_expected->size(), 1);

  // std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);
  double tol = 1e-4;
  for (Eigen::Index i = 0; i < bschur_expected_map.rows(); i++) {
    if (fabs(bschur_map(i) - bschur_expected_map(i)) > tol) {
      std::cerr << "Expected " << bschur_expected_map(i) << " at " << i
                << ", got " << bschur_map(i) << std::endl;
      FAIL();
    }
  }

  ASSERT_TRUE(bschur_map.isApprox(bschur_expected_map, tol));

  // load update data
  auto xp = loadFromFile<double>(engine, "data/xp.txt");
  auto xl_expected = loadFromFile<double>(engine, "data/xl.txt");

  auto cl = bl;
  auto xl = engine.create_buffer<double>(nullptr, xl_expected->size(),
                                         compute::BufferType::Host);
  memset(xl->map(), 0, xl->size() * sizeof(double));

  // calculate
  auto calc_cl = engine.create_op_sequence();
  auto calc_xl = engine.create_op_sequence();
  hpl->subgroup_right_multiply_vec_add(calc_cl, cl, xp, false);
  hll_inv->subgroup_block_diagonal_multiply_vec_add(calc_xl, xl, cl, true);

  calc_cl->execute();
  calc_xl->execute();

  // check
  auto xl_map = Eigen::Map<Eigen::MatrixXd>(xl->map(), xl->size(), 1);
  auto xl_expected_map =
      Eigen::Map<Eigen::MatrixXd>(xl_expected->map(), xl_expected->size(), 1);
  ASSERT_TRUE(xl_map.isApprox(xl_expected_map, 1e-6));
}

TEST(SparseBlockMatrix, SchurTestDevice) {
  auto btd = BufferType::DeviceCached;

  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);
  hpl->sort_by_index();

  auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur_expected->loadFromFile("data/hschur.txt", btd);

  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->take_structure_from(hschur_expected);
  hschur->sort_by_index();
  hschur->allocate_memory(btd);
  hschur->zero_memory();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl_inv_hll->take_structure_from(hpl);
  hpl_inv_hll->allocate_memory(btd);
  hpl_inv_hll->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  auto calc_HplinvHll = engine.create_op_sequence();
  hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll,
                                            hll_inv);

  auto ta = std::chrono::high_resolution_clock::now();
  auto calc_Hschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false,
                                                true, true);
  // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur,
  // hpl, false, 0, true, true); calc_Hschur->insert_host_sync_barrier();
  auto tb = std::chrono::high_resolution_clock::now();

  // fmt::print("add1 took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
  // ta).count());

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});

  // do calculations
  inversion_op->execute();
  hpp->copy_blocks_into(hschur);
  calc_HplinvHll->execute();

  sync_Hschur->sync_device();
  calc_Hschur->execute();
  sync_Hschur->sync_local();
  // check result
  auto hschur_csc = hschur->to_csc();
  auto hschur_expected_csc = hschur_expected->to_csc();

  ASSERT_TRUE(hschur_csc.isApprox(hschur_expected_csc));

  // load bschur data
  auto bp = loadFromFile<double>(engine, "data/bp.txt", btd);
  auto bl = loadFromFile<double>(engine, "data/bl.txt", btd);
  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  // auto bschur = engine.create_buffer<double>(nullptr,
  // bschur_expected->size(), compute::BufferType::Host);
  auto bschur = bp;

  auto sync_b = create_sync_object();
  sync_b->rec<double>({bp, bl, bschur_expected});

  // calculate bschur
  sync_b->sync_device();
  auto calc_bschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_multiply_vec_add(calc_bschur, bschur, bl, false);
  // auto calc_bschur = hpl_inv_hll->subgroup_multiply_vec_add(bschur, bl,
  // false);
  calc_bschur->execute();
  sync_b->sync_local();
  // check
  auto bschur_map =
      Eigen::Map<Eigen::MatrixXd>(bschur->map(), bschur->size(), 1);
  auto bschur_expected_map = Eigen::Map<Eigen::MatrixXd>(
      bschur_expected->map(), bschur_expected->size(), 1);

  // std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);
  double tol = 1e-4;
  for (Eigen::Index i = 0; i < bschur_expected_map.rows(); i++) {
    if (fabs(bschur_map(i) - bschur_expected_map(i)) > tol) {
      std::cerr << "Expected " << bschur_expected_map(i) << " at " << i
                << ", got " << bschur_map(i) << std::endl;
      FAIL();
    }
  }

  ASSERT_TRUE(bschur_map.isApprox(bschur_expected_map, tol));

  // load update data
  auto xp = loadFromFile<double>(engine, "data/xp.txt", btd);
  auto xl_expected = loadFromFile<double>(engine, "data/xl.txt", btd);

  auto cl = bl;
  auto xl = engine.create_buffer<double>(nullptr, xl_expected->size(), btd);
  memset(xl->map(), 0, xl->size() * sizeof(double));

  // calculate

  auto calc_cl = engine.create_op_sequence();
  auto calc_xl = engine.create_op_sequence();
  hpl->subgroup_right_multiply_vec_add(calc_cl, cl, xp, false);
  hll_inv->subgroup_block_diagonal_multiply_vec_add(calc_xl, xl, cl, true);

  // auto calc_cl = hpl->subgroup_right_multiply_vec_add(cl, xp, false);
  // auto calc_xl = hll_inv->subgroup_block_diagonal_multiply_vec_add(xl, cl,
  // true);

  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp, xl_expected, xl});
  sync_x->sync_device();
  calc_cl->execute();
  calc_xl->execute();
  sync_x->sync_local();

  // check
  auto xl_map = Eigen::Map<Eigen::MatrixXd>(xl->map(), xl->size(), 1);
  auto xl_expected_map =
      Eigen::Map<Eigen::MatrixXd>(xl_expected->map(), xl_expected->size(), 1);
  ASSERT_TRUE(xl_map.isApprox(xl_expected_map, 1e-6));
}

TEST(SparseBlockMatrix, SchurDiagonal) {
  auto btd = BufferType::DeviceCached;

  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);
  hpl->sort_by_index();

  auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur_expected->loadFromFile("data/hschur.txt", btd);

  auto expected_diagonal = std::make_shared<SparseBlockMatrix<double>>(engine);
  expected_diagonal->take_diagonal_structure_from(hschur_expected);
  expected_diagonal->allocate_memory();
  expected_diagonal->zero_memory();

  for (size_t i = 0; i < hschur_expected->num_rows(); i++) {
    auto m = hschur_expected->get_block_map(i, i);
    auto d = expected_diagonal->get_block_map(i, i);
    d = m;
  }

  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->take_diagonal_structure_from(hpp);
  hschur->sort_by_index();
  hschur->allocate_memory(btd);
  hschur->zero_memory();

  for (size_t i = 0; i < hpp->num_rows(); i++) {
    auto m = hpp->get_block_map(i, i);
    auto d = hschur->get_block_map(i, i);
    d = m;
  }

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  // auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  // hpl_inv_hll->take_structure_from(hpl);
  // hpl_inv_hll->allocate_memory(btd);
  // hpl_inv_hll->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  // auto calc_HplinvHll =
  // hpl->subgroup_block_diagonal_multiply_add(hpl_inv_hll, hll_inv);

  // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur,
  // hpl, false, 0, true, true); calc_Hschur->insert_host_sync_barrier();
  auto calc_Hschur = engine.create_op_sequence();
  hpl->multiply_block_diagonal_self_transpose(calc_Hschur, hschur, hll_inv,
                                              false);

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});

  // do calculations
  inversion_op->execute();
  // hpp->copy_blocks_into(hschur);
  // calc_HplinvHll->execute();

  sync_Hschur->sync_device();
  auto ta = std::chrono::high_resolution_clock::now();
  calc_Hschur->execute();
  auto tb = std::chrono::high_resolution_clock::now();

  // fmt::print("schur diagonal took: {} ms\n", 1000 *
  // std::chrono::duration<double>(tb - ta).count());

  sync_Hschur->sync_local();
  // check result
  // auto hschur_csc = hschur->to_csc();
  // auto hschur_expected_csc = hschur_expected->to_csc();
  auto expected = expected_diagonal->to_csc();
  auto actual = hschur->to_csc();

  for (size_t i = 0; i < hschur->num_rows(); i++) {
    auto m1 = hschur->get_block_map(i, i);
    auto m2 = expected_diagonal->get_block_map(i, i);

    if (!m1.isApprox(m2, 1e-4)) {
      std::cout << "expected =\n" << m2 << "\n";
      std::cout << "actual =\n" << m1 << "\n";
      break;
    }
  }

  ASSERT_TRUE(actual.isApprox(expected));
}

TEST(SparseBlockMatrix, SchurTestDevice2) {
  auto btd = BufferType::DeviceCached;

  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);

  auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur_expected->loadFromFile("data/hschur.txt", btd);

  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->take_structure_from(hschur_expected);
  hschur->sort_by_index();
  hschur->allocate_memory(btd);
  hschur->zero_memory();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl_inv_hll->take_structure_from(hpl);
  hpl_inv_hll->allocate_memory(btd);
  hpl_inv_hll->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  auto calc_HplinvHll = engine.create_op_sequence();
  hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll,
                                            hll_inv);

  // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur,
  // hpl, false, 0, true, true); calc_Hschur->insert_host_sync_barrier();
  auto seq = engine.create_op_sequence();
  auto ta = std::chrono::high_resolution_clock::now();
  hpl_inv_hll->subgroup_transposed_multiply_add2(seq, hschur, hpl, false, true);
  auto tb = std::chrono::high_resolution_clock::now();

  // fmt::print("add2 took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
  // ta).count());

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});

  // do calculations
  inversion_op->execute();
  hpp->copy_blocks_into(hschur);
  calc_HplinvHll->execute();

  sync_Hschur->sync_device();
  // calc_Hschur->execute();
  seq->execute();
  sync_Hschur->sync_local();
  // check result
  auto hschur_csc = hschur->to_csc();
  auto hschur_expected_csc = hschur_expected->to_csc();

  ASSERT_TRUE(hschur_csc.isApprox(hschur_expected_csc));

  // load bschur data
  auto bp = loadFromFile<double>(engine, "data/bp.txt", btd);
  auto bl = loadFromFile<double>(engine, "data/bl.txt", btd);
  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  // auto bschur = engine.create_buffer<double>(nullptr,
  // bschur_expected->size(), compute::BufferType::Host);
  auto bschur = bp;

  auto sync_b = create_sync_object();
  sync_b->rec<double>({bp, bl, bschur_expected});

  // calculate bschur
  sync_b->sync_device();
  auto calc_bschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_multiply_vec_add(calc_bschur, bschur, bl, false);
  // auto calc_bschur = hpl_inv_hll->subgroup_multiply_vec_add(bschur, bl,
  // false);
  calc_bschur->execute();
  sync_b->sync_local();
  // check
  auto bschur_map =
      Eigen::Map<Eigen::MatrixXd>(bschur->map(), bschur->size(), 1);
  auto bschur_expected_map = Eigen::Map<Eigen::MatrixXd>(
      bschur_expected->map(), bschur_expected->size(), 1);

  // std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);
  double tol = 1e-4;
  for (Eigen::Index i = 0; i < bschur_expected_map.rows(); i++) {
    if (fabs(bschur_map(i) - bschur_expected_map(i)) > tol) {
      std::cerr << "Expected " << bschur_expected_map(i) << " at " << i
                << ", got " << bschur_map(i) << std::endl;
      FAIL();
    }
  }

  ASSERT_TRUE(bschur_map.isApprox(bschur_expected_map, tol));

  // load update data
  auto xp = loadFromFile<double>(engine, "data/xp.txt", btd);
  auto xl_expected = loadFromFile<double>(engine, "data/xl.txt", btd);

  auto cl = bl;
  auto xl = engine.create_buffer<double>(nullptr, xl_expected->size(), btd);
  memset(xl->map(), 0, xl->size() * sizeof(double));

  // calculate

  auto calc_cl = engine.create_op_sequence();
  auto calc_xl = engine.create_op_sequence();
  hpl->subgroup_right_multiply_vec_add(calc_cl, cl, xp, false);
  hll_inv->subgroup_block_diagonal_multiply_vec_add(calc_xl, xl, cl, true);

  // auto calc_cl = hpl->subgroup_right_multiply_vec_add(cl, xp, false);
  // auto calc_xl = hll_inv->subgroup_block_diagonal_multiply_vec_add(xl, cl,
  // true);

  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp, xl_expected, xl});
  sync_x->sync_device();
  calc_cl->execute();
  calc_xl->execute();
  sync_x->sync_local();

  // check
  auto xl_map = Eigen::Map<Eigen::MatrixXd>(xl->map(), xl->size(), 1);
  auto xl_expected_map =
      Eigen::Map<Eigen::MatrixXd>(xl_expected->map(), xl_expected->size(), 1);
  ASSERT_TRUE(xl_map.isApprox(xl_expected_map, 1e-6));
}

TEST(SparseBlockMatrix, SchurTestDevice3) {
  auto btd = BufferType::DeviceCached;

  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);

  auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur_expected->loadFromFile("data/hschur.txt", btd);

  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->take_structure_from(hschur_expected);
  hschur->sort_by_index();
  hschur->allocate_memory(btd);
  hschur->zero_memory();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl_inv_hll->take_structure_from(hpl);
  hpl_inv_hll->allocate_memory(btd);
  hpl_inv_hll->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  auto calc_HplinvHll = engine.create_op_sequence();
  hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll,
                                            hll_inv);

  // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur,
  // hpl, false, 0, true, true); calc_Hschur->insert_host_sync_barrier();
  auto seq = engine.create_op_sequence();
  auto ta = std::chrono::high_resolution_clock::now();
  hpl_inv_hll->subgroup_transposed_multiply_add3(seq, hschur, hpl, false, true);
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("add3 took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
  // ta).count());

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});

  // do calculations
  inversion_op->execute();
  hpp->copy_blocks_into(hschur);
  calc_HplinvHll->execute();

  sync_Hschur->sync_device();
  // calc_Hschur->execute();
  seq->execute();
  sync_Hschur->sync_local();
  // check result
  auto hschur_csc = hschur->to_csc();
  auto hschur_expected_csc = hschur_expected->to_csc();

  ASSERT_TRUE(hschur_csc.isApprox(hschur_expected_csc));

  // load bschur data
  auto bp = loadFromFile<double>(engine, "data/bp.txt", btd);
  auto bl = loadFromFile<double>(engine, "data/bl.txt", btd);
  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  // auto bschur = engine.create_buffer<double>(nullptr,
  // bschur_expected->size(), compute::BufferType::Host);
  auto bschur = bp;

  auto sync_b = create_sync_object();
  sync_b->rec<double>({bp, bl, bschur_expected});

  // calculate bschur
  sync_b->sync_device();
  auto calc_bschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_multiply_vec_add(calc_bschur, bschur, bl, false);
  calc_bschur->execute();
  sync_b->sync_local();
  // check
  auto bschur_map =
      Eigen::Map<Eigen::MatrixXd>(bschur->map(), bschur->size(), 1);
  auto bschur_expected_map = Eigen::Map<Eigen::MatrixXd>(
      bschur_expected->map(), bschur_expected->size(), 1);

  // std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);
  double tol = 1e-4;
  for (Eigen::Index i = 0; i < bschur_expected_map.rows(); i++) {
    if (fabs(bschur_map(i) - bschur_expected_map(i)) > tol) {
      std::cerr << "Expected " << bschur_expected_map(i) << " at " << i
                << ", got " << bschur_map(i) << std::endl;
      FAIL();
    }
  }

  ASSERT_TRUE(bschur_map.isApprox(bschur_expected_map, tol));

  // load update data
  auto xp = loadFromFile<double>(engine, "data/xp.txt", btd);
  auto xl_expected = loadFromFile<double>(engine, "data/xl.txt", btd);

  auto cl = bl;
  auto xl = engine.create_buffer<double>(nullptr, xl_expected->size(), btd);
  memset(xl->map(), 0, xl->size() * sizeof(double));

  // calculate

  auto calc_cl = engine.create_op_sequence();
  auto calc_xl = engine.create_op_sequence();
  hpl->subgroup_right_multiply_vec_add(calc_cl, cl, xp, false);
  hll_inv->subgroup_block_diagonal_multiply_vec_add(calc_xl, xl, cl, true);

  // auto calc_cl = hpl->subgroup_right_multiply_vec_add(cl, xp, false);
  // auto calc_xl = hll_inv->subgroup_block_diagonal_multiply_vec_add(xl, cl,
  // true);

  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp, xl_expected, xl});
  sync_x->sync_device();
  calc_cl->execute();
  calc_xl->execute();
  sync_x->sync_local();

  // check
  auto xl_map = Eigen::Map<Eigen::MatrixXd>(xl->map(), xl->size(), 1);
  auto xl_expected_map =
      Eigen::Map<Eigen::MatrixXd>(xl_expected->map(), xl_expected->size(), 1);
  ASSERT_TRUE(xl_map.isApprox(xl_expected_map, 1e-6));
}

TEST(Solver, SelfInnerProduct) {
  int vlen = 1e6;

  Eigen::VectorXd vec = Eigen::VectorXd::Random(vlen);
  auto x = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
  auto out = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
  auto out2 = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);

  auto s = engine.create_op_sequence();
  s->self_inner_product(x, out, out2);

  auto ta = std::chrono::high_resolution_clock::now();
  s->execute();
  auto tb = std::chrono::high_resolution_clock::now();

  std::cout << "Expected: " << vec.dot(vec) << "\n";
  std::cout << "Got: " << out2->map()[0] << "\n";
  // fmt::print("Took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
  // ta).count());

  ASSERT_TRUE(fabs(vec.dot(vec) - out2->map()[0]) < 1e-4);
}

TEST(Solver, InnerProduct) {
  int vlen = 1e6;

  Eigen::VectorXd vec = Eigen::VectorXd::Random(vlen);
  Eigen::VectorXd vec2 = Eigen::VectorXd::Random(vlen);
  auto x = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
  auto x2 = engine.create_buffer<double>(vec2.data(), vlen, BufferType::Host);

  auto out = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
  auto out2 = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);

  auto s = engine.create_op_sequence();
  s->inner_product(x, x2, out, out2);

  auto ta = std::chrono::high_resolution_clock::now();
  s->execute();
  auto tb = std::chrono::high_resolution_clock::now();

  std::cout << "Expected: " << vec.dot(vec2) << "\n";
  std::cout << "Got: " << out2->map()[0] << "\n";
  // fmt::print("Took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
  // ta).count());

  ASSERT_TRUE(fabs(vec.dot(vec2) - out2->map()[0]) < 1e-4);
}

TEST(Solver, InnerProductDevice) {
  int vlen = 4e6;

  Eigen::VectorXd vec = Eigen::VectorXd::Random(vlen);
  Eigen::VectorXd vec2 = Eigen::VectorXd::Random(vlen);
  auto x =
      engine.create_buffer<double>(vec.data(), vlen, BufferType::DeviceCached);
  auto x2 =
      engine.create_buffer<double>(vec2.data(), vlen, BufferType::DeviceCached);

  auto out = engine.create_buffer<double>(nullptr, vlen, BufferType::Device);
  auto out2 =
      engine.create_buffer<double>(nullptr, vlen, BufferType::DeviceCached);

  auto sync_in = create_sync_object();
  sync_in->rec<double>({x, x2});
  sync_in->sync_device();

  auto sync_out = create_sync_object();
  sync_out->rec<double>({out2}, {{0, 1}});

  auto s = engine.create_op_sequence();
  s->inner_product(x, x2, out, out2);
  auto ta = std::chrono::high_resolution_clock::now();
  s->execute();
  sync_out->sync_local();
  auto tb = std::chrono::high_resolution_clock::now();

  std::cout << "Expected: " << vec.dot(vec2) << "\n";
  std::cout << "Got: " << out2->map()[0] << "\n";
  // fmt::print("Took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
  // ta).count());

  ASSERT_TRUE(fabs(vec.dot(vec2) - out2->map()[0]) < 1e-4);
}

TEST(Solver, AddVec) {
  int vlen = 1e6;

  Eigen::VectorXd data_u = Eigen::VectorXd::Random(vlen);
  Eigen::VectorXd data_v = Eigen::VectorXd::Random(vlen);
  Eigen::VectorXd data_w = Eigen::VectorXd::Zero(vlen);

  auto u = engine.create_buffer<double>(data_u.data(), vlen,
                                        BufferType::DeviceCached);
  auto v = engine.create_buffer<double>(data_v.data(), vlen,
                                        BufferType::DeviceCached);
  auto w = engine.create_buffer<double>(data_w.data(), vlen,
                                        BufferType::DeviceCached);
  auto w2 = engine.create_buffer<double>(data_w.data(), vlen,
                                         BufferType::DeviceCached);

  double scale = 4.5;
  auto c = engine.create_buffer<double>(&scale, 1, BufferType::DeviceCached);

  auto sync_in = create_sync_object();
  sync_in->rec<double>({u, v, c});
  sync_in->sync_device();

  {
    auto sync_out = create_sync_object();
    sync_out->rec<double>({w});

    auto s = engine.create_op_sequence();
    double a = -0.25;
    s->add_vec(u, v, w, a);
    auto ta = std::chrono::high_resolution_clock::now();
    s->execute();
    sync_out->sync_local();
    auto tb = std::chrono::high_resolution_clock::now();

    // fmt::print("Took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
    // ta).count());
    auto wMap = Eigen::Map<Eigen::MatrixXd>(w->map(), w->size(), 1);

    ASSERT_EQ(data_u + a * data_v, wMap);
  }

  // repeat for w2
  {
    auto s2 = engine.create_op_sequence();
    s2->add_vec(u, v, w2, c);
    auto sync_out2 = create_sync_object();
    sync_out2->rec<double>({w2});

    auto ta = std::chrono::high_resolution_clock::now();
    s2->execute();
    sync_out2->sync_local();
    auto tb = std::chrono::high_resolution_clock::now();

    // fmt::print("Took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
    // ta).count());
    auto w2Map = Eigen::Map<Eigen::MatrixXd>(w2->map(), w2->size(), 1);

    ASSERT_TRUE((data_u + scale * data_v).isApprox(w2Map));

    // ASSERT_EQ(data_u + scale*data_v, w2Map);
  }
}

TEST(Solver, CopyVec) {
  int vlen = 1e6;

  Eigen::VectorXd data_u = Eigen::VectorXd::Random(vlen);
  Eigen::VectorXd data_v = Eigen::VectorXd::Random(vlen);

  auto u = engine.create_buffer<double>(data_u.data(), vlen,
                                        BufferType::DeviceCached);
  auto v = engine.create_buffer<double>(data_v.data(), vlen,
                                        BufferType::DeviceCached);

  auto sync_in = create_sync_object();
  sync_in->rec<double>({u, v});
  sync_in->sync_device();

  auto s = engine.create_op_sequence();
  s->copy_vec(u, v);
  auto ta = std::chrono::high_resolution_clock::now();
  s->execute();
  sync_in->sync_local();
  auto tb = std::chrono::high_resolution_clock::now();

  // fmt::print("Took: {} ms\n", 1000 * std::chrono::duration<double>(tb -
  // ta).count());
  auto vMap = Eigen::Map<Eigen::MatrixXd>(v->map(), v->size(), 1);

  ASSERT_EQ(data_u, vMap);
}

// Matrix read test
TEST(SparseBlockMatrix, CopyBlocks) {
  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt");

  auto dest1 = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest1->take_structure_from(hpp);
  dest1->allocate_memory();

  auto dest2 = std::make_shared<SparseBlockMatrix<double>>(engine);
  dest2->take_structure_from(hpp);
  dest2->allocate_memory();

  auto copy_check = [](MatPtr<double> src, MatPtr<double> dest) {
    const auto rows = src->get_row_indices();
    for (size_t i = 0; i < rows.size(); i++) {
      for (auto j : rows[i]) {
        auto msrc = src->get_block_map(i, j);
        auto mdest = dest->get_block_map(i, j);

        if (msrc != mdest) {
          return false;
        }
      }
    }
    return true;
  };

  hpp->copy_blocks_into(dest1);

  ASSERT_TRUE(copy_check(hpp, dest1));

  auto seq = engine.create_op_sequence();

  hpp->copy_blocks_into(seq, dest2);
  seq->execute();
  ASSERT_TRUE(copy_check(hpp, dest2));
}

TEST(SparseBlockMatrix, UpperTriangularMultiplication) {
  auto btd = BufferType::DeviceCached;

  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);
  hpl->sort_by_index();

  auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur_expected->loadFromFile("data/hschur.txt", btd);

  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->take_structure_from(hschur_expected);
  hschur->sort_by_index();
  hschur->allocate_memory(btd);
  hschur->zero_memory();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl_inv_hll->take_structure_from(hpl);
  hpl_inv_hll->allocate_memory(btd);
  hpl_inv_hll->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  auto calc_HplinvHll = engine.create_op_sequence();
  hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll,
                                            hll_inv);

  auto calc_Hschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false,
                                                true, true);

  // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur,
  // hpl, false, 0, true, true); calc_Hschur->insert_host_sync_barrier();

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});

  // do calculations
  inversion_op->execute();
  hpp->copy_blocks_into(hschur);
  calc_HplinvHll->execute();

  sync_Hschur->sync_device();
  calc_Hschur->execute();
  sync_Hschur->sync_local();
  // check result
  auto hschur_csc = hschur->to_csc();
  auto hschur_expected_csc = hschur_expected->to_csc();

  ASSERT_TRUE(hschur_csc.isApprox(hschur_expected_csc));

  Eigen::VectorXd v1 = Eigen::VectorXd::Ones(hschur_expected_csc.rows());
  Eigen::VectorXd zero_vec = Eigen::VectorXd::Zero(hschur_expected_csc.rows());
  auto bufV1 =
      engine.create_buffer(v1.data(), v1.rows(), BufferType::DeviceCached);
  auto resultVec = engine.create_buffer<double>(
      zero_vec.data(), zero_vec.rows(), BufferType::DeviceCached);

  auto sync_vecs = create_sync_object();
  sync_vecs->rec<double>({bufV1, resultVec});
  sync_vecs->sync_device();
  auto seq = engine.create_op_sequence();
  hschur->multiply_vec_add_upper(seq, resultVec, bufV1);
  hschur->multiply_vec_lower_no_diagonal(seq, resultVec, bufV1);

  seq->execute();

  sync_vecs->sync_local();

  auto resultVecMap =
      Eigen::Map<Eigen::MatrixXd>(resultVec->map(), resultVec->size(), 1);

  auto expected_symmetric = hschur_expected_csc.selfadjointView<Eigen::Upper>();

  ASSERT_TRUE((expected_symmetric * v1).isApprox(resultVecMap, 1e-6));
}

TEST(LinearSolver, PCGCPU) {
  auto btd = BufferType::DeviceCached;

  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);
  hpl->sort_by_index();

  auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur_expected->loadFromFile("data/hschur.txt", btd);

  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->take_structure_from(hschur_expected);
  hschur->sort_by_index();
  hschur->allocate_memory(btd);
  hschur->zero_memory();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl_inv_hll->take_structure_from(hpl);
  hpl_inv_hll->allocate_memory(btd);
  hpl_inv_hll->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  auto calc_HplinvHll = engine.create_op_sequence();
  hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll,
                                            hll_inv);

  auto calc_Hschur = engine.create_op_sequence();
  hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false,
                                                true, true);
  // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur,
  // hpl, false, 0, true, true); calc_Hschur->insert_host_sync_barrier();

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});

  // do calculations
  inversion_op->execute();
  hpp->copy_blocks_into(hschur);
  calc_HplinvHll->execute();

  sync_Hschur->sync_device();
  calc_Hschur->execute();
  sync_Hschur->sync_local();
  // check result
  auto hschur_csc = hschur->to_csc();
  auto hschur_expected_csc = hschur_expected->to_csc();

  ASSERT_TRUE(hschur_csc.isApprox(hschur_expected_csc));

  // allocate
  Eigen::VectorXd zero_vec = Eigen::VectorXd::Zero(hschur->num_scalar_rows());
  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  Eigen::VectorXd b = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
                                                  bschur_expected->size(), 1);

  // stopping criteria
  double tol = 1e-7;

  // end of pre-loop initialization

  // Create multiplications/sequences

  auto A = hschur_expected_csc.selfadjointView<Eigen::Upper>();
  auto m = std::make_shared<SparseBlockMatrix<double>>(engine);
  m->take_diagonal_structure_from(hschur_expected);
  m->allocate_memory(btd);
  std::cout << "M num blocks: " << m->num_blocks() << std::endl;
  hschur_expected->copy_inverse_diagonal_blocks_into(m);
  auto Minv = m->to_csc();
  Eigen::VectorXd xk = zero_vec;
  Eigen::VectorXd rk = b - A * xk;
  Eigen::VectorXd zk = Minv * rk;
  Eigen::VectorXd pk = zk;
  Eigen::VectorXd xk1 = xk;

  int max_iter = 500;
  double res;
  int k;
  auto ta = std::chrono::high_resolution_clock::now();
  for (k = 0; k < max_iter; k++) {
    // compute vk = A*p_k
    Eigen::VectorXd vk = A * pk;
    double akn_coeff = rk.dot(zk);
    double akd_coeff = pk.dot(vk);
    double ak = akn_coeff / akd_coeff;
    if (akn_coeff < tol) break;

    xk1 = xk + ak * pk;
    Eigen::VectorXd rk1 = rk - ak * vk;

    res = rk.norm();
    // std::cout << "Residual: " << res << std::endl;
    // if (res < tol) break;

    // zk1 = M^-1*rk1
    Eigen::VectorXd zk1 = Minv * rk1;

    // bkn = dot(rk1, zk1) and bkd = dot(rk, zk)
    // we already have akn = dot(rk, zk)

    double bk = rk1.dot(zk1) / akn_coeff;

    // pk1 = zk1 + bk*pk
    Eigen::VectorXd pk1 = zk1 + bk * pk;

    // update variables
    xk = xk1;
    rk = rk1;
    zk = zk1;
    pk = pk1;
  }

  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Took: {}\n", std::chrono::duration<double>(tb - ta).count());

  std::cout << "k = " << k << std::endl;
  std::cout << "Residual at end: " << res << std::endl;
  Eigen::VectorXd d1 = A * xk1;
  Eigen::VectorXd d2 = b;
  std::cout << "Relative Residual: " << res / d2.norm() << std::endl;

  std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
  ASSERT_TRUE((d1).isApprox(d2, 1e-3));
}

TEST(LinearSolver, PCGExplicitSchur) {
  auto btd = BufferType::DeviceCached;

  // Load Data
  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->loadFromFile("data/hschur.txt", btd);
  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});
  sync_Hschur->sync_device();
  auto hschur_csc = hschur->to_csc();

  auto pcg = new PCGSolver<double>(engine);
  // pcg->set_max_iterations(100000);
  // pcg->set_tolerance(1e-20);

  LinearSolver<double>* solver = pcg;

  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  auto xp =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // sync bschur
  auto sync_b = create_sync_object();
  sync_b->rec<double>({bschur_expected});
  sync_b->sync_device();

  // Solver Setup
  solver->setup(hschur, xp, bschur_expected);

  // Run Solver
  auto ta = std::chrono::high_resolution_clock::now();
  solver->solve(hschur, xp, bschur_expected);
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Took: {}\n", std::chrono::duration<double>(tb - ta).count());
  delete solver;
  // Read output
  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp});
  sync_x->sync_local();

  auto xk1_map = Eigen::Map<Eigen::MatrixXd>(xp->map(), xp->size(), 1);

  auto A = hschur_csc.selfadjointView<Eigen::Upper>();

  auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
                                                bschur_expected->size(), 1);

  // std::cout << "k = " << k << std::endl;
  // std::cout << "Residual at end: " << res << std::endl;
  auto d1 = A * xk1_map;
  auto d2 = bschur_map;
  // std::cout << "Relative Residual: " << res / d2.norm()  << std::endl;
  std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
  ASSERT_TRUE((d1).isApprox(d2, 1e-1));
}

TEST(LinearSolver, ImplicitSchurCalc) {
  auto btd = BufferType::DeviceCached;

  // Load Data
  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);
  hpl->sort_by_index();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  // do calculations
  inversion_op->execute();

  // Load expected Hschur
  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->loadFromFile("data/hschur.txt", btd);
  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});
  sync_Hschur->sync_device();
  auto hschur_csc = hschur->to_csc();

  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  auto xp =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // sync bschur
  auto sync_b = create_sync_object();
  sync_b->rec<double>({bschur_expected});
  sync_b->sync_device();

  auto test_seq = engine.create_op_sequence();

  auto y_implicit =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);
  auto y_explicit =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  auto l0 =
      engine.create_buffer<double>(nullptr, hll_inv->num_scalar_rows(), btd);
  auto l1 =
      engine.create_buffer<double>(nullptr, hll_inv->num_scalar_rows(), btd);
  auto test_input =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // ASSERT_TRUE(hpp->debug_check_symmetry());

  test_seq->fill_vec(y_implicit, 0.0);
  test_seq->fill_vec(y_explicit, 0.0);
  test_seq->fill_vec(test_input, 1.0);

  test_seq->insert_cc_barrier();

  PCGSolver<double>::compute_Ax(test_seq, nullptr, hpp, hpl, hll_inv,
                                test_input, y_implicit, l0, l1, true);
  test_seq->insert_cc_barrier();
  PCGSolver<double>::compute_Ax(test_seq, hschur, nullptr, nullptr, nullptr,
                                test_input, y_explicit, l0, l1, true);
  test_seq->insert_cc_barrier();

  // Solver Setup
  test_seq->execute();

  // Read output
  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp, y_implicit, y_explicit});
  sync_x->sync_local();

  auto map_y_imp =
      Eigen::Map<Eigen::MatrixXd>(y_implicit->map(), y_implicit->size(), 1);
  auto map_y_exp =
      Eigen::Map<Eigen::MatrixXd>(y_explicit->map(), y_explicit->size(), 1);

  std::cout << "Difference: " << (map_y_imp - map_y_exp).norm() << std::endl;
  ASSERT_TRUE((map_y_imp).isApprox(map_y_exp, 1e-1));
}

TEST(LinearSolver, PCGImplicitSchur) {
  auto btd = BufferType::DeviceCached;

  // Load Data
  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);
  hpl->sort_by_index();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  // do calculations
  inversion_op->execute();

  // Load expected Hschur
  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->loadFromFile("data/hschur.txt", btd);
  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});
  sync_Hschur->sync_device();
  auto hschur_csc = hschur->to_csc();

  PCGSolver<double>* solver = new PCGSolver<double>(engine);

  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  auto xp =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // sync bschur
  auto sync_b = create_sync_object();
  sync_b->rec<double>({bschur_expected});
  sync_b->sync_device();

  // Solver Setup
  solver->set_preconditioner(Preconditioner::Hpp);
  solver->setup(hpp, hpl, hll_inv, xp, bschur_expected);

  // Run Solver
  auto ta = std::chrono::high_resolution_clock::now();
  solver->solve(hpp, hpl, hll_inv, xp, bschur_expected);
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Took: {}\n", std::chrono::duration<double>(tb - ta).count());
  delete solver;
  // Read output
  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp});
  sync_x->sync_local();

  auto xk1_map = Eigen::Map<Eigen::MatrixXd>(xp->map(), xp->size(), 1);

  auto A = hschur_csc.selfadjointView<Eigen::Upper>();

  auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
                                                bschur_expected->size(), 1);

  // std::cout << "k = " << k << std::endl;
  // std::cout << "Residual at end: " << res << std::endl;
  auto d1 = A * xk1_map;
  auto d2 = bschur_map;
  // std::cout << "Relative Residual: " << res / d2.norm()  << std::endl;
  std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
  ASSERT_TRUE((d1).isApprox(d2, 1e-1));
}

TEST(LinearSolver, PCGImplicitExactSchurPreconditioner) {
  auto btd = BufferType::DeviceCached;

  // Load Data
  auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpp->loadFromFile("data/hpp.txt", btd);

  auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll->loadFromFile("data/hll.txt", btd);

  auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
  hpl->loadFromFile("data/hpl.txt", btd);
  hpl->sort_by_index();

  auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
  hll_inv->take_structure_from(hll);
  hll_inv->allocate_memory(btd);
  hll_inv->zero_memory();

  // prepare calculations
  auto inversion_op = engine.create_op_sequence();
  hll->create_inversion_op(inversion_op, hll_inv);

  // Sync
  auto sync_Hll_Hpl = create_sync_object();
  sync_Hll_Hpl->rec<double>(
      {hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
  sync_Hll_Hpl->sync_device();

  // do calculations
  inversion_op->execute();

  // Load expected Hschur
  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->loadFromFile("data/hschur.txt", btd);
  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});
  sync_Hschur->sync_device();
  auto hschur_csc = hschur->to_csc();

  PCGSolver<double>* solver = new PCGSolver<double>(engine);

  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  auto xp =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // sync bschur
  auto sync_b = create_sync_object();
  sync_b->rec<double>({bschur_expected});
  sync_b->sync_device();

  // Solver Setup
  solver->set_preconditioner(Preconditioner::Schur);
  solver->setup(hpp, hpl, hll_inv, xp, bschur_expected);

  // Run Solver
  auto ta = std::chrono::high_resolution_clock::now();
  // solver->solve(hschur, hpp, hpl, hll_inv, xp, bschur_expected);
  solver->solve(hpp, hpl, hll_inv, xp, bschur_expected);
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Took: {}\n", std::chrono::duration<double>(tb - ta).count());
  delete solver;
  // Read output
  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp});
  sync_x->sync_local();

  auto xk1_map = Eigen::Map<Eigen::MatrixXd>(xp->map(), xp->size(), 1);

  auto A = hschur_csc.selfadjointView<Eigen::Upper>();

  auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
                                                bschur_expected->size(), 1);

  // std::cout << "k = " << k << std::endl;
  // std::cout << "Residual at end: " << res << std::endl;
  auto d1 = A * xk1_map;
  auto d2 = bschur_map;
  // std::cout << "Relative Residual: " << res / d2.norm()  << std::endl;
  std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
  ASSERT_TRUE((d1).isApprox(d2, 1e-1));
}

TEST(LinearSolver, LDLT) {
  auto btd = BufferType::DeviceCached;

  // Load Data
  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->loadFromFile("data/hschur.txt", btd);
  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});
  sync_Hschur->sync_device();
  auto hschur_csc = hschur->to_csc();

  LinearSolver<double>* solver = new LDLTSolver<double>();

  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  auto xp =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // sync bschur
  auto sync_b = create_sync_object();
  sync_b->rec<double>({bschur_expected});
  sync_b->sync_device();

  // Solver Setup
  solver->setup(hschur, xp, bschur_expected);

  // Run Solver
  auto ta = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(solver->solve(hschur, xp, bschur_expected));
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Took: {}\n", std::chrono::duration<double>(tb - ta).count());

  // Read output
  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp});
  // sync_x->sync_local();

  auto xk1_map = Eigen::Map<Eigen::MatrixXd>(xp->map(), xp->size(), 1);

  auto A = hschur_csc.selfadjointView<Eigen::Upper>();

  auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
                                                bschur_expected->size(), 1);

  Eigen::VectorXd d1 = A * xk1_map;
  auto d2 = bschur_map;
  std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
  ASSERT_TRUE((d1).isApprox(d2, 1e-3));

  // repeat once more time to test other path
  ta = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(solver->solve(hschur, xp, bschur_expected));
  tb = std::chrono::high_resolution_clock::now();
  // // fmt::print("2nd Iteration took: {}\n",std::chrono::duration<double>(tb -
  // ta).count());
  d1 = A * xk1_map;
  ASSERT_TRUE((d1).isApprox(d2, 1e-3));

  delete solver;
}

TEST(LinearSolver, DenseLDLT) {
  auto btd = BufferType::DeviceCached;

  // Load Data
  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->loadFromFile("data/hschur.txt", btd);
  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});
  sync_Hschur->sync_device();
  auto hschur_csc = hschur->to_csc();

  LinearSolver<double>* solver = new DenseLDLTSolver<double>();

  auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
  auto xp =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // sync bschur
  auto sync_b = create_sync_object();
  sync_b->rec<double>({bschur_expected});
  sync_b->sync_device();

  // Solver Setup
  solver->setup(hschur, xp, bschur_expected);

  // Run Solver
  auto ta = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(solver->solve(hschur, xp, bschur_expected));
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("Took: {}\n", std::chrono::duration<double>(tb - ta).count());

  // Read output
  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp});
  // sync_x->sync_local();

  auto xk1_map = Eigen::Map<Eigen::MatrixXd>(xp->map(), xp->size(), 1);

  auto A = hschur_csc.selfadjointView<Eigen::Upper>();

  auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
                                                bschur_expected->size(), 1);

  Eigen::VectorXd d1 = A * xk1_map;
  auto d2 = bschur_map;
  std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
  ASSERT_TRUE((d1).isApprox(d2, 1e-3));

  // repeat once more time to test other path
  ta = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(solver->solve(hschur, xp, bschur_expected));
  tb = std::chrono::high_resolution_clock::now();
  // // fmt::print("2nd Iteration took: {}\n",std::chrono::duration<double>(tb -
  // ta).count());
  d1 = A * xk1_map;
  ASSERT_TRUE((d1).isApprox(d2, 1e-3));

  delete solver;
}

#ifdef CUDA_ENABLED

#include <solver/linear_solver_cuda.hpp>

TEST(PGO, LLTCUDA) {
  auto btd = BufferType::DeviceCached;

  // Load Data
  auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
  hschur->loadFromFile("data-pgo/hpp.txt", btd);
  auto sync_Hschur = create_sync_object();
  sync_Hschur->rec<double>({hschur->get_buffer()});
  sync_Hschur->sync_device();
  auto hschur_csc = hschur->to_csc();

  LinearSolver<double>* solver = new LLTSolverCUDA<double>();

  auto bschur_expected =
      loadFromFile<double>(engine, "data-pgo/bschur.txt", btd);
  auto xp =
      engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

  // sync bschur
  auto sync_b = create_sync_object();
  sync_b->rec<double>({bschur_expected});
  sync_b->sync_device();

  // Solver Setup
  solver->setup(hschur, xp, bschur_expected);

  // Run Solver
  auto ta = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(solver->solve(hschur, xp, bschur_expected));
  auto tb = std::chrono::high_resolution_clock::now();
  // fmt::print("PGO-LDLT Took: {}\n", std::chrono::duration<double>(tb -
  // ta).count());

  // Read output
  auto sync_x = create_sync_object();
  sync_x->rec<double>({xp});
  // sync_x->sync_local();

  auto xk1_map = Eigen::Map<Eigen::MatrixXd>(xp->map(), xp->size(), 1);

  auto A = hschur_csc.selfadjointView<Eigen::Upper>();

  auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
                                                bschur_expected->size(), 1);

  Eigen::VectorXd d1 = A * xk1_map;
  auto d2 = bschur_map;
  std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
  ASSERT_TRUE((d1).isApprox(d2, 1e-3));

  // repeat once more time to test other path
  ta = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(solver->solve(hschur, xp, bschur_expected));
  tb = std::chrono::high_resolution_clock::now();
  // // fmt::print("2nd Iteration took: {}\n",std::chrono::duration<double>(tb -
  // ta).count());
  d1 = A * xk1_map;
  ASSERT_TRUE((d1).isApprox(d2, 1e-3));

  delete solver;
}

#endif

/*
TEST(PGO, PCGExplicitSchur)
{

    auto btd = BufferType::DeviceCached;

    // Load Data
    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->loadFromFile("data-pgo/hpp.txt", btd);
    auto sync_Hschur = create_sync_object();
    sync_Hschur->rec<double>({hschur->get_buffer()});
    sync_Hschur->sync_device();
    auto hschur_csc = hschur->to_csc();

    auto pcg = new PCGSolver<double>(engine);
    // pcg->set_max_iterations(50);
    // pcg->set_tolerance(1e-1);
    // pcg->set_tolerance(1e-20);

    LinearSolver<double> *solver = pcg;

    auto bschur_expected = loadFromFile<double>(engine, "data-pgo/bschur.txt",
btd); auto xp = engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(),
btd);

    // sync bschur
    auto sync_b = create_sync_object();
    sync_b->rec<double>({bschur_expected});
    sync_b->sync_device();

    // Solver Setup
    auto ts0 = std::chrono::high_resolution_clock::now();
    solver->setup(hschur, xp, bschur_expected);
    auto ts1 = std::chrono::high_resolution_clock::now();
    fmt::print("PGO-PCG Setup Took: {}\n", std::chrono::duration<double>(ts1 -
ts0).count());

    // Run Solver
    auto ta = std::chrono::high_resolution_clock::now();
    // solver->solve(hschur, xp, bschur_expected);
    auto tb = std::chrono::high_resolution_clock::now();
    fmt::print("PGO-PCG Took: {}\n", std::chrono::duration<double>(tb -
ta).count()); delete solver;
    // Read output
    auto sync_x = create_sync_object();
    sync_x->rec<double>({xp});
    sync_x->sync_local();

    auto xk1_map = Eigen::Map<Eigen::MatrixXd>(xp->map(), xp->size(), 1);

    auto A = hschur_csc.selfadjointView<Eigen::Upper>();

    auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(),
bschur_expected->size(), 1);

    // std::cout << "k = " << k << std::endl;
    // std::cout << "Residual at end: " << res << std::endl;
    auto d1 = A * xk1_map;
    auto d2 = bschur_map;
    // std::cout << "Relative Residual: " << res / d2.norm()  << std::endl;
    std::cout << "Difference: " << (d1 - d2).norm() << std::endl;
    ASSERT_TRUE((d1).isApprox(d2, 1e-1));
}
*/