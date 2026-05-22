#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <solver/sparse_block_matrix.hpp>
// Interface for Linear Solvers
namespace compute {

template <typename DataType>
class LinearSolver {
 public:
  virtual ~LinearSolver(){};

  virtual void setup(MatPtr<DataType> A, BufferPtr<DataType> x,
                     BufferPtr<DataType> b) {
    (void)(A);
    (void)(x);
    (void)(b);
  };
  virtual bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
                     BufferPtr<DataType> b) = 0;
  virtual bool result_gpu() = 0;
  virtual size_t device_mem() { return 0; }
};

template <typename DataType>
class ImplicitSchurSolver {
 public:
  virtual ~ImplicitSchurSolver(){};

  virtual void setup(MatPtr<DataType> Hpp, MatPtr<DataType> Hpl,
                     MatPtr<DataType> Hllinv, BufferPtr<DataType> x,
                     BufferPtr<DataType> b) = 0;

  virtual bool solve(MatPtr<DataType> Hpp, MatPtr<DataType> Hpl,
                     MatPtr<DataType> Hllinv, BufferPtr<DataType> x,
                     BufferPtr<DataType> b) = 0;
};

template <class Method>
class Decomp : public Method {
 public:
  void analyzePattern_block(MatPtr<typename Method::Scalar> A,
                            const typename Method::MatrixType& matrix,
                            bool LDLT) {
    // generate permutation based on block ordering
    auto perm = A->get_ordered_permutation(true);

    // analyze pattern using this permutation
    this->m_Pinv = perm;
    this->m_P = perm.inverse();

    Eigen::SparseMatrix<double, Eigen::ColMajor> pattern(A->num_scalar_rows(),
                                                         A->num_scalar_cols());
    pattern.selfadjointView<Eigen::Upper>() =
        matrix.template selfadjointView<Method::UpLo>().twistedBy(this->m_P);
    this->analyzePattern_preordered(pattern, LDLT);
  }
};

template <typename DataType>
class LDLTSolver : public LinearSolver<DataType> {
 private:
  using decomp_method =
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<DataType, Eigen::ColMajor>,
                            Eigen::Upper>;
  Decomp<decomp_method> decomp;

  Eigen::SparseMatrix<DataType, Eigen::ColMajor> matrix;
  bool first_iter;
  bool block_ordering;

 public:
  LDLTSolver(bool use_block_ordering = false)
      : first_iter(true), block_ordering(use_block_ordering) {}

  bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
             BufferPtr<DataType> b) override {
    if (first_iter) {
      A->fill_csc2(matrix, true);
      if (block_ordering) {
        decomp.analyzePattern_block(A, matrix, true);
      } else {
        decomp.analyzePattern(matrix);
      }
      first_iter = false;
    } else {
      A->fill_csc_values2(matrix.valuePtr(), true);
    }
    decomp.factorize(matrix);
    if (decomp.info() != Eigen::Success) {
      std::cerr << "LDLT matrix decomposition failed!";
      return false;
    }

    auto map_b = Eigen::Map<Eigen::MatrixXd>(b->map(), b->size(), 1);
    auto map_x = Eigen::Map<Eigen::MatrixXd>(x->map(), x->size(), 1);
    map_x = decomp.solve(map_b);
    return true;
  }

  bool result_gpu() override { return false; }
};

template <typename DataType>
class LLTSolver : public LinearSolver<DataType> {
 private:
  using decomp_method =
      Eigen::SimplicialLLT<Eigen::SparseMatrix<DataType, Eigen::ColMajor>,
                           Eigen::Upper>;
  Decomp<decomp_method> decomp;

  Eigen::SparseMatrix<DataType, Eigen::ColMajor> matrix;
  bool first_iter;
  bool block_ordering;

 public:
  LLTSolver(bool use_block_ordering = false)
      : first_iter(true), block_ordering(use_block_ordering) {}

  bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
             BufferPtr<DataType> b) override {
    if (first_iter) {
      A->fill_csc2(matrix, true);
      if (block_ordering) {
        decomp.analyzePattern_block(A, matrix, false);
      } else {
        decomp.analyzePattern(matrix);
      }
      first_iter = false;
    } else {
      A->fill_csc_values2(matrix.valuePtr(), true);
    }
    decomp.factorize(matrix);
    if (decomp.info() != Eigen::Success) {
      std::cerr << "LLT matrix decomposition failed!";
      return false;
    }

    auto map_b = Eigen::Map<Eigen::MatrixXd>(b->map(), b->size(), 1);
    auto map_x = Eigen::Map<Eigen::MatrixXd>(x->map(), x->size(), 1);
    map_x = decomp.solve(map_b);
    return true;
  }

  bool result_gpu() override { return false; }
};

template <typename DataType>
class DenseLDLTSolver : public LinearSolver<DataType> {
 private:
  using decomp_method = Eigen::LDLT<Eigen::MatrixXd, Eigen::Upper>;
  decomp_method decomp;

  Eigen::MatrixXd matrix;
  bool first_iter;

 public:
  DenseLDLTSolver() : first_iter(true) {}

  bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
             BufferPtr<DataType> b) override {
    if (first_iter) {
      A->init_dense(matrix, true);
      first_iter = false;
    } else {
      A->fill_dense(matrix, true);
    }
    decomp.compute(matrix);
    if (decomp.info() != Eigen::Success) {
      std::cerr << "LDLT matrix decomposition failed!";
      return false;
    }

    auto map_b = Eigen::Map<Eigen::MatrixXd>(b->map(), b->size(), 1);
    auto map_x = Eigen::Map<Eigen::MatrixXd>(x->map(), x->size(), 1);
    map_x = decomp.solve(map_b);
    return true;
  }

  bool result_gpu() override { return false; }
};

template <typename DataType>
class DenseLLTSolver : public LinearSolver<DataType> {
 private:
  using decomp_method = Eigen::LLT<Eigen::MatrixXd, Eigen::Upper>;
  decomp_method decomp;

  Eigen::MatrixXd matrix;
  bool first_iter;

 public:
  DenseLLTSolver() : first_iter(true) {}

  bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
             BufferPtr<DataType> b) override {
    if (first_iter) {
      A->init_dense(matrix, true);
      first_iter = false;
    } else {
      A->fill_dense(matrix, true);
    }
    decomp.compute(matrix);
    if (decomp.info() != Eigen::Success) {
      std::cerr << "LDLT matrix decomposition failed!";
      return false;
    }

    auto map_b = Eigen::Map<Eigen::MatrixXd>(b->map(), b->size(), 1);
    auto map_x = Eigen::Map<Eigen::MatrixXd>(x->map(), x->size(), 1);
    map_x = decomp.solve(map_b);
    return true;
  }

  bool result_gpu() override { return false; }
};

}  // namespace compute