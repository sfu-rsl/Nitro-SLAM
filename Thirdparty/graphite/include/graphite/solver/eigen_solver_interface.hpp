/// @file eigen_solver_interface.hpp
#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace graphite {

// We need to make sure the eigen solver function calls are not
// compiled in a .cu file. So we declare specializations here and
// implement them in a .cpp file.
// TODO: Is there a better way to do this?

template <typename T>
using VecMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename Solver> Solver *create_eigen_ldlt_solver() {
  return nullptr;
}

template <typename Solver> void destroy_eigen_ldlt_solver(Solver *solver) {}

template <typename T> class EigenLDLTSolverImpl;

template <>
EigenLDLTSolverImpl<double> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<double>>();

template <>
EigenLDLTSolverImpl<float> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<float>>();

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<double>>(
    EigenLDLTSolverImpl<double> *solver);

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<float>>(
    EigenLDLTSolverImpl<float> *solver);

template <typename T> class EigenLDLTWrapper {
private:
  // using solver_impl = std::conditional_t<
  //     std::is_same<T, double>::value, EigenLDLTSolverImpl<double>,

  using solver_impl = EigenLDLTSolverImpl<T>;

  solver_impl *solver;

public:
  EigenLDLTWrapper() { solver = create_eigen_ldlt_solver<solver_impl>(); }

  ~EigenLDLTWrapper() {
    destroy_eigen_ldlt_solver<solver_impl>(solver);
    solver = nullptr;
  }

  bool analyze_pattern(const Eigen::SparseMatrix<T, Eigen::ColMajor> &matrix) {
    return false;
  }
  bool factorize(const Eigen::SparseMatrix<T, Eigen::ColMajor> &matrix) {
    return false;
  }
  bool solve(const VecMap<T> &b, VecMap<T> &x) { return false; }
};

template <>
bool EigenLDLTWrapper<double>::analyze_pattern(
    const Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix);

template <>
bool EigenLDLTWrapper<double>::factorize(
    const Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix);

template <>
bool EigenLDLTWrapper<double>::solve(const VecMap<double> &b,
                                     VecMap<double> &x);

template <>
bool EigenLDLTWrapper<float>::analyze_pattern(
    const Eigen::SparseMatrix<float, Eigen::ColMajor> &matrix);
template <>
bool EigenLDLTWrapper<float>::factorize(
    const Eigen::SparseMatrix<float, Eigen::ColMajor> &matrix);
template <>
bool EigenLDLTWrapper<float>::solve(const VecMap<float> &b, VecMap<float> &x);

} // namespace graphite