/// @file eigen_solver.cpp
#include <graphite/solver/eigen_solver_interface.hpp>

namespace graphite {

template <class Method>
class Decomp : public Method {
 public:
};


template <typename T>
class EigenLDLTSolverImpl {
  public:

  using decomp_method =
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor>, Eigen::Upper>;
  Decomp<decomp_method> decomp;

  bool analyze_pattern(const Eigen::SparseMatrix<T, Eigen::ColMajor> &matrix) {
    decomp.analyzePattern(matrix);
    return decomp.info() == Eigen::Success;
  }

  bool factorize(const Eigen::SparseMatrix<T, Eigen::ColMajor> &matrix) {
    decomp.factorize(matrix);
    return decomp.info() == Eigen::Success;
  }

  bool solve(const VecMap<T> &b,
             VecMap<T> &x) {
        x = decomp.solve(b);
        return decomp.info() == Eigen::Success;            
    }
};

// Implementation for solver creation

template <>
EigenLDLTSolverImpl<double>* create_eigen_ldlt_solver<EigenLDLTSolverImpl<double>>() {
  return new EigenLDLTSolverImpl<double>();
}

template <>
EigenLDLTSolverImpl<float>* create_eigen_ldlt_solver<EigenLDLTSolverImpl<float>>() {
  return new EigenLDLTSolverImpl<float>();
}

// Implementation for solver destruction
template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<double>>(EigenLDLTSolverImpl<double>* solver) {
    delete solver;
}

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<float>>(EigenLDLTSolverImpl<float>* solver) {
    delete solver;
}

// Implementations of wrapper function specializations

template <>
bool EigenLDLTWrapper<double>::analyze_pattern(const Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix) {
  return solver->analyze_pattern(matrix);
}

template <>
bool EigenLDLTWrapper<double>::factorize(const Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix) {
  return solver->factorize(matrix);
}

template <>
bool EigenLDLTWrapper<double>::solve(const VecMap<double> &b,
                                     VecMap<double> &x) {
  return solver->solve(b, x);
}

template <>
bool EigenLDLTWrapper<float>::analyze_pattern(const Eigen::SparseMatrix<float, Eigen::ColMajor> &matrix) {
  return solver->analyze_pattern(matrix);
}

template <>
bool EigenLDLTWrapper<float>::factorize(const Eigen::SparseMatrix<float, Eigen::ColMajor> &matrix) {
  return solver->factorize(matrix);
}

template <>
bool EigenLDLTWrapper<float>::solve(const VecMap<float> &b,
                                     VecMap<float> &x) {
  return solver->solve(b, x);
}

} // namespace graphite