/// @file cudss.hpp
#pragma once
#include <cuda_runtime.h>
#include <cudss.h>
#include <graphite/hessian.hpp>
#include <graphite/solver/solver.hpp>

namespace graphite {

template <typename T> cudaDataType_t get_cuda_data_type();

template <> inline cudaDataType_t get_cuda_data_type<double>() {
  return CUDA_R_64F;
}

template <> inline cudaDataType_t get_cuda_data_type<float>() {
  return CUDA_R_32F;
}

template <typename T, typename S> class cudssSolver : public Solver<T, S> {
private:
  using StorageIndex = int32_t;

  void fill_matrix_structure() {
    const auto dim = d_matrix.d_pointers.size() - 1;
    const auto nnz = d_matrix.d_values.size();
    const cudssMatrixType_t matrix_type = CUDSS_MTYPE_SYMMETRIC;
    const cudssMatrixViewType_t view_type = CUDSS_MVIEW_LOWER;
    const cudssIndexBase_t index_base = CUDSS_BASE_ZERO;

    if (m_A != NULL) {
      cudssMatrixDestroy(m_A);
      m_A = NULL;
    }

    cudssMatrixCreateCsr(&m_A, dim, dim, nnz, d_matrix.d_pointers.data().get(),
                         nullptr, d_matrix.d_indices.data().get(),
                         d_matrix.d_values.data().get(), CUDA_R_32I,
                         get_cuda_data_type<S>(), matrix_type, view_type,
                         index_base);
  }

  void fill_matrix_values() {
    cudssMatrixSetValues(m_A, d_matrix.d_values.data().get());
  }

  Hessian<T, S> H;
  CSCMatrix<S, StorageIndex> d_matrix;

  bool factorization_failed;

  cudaStream_t stream;
  cudssHandle_t handle;

  cudssConfig_t solver_config;
  cudssData_t solver_data;

  cudssMatrix_t m_x, m_b, m_A;

  thrust::device_vector<T> solver_x;

public:
  cudssSolver(bool use_hybrid_execution) {
    stream = NULL;
    m_x = NULL;
    m_b = NULL;
    m_A = NULL;
    factorization_failed = false;

    cudaStreamCreate(&stream);
    cudssCreate(&handle);
    cudssSetStream(handle, stream);

    cudssConfigCreate(&solver_config);
    int enable_hybrid_exec_mode = use_hybrid_execution ? 1 : 0;
    cudssConfigSet(solver_config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE,
                   &enable_hybrid_exec_mode, sizeof(enable_hybrid_exec_mode));
    cudssAlgType_t reordering_alg = CUDSS_ALG_DEFAULT;
    cudssConfigSet(solver_config, CUDSS_CONFIG_REORDERING_ALG, &reordering_alg,
                   sizeof(reordering_alg));
    cudssDataCreate(handle, &solver_data);
  }

  ~cudssSolver() {
    if (m_A != NULL) {
      cudssMatrixDestroy(m_A);
      m_A = NULL;
    }
    if (m_b != NULL) {
      cudssMatrixDestroy(m_b);
      m_b = NULL;
    }
    if (m_x != NULL) {
      cudssMatrixDestroy(m_x);
      m_x = NULL;
    }

    cudssDataDestroy(handle, solver_data);
    cudssConfigDestroy(solver_config);
    cudssDestroy(handle);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

  virtual void update_structure(Graph<T, S> *graph,
                                StreamPool &streams) override {
    H.build_structure(graph, streams);
    H.build_csc_structure(graph, d_matrix);
    fill_matrix_structure();

    // Create matrices for b and x
    if (m_b != NULL) {
      cudssMatrixDestroy(m_b);
      m_b = NULL;
    }
    const auto dim = graph->get_hessian_dimension();
    auto &b = graph->get_b();
    int ldb = dim;
    int ldx = dim;
    cudssMatrixCreateDn(&m_b, dim, 1, ldb, b.data().get(),
                        get_cuda_data_type<T>(), CUDSS_LAYOUT_COL_MAJOR);

    if (m_x != NULL) {
      cudssMatrixDestroy(m_x);
      m_x = NULL;
    }
    solver_x.resize(b.size());
    cudssMatrixCreateDn(&m_x, dim, 1, ldx, solver_x.data().get(),
                        get_cuda_data_type<T>(), CUDSS_LAYOUT_COL_MAJOR);

    // Factorize
    factorization_failed = false;
    auto status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config,
                               solver_data, m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      factorization_failed = true;
      std::cerr << "cudss Analysis failed with error code: " << status
                << std::endl;
    }
    cudaStreamSynchronize(stream);
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

    if (factorization_failed) {
      return false;
    }

    auto dim = graph->get_hessian_dimension();

    thrust::fill(thrust::device, x, x + dim, static_cast<T>(0.0));
    thrust::copy(thrust::device, x, x + dim, solver_x.data());

    cudssStatus_t status;

    // set values for b and x
    cudssMatrixSetValues(m_b, graph->get_b().data().get());
    cudssMatrixSetValues(m_x, solver_x.data().get());

    status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config,
                          solver_data, m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      std::cerr << "cudss Factorization failed with error code: " << status
                << std::endl;
      return false;
    }

    status = cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data,
                          m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      std::cerr << "cudss Solve failed with error code: " << status
                << std::endl;
      return false;
    }
    cudaStreamSynchronize(stream);

    thrust::copy(thrust::device, solver_x.data(), solver_x.data() + dim, x);

    return true;
  }
};

} // namespace graphite