#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include <solver/linear_solver.hpp>
#include <type_traits>

namespace compute {

template <typename DataType>
class LLTSolverCUDA : public LinearSolver<DataType> {
 private:
  bool first_iter;
  bool block_ordering;
  cusolverSpHandle_t handle;
  cusparseHandle_t sphandle;
  Eigen::SparseMatrix<DataType, Eigen::ColMajor> matrix;
  cusparseMatDescr_t desc;
  csrcholInfo_t chol_info;

  // matrix buffers
  DataType* data;
  int* outerIndices;
  int* innerIndices;
  DataType* b_dev;
  DataType* x_dev;
  int* p_dev;
  int* pt_dev;
  int* vp_dev;
  DataType* data_perm;
  int reorder;
  size_t internal_bytes;
  size_t workspace_bytes;
  size_t device_bytes;
  void* workspace_buffer;

  cusparseDnVecDescr_t data_vec;
  cusparseSpVecDescr_t data_perm_vec;

  cusparseDnVecDescr_t x_dev_vec;
  cusparseSpVecDescr_t b_dev_vec;
  cusparseSpVecDescr_t b_dev_vec_pt;


  std::vector<int> pt;
  Eigen::SparseMatrix<DataType, Eigen::RowMajor> full_matrix;
  using decomp_method =
      Eigen::SimplicialLLT<Eigen::SparseMatrix<DataType, Eigen::ColMajor>,
                           Eigen::Upper>;
  Decomp<decomp_method> decomp;

  void permute_matrix(
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t>& P) {
    int n = matrix.cols();

    size_t buffer_size = 0;
    cusolverSpXcsrperm_bufferSizeHost(
        handle, n, n, full_matrix.nonZeros(), desc, full_matrix.outerIndexPtr(),
        full_matrix.innerIndexPtr(), P.indices().data(), P.indices().data(),
        &buffer_size);

    std::vector<unsigned char> buffer(buffer_size);

    std::vector<int> value_perm(full_matrix.nonZeros());
    std::iota(value_perm.begin(), value_perm.end(), 0);

    auto result = cusolverSpXcsrpermHost(
        handle, n, n, full_matrix.nonZeros(), desc, full_matrix.outerIndexPtr(),
        full_matrix.innerIndexPtr(), P.indices().data(), P.indices().data(),
        value_perm.data(), buffer.data());

    if (result != CUSOLVER_STATUS_SUCCESS) {
      std::cout << "Error: Permutation failed" << std::endl;
    }

    const size_t sz_perm_values = sizeof(int) * full_matrix.nonZeros();
    cudaMalloc((void**)&vp_dev, sz_perm_values);
    device_bytes += sz_perm_values;

    cudaMemcpy(vp_dev, value_perm.data(), sizeof(int) * full_matrix.nonZeros(),
               cudaMemcpyKind::cudaMemcpyDefault);
    
    // Create cusparse vectors for permutation
    cusparseCreateDnVec(&data_vec, full_matrix.nonZeros(), data, CUDA_R_64F);
    cusparseCreateSpVec(&data_perm_vec, full_matrix.nonZeros(), full_matrix.nonZeros(),
              vp_dev, data_perm, 
              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  }

  void permute_matrix_values() {
    cudaMemcpy(data, full_matrix.valuePtr(),
               sizeof(DataType) * full_matrix.nonZeros(),
               cudaMemcpyKind::cudaMemcpyDefault);
    // cusparseDgthr(sphandle, full_matrix.nonZeros(), data, data_perm, vp_dev,
    //               CUSPARSE_INDEX_BASE_ZERO);

    cusparseGather(sphandle, data_vec, data_perm_vec);
  }

 public:
  LLTSolverCUDA(bool use_block_ordering = false)
      : first_iter(true),
        block_ordering(use_block_ordering),
        data(nullptr),
        outerIndices(nullptr),
        innerIndices(nullptr),
        b_dev(nullptr),
        x_dev(nullptr),
        p_dev(nullptr),
        pt_dev(nullptr),
        vp_dev(nullptr),
        data_perm(nullptr),
        device_bytes(0),
        workspace_buffer(nullptr) {
    static_assert(std::is_same<double, DataType>(),
                  "Only doubles supported by solver!");
    cusolverSpCreate(&handle);
    cusparseCreate(&sphandle);

    cusolverSpCreateCsrcholInfo(&chol_info);

    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    // cusparseSetMatFillMode(
    //     desc, CUSPARSE_FILL_MODE_LOWER);  // we generate UT CCS matrices,
    //     which
    //                                       // should be CSR lower when
    //                                       transposed
    cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
  }

  ~LLTSolverCUDA() {
    // destroy vectors
    if (data) {
      cusparseDestroyDnVec(data_vec);
      cusparseDestroySpVec(data_perm_vec);
    }
    cudaFree(data);
    cudaFree(data_perm);
    cudaFree(outerIndices);
    cudaFree(innerIndices);

    if (b_dev) {
      cusparseDestroySpVec(b_dev_vec_pt);
      cusparseDestroySpVec(b_dev_vec);
      cusparseDestroyDnVec(x_dev_vec);
    }

    cudaFree(b_dev);
    cudaFree(x_dev);

    cudaFree(p_dev);
    cudaFree(pt_dev);
    cudaFree(vp_dev);

    cudaFree(workspace_buffer);

    cusparseDestroyMatDescr(desc);
    cusolverSpDestroyCsrcholInfo(chol_info);
    cusparseDestroy(sphandle);
    cusolverSpDestroy(handle);
  }

  bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
             BufferPtr<DataType> b) override {
    if (first_iter) {
      A->fill_csc2(matrix, true);
      // create full symmetric matrix with both upper and lower triangular parts
      full_matrix = matrix.template selfadjointView<Eigen::Upper>();
      full_matrix.makeCompressed();
      // Get permutation matrix and copy it, and its inverse, into the device
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t> P;
      if (block_ordering) {
        P = A->get_ordered_permutation(true);
      } else {
        Eigen::AMDOrdering<int32_t> amd_ordering;
        amd_ordering(matrix, P);
      }
      // copy into device
      int n = full_matrix.cols();

      const size_t sz_perm = sizeof(int) * n;

      cudaMalloc((void**)&p_dev, sz_perm);
      device_bytes += sz_perm;
      cudaMemcpy(p_dev, P.indices().data(), sz_perm,
                 cudaMemcpyKind::cudaMemcpyDefault);

      pt.resize(n);

      for (int i = 0; i < n; i++) {
        pt[P.indices()[i]] = i;
      }

      cudaMalloc((void**)&pt_dev, sz_perm);
      device_bytes += sz_perm;
      cudaMemcpy(pt_dev, pt.data(), sz_perm, cudaMemcpyKind::cudaMemcpyDefault);

      // carry on
      int nnz = full_matrix.nonZeros();

      const size_t sz_nnz_values = sizeof(DataType) * nnz;
      cudaMalloc((void**)&data, sz_nnz_values);
      device_bytes += sz_nnz_values;

      cudaMalloc((void**)&data_perm, sz_nnz_values);
      device_bytes += sz_nnz_values;

      const size_t sz_outer_indices =
          sizeof(int) * (full_matrix.outerSize() + 1);
      cudaMalloc((void**)&outerIndices, sz_outer_indices);
      device_bytes += sz_outer_indices;

      const size_t sz_inner_indices = sizeof(int) * nnz;
      cudaMalloc((void**)&innerIndices, sz_inner_indices);
      device_bytes += sz_inner_indices;

      cudaMalloc((void**)&b_dev, b->mem_size());
      cusparseCreateSpVec(&b_dev_vec, b->size(), b->size(),
              p_dev, b_dev, 
              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
      cusparseCreateSpVec(&b_dev_vec_pt, b->size(), b->size(),
              pt_dev, b_dev, 
              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

      device_bytes += b->mem_size();

      cudaMalloc((void**)&x_dev, x->mem_size());
      cusparseCreateDnVec(&x_dev_vec, x->size(), x_dev, CUDA_R_64F);
      device_bytes += x->mem_size();

      // now permute the matrix
      permute_matrix(P);  // permute matrix on host

      cudaMemcpy(outerIndices, full_matrix.outerIndexPtr(),
                 sizeof(int) * (full_matrix.outerSize() + 1),
                 cudaMemcpyKind::cudaMemcpyDefault);
      cudaMemcpy(innerIndices, full_matrix.innerIndexPtr(), sizeof(int) * nnz,
                 cudaMemcpyKind::cudaMemcpyDefault);
      permute_matrix_values();  // upload data and permute

      // Analyze
      cusolverSpXcsrcholAnalysis(handle, full_matrix.rows(),
                                 full_matrix.nonZeros(), desc, outerIndices,
                                 innerIndices, chol_info);

      // Create workspace buffer
      workspace_bytes = 0;
      cusolverSpDcsrcholBufferInfo(handle, full_matrix.rows(),
                                   full_matrix.nonZeros(), desc, data_perm,
                                   outerIndices, innerIndices, chol_info,
                                   &internal_bytes, &workspace_bytes);

      cudaMalloc(&workspace_buffer, workspace_bytes);
      device_bytes += workspace_bytes;

      first_iter = false;
    } else {
      A->fill_csc_values2(matrix.valuePtr(), true);

      // permute and copy matrix values
      full_matrix = matrix.template selfadjointView<Eigen::Upper>();
      full_matrix.makeCompressed();
      permute_matrix_values();
    }

    // copy b into x_dev and permute into b_dev
    cudaMemcpy(x_dev, b->map(), b->mem_size(),
               cudaMemcpyKind::cudaMemcpyDefault);
    // cusparseDgthr(sphandle, b->size(), x_dev, b_dev, p_dev,
    //               CUSPARSE_INDEX_BASE_ZERO);
    cusparseGather(sphandle, x_dev_vec, b_dev_vec);

    cudaMemset(x_dev, 0, x->mem_size());

    // auto t0 = std::chrono::high_resolution_clock::now();

    // solve
    cusolverSpDcsrcholFactor(handle, full_matrix.rows(), full_matrix.nonZeros(),
                             desc, data_perm, outerIndices, innerIndices,
                             chol_info, workspace_buffer);

    // check singularity
    int singularity = -1;
    if (cusolverStatus_t::CUSOLVER_STATUS_SUCCESS !=
        cusolverSpDcsrcholZeroPivot(handle, chol_info, 1e-14, &singularity)) {
      std::cerr << "Error: LLTSolverCUDA Zero-Pivot check failed!" << std::endl;
      return false;
    }

    if (singularity != -1) {
      std::cerr << "Error: Matrix is not positive definite!" << std::endl;
      return false;
    }

    auto status = cusolverSpDcsrcholSolve(handle, full_matrix.rows(), b_dev,
                                          x_dev, chol_info, workspace_buffer);

    // auto t1 = std::chrono::high_resolution_clock::now();
    // fmt::print("cuSolver Took: {}\n", std::chrono::duration<double>(t1 -
    // t0).count());

    if (status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "Error: LLTSolverCUDA returned status " << status << "!\n";
      return false;
    }

    // permute result and copy back
    // cusparseDgthr(sphandle, x->size(), x_dev, b_dev, pt_dev,
    //               CUSPARSE_INDEX_BASE_ZERO);
    cusparseGather(sphandle, x_dev_vec, b_dev_vec_pt);

    cudaMemcpy(x->map(), b_dev, x->mem_size(),
               cudaMemcpyKind::cudaMemcpyDefault);

    return true;
  }

  bool result_gpu() override { return false; }

  virtual size_t device_mem() override { return device_bytes; }
};

}  // namespace compute