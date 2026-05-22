/// @file block_jacobi.hpp
#pragma once
#include <cublas_v2.h>
#include <graphite/preconditioner/preconditioner.hpp>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace graphite {

template <typename T, typename S>
class BlockJacobiPreconditioner : public Preconditioner<T, S> {
private:
  using P = std::conditional_t<is_low_precision<S>::value, T, S>;
  size_t dimension;
  std::vector<std::pair<size_t, size_t>> block_sizes;
  std::unordered_map<BaseVertexDescriptor<T, S> *, thrust::device_vector<P>>
      block_diagonals;

  std::unordered_map<BaseVertexDescriptor<T, S> *, thrust::device_vector<P>>
      scalar_diagonals;

  std::unordered_map<BaseVertexDescriptor<T, S> *, thrust::device_vector<P>>
      P_inv;

  cublasHandle_t handle;

  // For batched inversion
  // TODO: Figure out a better way to handle the memory
  thrust::host_vector<P *> A_ptrs, Ainv_ptrs;
  thrust::device_vector<P *> A_ptrs_device, Ainv_ptrs_device;
  thrust::device_vector<int> info;

public:
  BlockJacobiPreconditioner() {
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  }

  ~BlockJacobiPreconditioner() { cublasDestroy(handle); }

  virtual void update_structure(Graph<T, S> *graph, StreamPool &streams) {

    this->dimension = dimension;
    auto &vertex_descriptors = graph->get_vertex_descriptors();

    for (auto &desc : vertex_descriptors) {
      // Reserve space
      const auto d = desc->dimension();
      const size_t num_values =
          d * d * desc->count(); // includes inactive vertices
      block_diagonals[desc].resize(num_values);
      scalar_diagonals[desc].resize(desc->count() * d);
      P_inv[desc].resize(num_values);
    }

    // Determine max sizes for buffers
    {
      size_t max_num_blocks = 0;
      size_t max_data_size = 0;

      for (auto &desc : vertex_descriptors) {
        const size_t num_blocks = desc->count();
        const size_t d = desc->dimension();
        const size_t block_size = d * d;

        max_num_blocks = std::max(max_num_blocks, num_blocks);
        max_data_size = std::max(max_data_size, num_blocks * block_size);
      }

      A_ptrs.resize(max_num_blocks);
      Ainv_ptrs.resize(max_num_blocks);
      info.resize(max_num_blocks);

      A_ptrs_device.resize(max_num_blocks);
      Ainv_ptrs_device.resize(max_num_blocks);
    }
  };

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) {
    const cudaStream_t stream = 0;
    auto &vertex_descriptors = graph->get_vertex_descriptors();
    auto &factor_descriptors = graph->get_factor_descriptors();
    auto jacobian_scales = graph->get_jacobian_scales().data().get();

    // Compute Hessian blocks on the diagonal
    for (auto &desc : vertex_descriptors) {
      thrust::fill(thrust::cuda::par_nosync.on(stream),
                   block_diagonals[desc].begin(), block_diagonals[desc].end(),
                   static_cast<S>(0.0));
    }
    for (auto &desc : factor_descriptors) {
      desc->compute_hessian_block_diagonal_async(block_diagonals,
                                                 jacobian_scales, stream);
    }
    // back up diagonals for each vertex descriptor
    for (auto &desc : vertex_descriptors) {

      auto b = block_diagonals[desc].data().get();
      auto s = scalar_diagonals[desc].data().get();

      auto start = thrust::make_counting_iterator<size_t>(0);
      auto end = start + scalar_diagonals[desc].size();
      const size_t D = desc->dimension();
      thrust::for_each(thrust::cuda::par_nosync.on(stream), start, end,
                       [b, s, D] __device__(const size_t idx) {
                         const size_t vertex_id = idx / D;
                         const auto block = b + vertex_id * D * D;
                         const size_t col = idx % D;
                         s[idx] = block[col * D + col];
                       });
    }

    cudaStreamSynchronize(stream);
  };

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  StreamPool &streams) {

    const cudaStream_t stream = 0;
    cublasSetStream(handle, stream);
    auto &vertex_descriptors = graph->get_vertex_descriptors();
    auto &factor_descriptors = graph->get_factor_descriptors();

    // Invert the blocks

    for (auto &desc : vertex_descriptors) {
      desc->augment_block_diagonal_async(block_diagonals[desc].data().get(),
                                         scalar_diagonals[desc].data().get(),
                                         damping_factor, stream);

      // Invert the block diagonal using cublas
      const auto d = desc->dimension();
      const size_t num_blocks = desc->count();
      const auto block_size = d * d;
      const size_t data_size = num_blocks * block_size;

      P *a_ptr = block_diagonals[desc].data().get();

      P *a_inv_ptr = P_inv[desc].data().get();
      for (size_t i = 0; i < num_blocks; ++i) {
        A_ptrs[i] = a_ptr + i * block_size;
        Ainv_ptrs[i] = a_inv_ptr + i * block_size;
      }

      cudaMemcpyAsync(A_ptrs_device.data().get(), A_ptrs.data(),
                      sizeof(P *) * num_blocks, cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(Ainv_ptrs_device.data().get(), Ainv_ptrs.data(),
                      sizeof(P *) * num_blocks, cudaMemcpyHostToDevice, stream);

      // cublas should use stream 0
      if constexpr (std::is_same<P, double>::value) {

        cublasDmatinvBatched(handle, d, A_ptrs_device.data().get(), d,
                             Ainv_ptrs_device.data().get(), d,
                             info.data().get(), num_blocks);
      } else if constexpr (std::is_same<P, float>::value) {
        cublasSmatinvBatched(handle, d, A_ptrs_device.data().get(), d,
                             Ainv_ptrs_device.data().get(), d,
                             info.data().get(), num_blocks);
      } else {
        static_assert(
            is_low_precision<S>::value || std::is_same<S, float>::value ||
                std::is_same<S, double>::value,
            "BlockJacobiPreconditioner only supports bfloat16, float, or "
            "double types.");
      }
    }

    // Final sync
    cudaStreamSynchronize(stream);
  };

  void apply(Graph<T, S> *graph, T *z, const T *r,
             StreamPool &streams) override {
    // Apply the preconditioner
    size_t i = 0;
    auto &vertex_descriptors = graph->get_vertex_descriptors();
    for (auto &desc : vertex_descriptors) {
      const auto d = desc->dimension();
      P *blocks = P_inv[desc].data().get();
      desc->apply_block_jacobi(z, r, blocks, streams.select(i));
      i++;
    }
    streams.sync_n(i);
  }
};

} // namespace graphite