/// @file stream.hpp
#pragma once
#include <cuda_runtime.h>

namespace graphite {

class StreamPool {
private:
  cudaStream_t *streams;
  size_t num_streams;
  bool cleanup_streams;

public:
  StreamPool(size_t num_streams);

  StreamPool(cudaStream_t *p_streams, size_t n);
  ~StreamPool();

  cudaStream_t &select(size_t index);

  void sync_all();

  void sync_n(size_t n);
};

} // namespace graphite