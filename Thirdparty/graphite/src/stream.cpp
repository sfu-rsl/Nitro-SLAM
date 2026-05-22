#include <graphite/stream.hpp>
#include <algorithm>
namespace graphite {

  StreamPool::StreamPool(size_t num_streams)
      : num_streams(num_streams), cleanup_streams(true) {
    streams = new cudaStream_t[num_streams];
    for (size_t i = 0; i < num_streams; ++i) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
  }

  StreamPool::StreamPool(cudaStream_t *p_streams, size_t n)
      : streams(p_streams), num_streams(n), cleanup_streams(false) {}

  StreamPool::~StreamPool() {
    if (cleanup_streams) {
        for (size_t i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
        delete[] streams;
    }
  }

  cudaStream_t &StreamPool::select(size_t index) { return streams[index % num_streams]; }

  void StreamPool::sync_all() {
    for (size_t i = 0; i < num_streams; ++i) {
      cudaStreamSynchronize(streams[i]);
    }
  }

  void StreamPool::sync_n(size_t n) {
    n = std::min(n, num_streams);
    for (size_t i = 0; i < n; ++i) {
      cudaStreamSynchronize(streams[i]);
    }
  }

}