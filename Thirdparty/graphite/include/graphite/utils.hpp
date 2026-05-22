/// @file utils.hpp
#pragma once
#include <array>
#include <graphite/vector.hpp>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

namespace graphite {

template <typename T>
void prefetch_vector_on_device_async(const thrust::universal_vector<T> &vec,
                                     int device_id, cudaStream_t stream) {
// Prefetch the vector to the device
// std::cout << "Prefetching vector of size " << vec.size() << " to device " <<
// device_id << std::endl;
#if !defined(_WIN32) && !defined(__WSL2__)
  cudaMemPrefetchAsync(vec.data().get(), vec.size() * sizeof(T), device_id,
                       stream);
#endif
}

template <typename T>
void prefetch_vector_on_device_async(const managed_vector<T> &vec,
                                     int device_id, cudaStream_t stream) {
// Prefetch the vector to the device
// std::cout << "Prefetching vector of size " << vec.size() << " to device " <<
// device_id << std::endl;
#if !defined(_WIN32) && !defined(__WSL2__)
  cudaMemPrefetchAsync(vec.data().get(), vec.size() * sizeof(T), device_id,
                       stream);
#endif
}

template <typename T>
void prefetch_vector_on_host(const thrust::universal_vector<T> &vec,
                             cudaStream_t stream) {
// Prefetch the vector to the host
// std::cout << "Prefetching vector of size " << vec.size() << " to host" <<
// std::endl;
#if !defined(_WIN32) && !defined(__WSL2__)
  cudaMemPrefetchAsync(vec.data().get(), vec.size() * sizeof(T),
                       cudaCpuDeviceId, stream);
  cudaStreamSynchronize(stream);
#endif
}

template <typename T>
void prefetch_vector_on_host(const managed_vector<T> &vec,
                             cudaStream_t stream) {
// Prefetch the vector to the host
// std::cout << "Prefetching vector of size " << vec.size() << " to host" <<
// std::endl;
#if !defined(_WIN32) && !defined(__WSL2__)
  cudaMemPrefetchAsync(vec.data().get(), vec.size() * sizeof(T),
                       cudaCpuDeviceId, stream);
  cudaStreamSynchronize(stream);
#endif
}

template <typename T>
void print_device_vector(const thrust::device_vector<T> &vec) {
  std::vector<T> host_vec(vec.size());
  thrust::copy(vec.begin(), vec.end(), host_vec.begin());
  for (const auto &val : host_vec) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

template <typename T> void print_device_vector(const T *vec, size_t size) {
  std::vector<T> host_vec(size);
  cudaMemcpy(host_vec.data(), vec, size * sizeof(T), cudaMemcpyDeviceToHost);
  for (const auto &val : host_vec) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

template <typename H> class HandleManager {
private:
  std::vector<H> handles;
  H last_handle;

public:
  HandleManager() : last_handle(0) {}

  H get() {
    if (handles.empty()) {
      return last_handle++;
    } else {
      H handle = handles.back();
      handles.pop_back();
      return handle;
    }
  }

  void release(H handle) { handles.push_back(handle); }

  void clear() {
    handles.clear();
    last_handle = 0;
  }
};

// Based on method from Boost
// https://www.boost.org/doc/libs/latest/libs/container_hash/doc/html/hash.html#ref_hash_combine

size_t mix(size_t x) {
  x ^= x >> 32;
  x *= 0xe9846af9b1a615d;
  x ^= x >> 32;
  x *= 0xe9846af9b1a615d;
  x ^= x >> 28;
  return x;
}

template <typename T> void hash_combine(size_t &seed, T const &v) {
  seed = mix(seed + 0x9e3779b9 + std::hash<T>{}(v));
}

} // namespace graphite