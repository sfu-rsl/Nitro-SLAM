/// @file vector.hpp
#pragma once
#include <thrust/device_ptr.h>
#include <thrust/universal_vector.h>
namespace graphite {

// https://github.com/NVIDIA/cccl/issues/810

template <typename T>
struct uninitialized_allocator : thrust::cuda::universal_allocator<T> {
  __host__ uninitialized_allocator() {}
  __host__ uninitialized_allocator(const uninitialized_allocator &other)
      : thrust::cuda::universal_allocator<T>(other) {}
  __host__ ~uninitialized_allocator() {}

  uninitialized_allocator &operator=(const uninitialized_allocator &) = default;

  template <typename U> struct rebind {
    typedef uninitialized_allocator<U> other;
  };

  __host__ __device__ void construct(T *) {}
};

template <typename T> class managed_vector {

private:
  size_t m_capacity;
  size_t m_size;

  thrust::universal_ptr<T> m_data;
  uninitialized_allocator<T> alloc;
  // thrust::cuda::universal_allocator<T> alloc;

  void deallocate() {
    if (m_capacity > 0) {
      for (size_t i = 0; i < m_size; i++) {
        m_data[i].~T();
      }
      alloc.deallocate(m_data, m_capacity);
      m_data = nullptr;
    }
  }

public:
  managed_vector() : m_capacity(0), m_size(0), m_data(nullptr) {}
  managed_vector(size_t size)
      : m_capacity(size), m_size(size), m_data(alloc.allocate(size)) {}

  managed_vector(const managed_vector &) = delete;
  managed_vector &operator=(const managed_vector &) = delete;
  managed_vector(managed_vector &&) = delete;
  managed_vector &operator=(managed_vector &&) = delete;

  ~managed_vector() { deallocate(); }

  size_t capacity() const { return m_capacity; }

  size_t size() const { return m_size; }

  void reserve(size_t size) {
    if (size > m_capacity) {

      thrust::universal_ptr<T> new_data = alloc.allocate(size);

      if (m_size > 0) {
        for (size_t i = 0; i < m_size; i++) {
          new (new_data.get() + i) T(m_data[i]);
        }
      }

      deallocate();

      m_data = new_data;
      m_capacity = size;
    }
  }

  void resize(size_t size) {
    if (size < m_size) {
      for (size_t i = size; i < m_size; i++) {
        m_data[i].~T();
      }
    } else if (size > m_capacity) {
      reserve(size);
    }
    m_size = size;
  }

  void push_back(const T &value) {
    if (m_size == m_capacity) {
      reserve(m_capacity == 0 ? 1 : m_capacity * 2);
    }
    new (m_data.get() + m_size) T(value);
    m_size++;
  }

  void pop_back() {
    if (m_size > 0) {
      m_data[m_size - 1].~T();
      m_size--;
    }
  }

  T &operator[](size_t index) { return m_data[index]; }
  const T &operator[](size_t index) const { return m_data[index]; }

  T &back() { return m_data[m_size - 1]; }

  const T &back() const { return m_data[m_size - 1]; }

  T *begin() { return m_data.get(); }

  T *end() { return m_data.get() + m_size; }

  thrust::universal_ptr<T> data() { return m_data; }

  const thrust::universal_ptr<T> data() const { return m_data; }

  void clear() {
    for (size_t i = 0; i < m_size; i++) {
      m_data[i].~T();
    }
    m_size = 0;
  }
};

} // namespace graphite