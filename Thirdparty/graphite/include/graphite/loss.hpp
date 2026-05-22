/// @file loss.hpp
#pragma once

namespace graphite {

template <typename T, int E> class Loss {
public:
  virtual __device__ __host__ ~Loss() {}

  virtual __device__ __host__ T loss(const T &x) const = 0;

  virtual __device__ __host__ T loss_derivative(const T &x) const = 0;
};

template <typename T, int E> class DefaultLoss final : public Loss<T, E> {
public:
  __device__ __host__ DefaultLoss() {}
  __device__ __host__ DefaultLoss(const DefaultLoss &other) {}

  __device__ __host__ T loss(const T &x) const override { return x; };

  __device__ __host__ T loss_derivative(const T &x) const override {
    return 1;
  };
};

template <typename T, int E> class HuberLoss final : public Loss<T, E> {
public:
  T delta;

  __device__ __host__ HuberLoss() : delta(100.0) {}
  __device__ __host__ HuberLoss(const HuberLoss &other) : delta(other.delta) {}

  __device__ __host__ HuberLoss(T delta) : delta(delta) {}

  __device__ __host__ T loss(const T &x) const override {
    if (x <= delta * delta) {
      return x;
    } else {
      return 2 * std::sqrt(x) * delta - delta * delta;
    }
  }

  __device__ __host__ T loss_derivative(const T &x) const override {
    if (x <= delta * delta) {
      return 1;
    } else {
      return delta / std::sqrt(x);
    }
  }
};

} // namespace graphite