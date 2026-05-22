/// @file dual.hpp
#pragma once
#include <cuda/std/cmath>
#include <cuda/std/limits>

namespace graphite {

template <typename T, typename D> struct Dual {
  T real;
  D dual;

  using DT = Dual<T, D>;

  __host__ __device__ Dual() : real(0), dual(0) {}
  __host__ __device__ Dual(T real, D dual) : real(real), dual(dual) {}
  __host__ __device__ Dual(T real) : real(real), dual(0) {}

  __host__ __device__ DT operator+(const DT &other) const {
    return DT(real + other.real, dual + other.dual);
  }

  __host__ __device__ DT operator-(const DT &other) const {
    return DT(real - other.real, dual - other.dual);
  }

  __host__ __device__ DT operator-() const { return DT(-real, -dual); }

  __host__ __device__ DT operator*(const DT &other) const {
    return DT(real * other.real, real * other.dual + dual * other.real);
  }

  __host__ __device__ DT operator/(const DT &other) const {
    if (other.real == 0) {
      // Handle division by zero case
      return DT(std::numeric_limits<T>::infinity(),
                std::numeric_limits<T>::infinity());
    }
    T denominator = other.real * other.real;
    return DT((real * other.real) / denominator,
              (dual * other.real - real * other.dual) / denominator);
  }

  __host__ __device__ DT &operator+=(const DT &other) {
    real += other.real;
    dual += other.dual;
    return *this;
  }

  __host__ __device__ DT &operator-=(const DT &other) {
    real -= other.real;
    dual -= other.dual;
    return *this;
  }

  __host__ __device__ DT &operator*=(const DT &other) {
    T new_real = real * other.real;
    dual = real * other.dual + dual * other.real;
    real = new_real;
    return *this;
  }

  __host__ __device__ DT &operator/=(const DT &other) {
    if (other.real == 0) {
      // Handle division by zero case
      real = std::numeric_limits<T>::infinity();
      dual = std::numeric_limits<T>::infinity();
      return *this;
    }
    T denominator = other.real * other.real;
    T new_real = (real * other.real) / denominator;
    dual = (dual * other.real - real * other.dual) / denominator;
    real = new_real;
    return *this;
  }

  __host__ __device__ friend DT sin(const DT &x) {
    return DT(std::sin(x.real), x.dual * std::cos(x.real));
  }

  __host__ __device__ friend DT cos(const DT &x) {
    return DT(std::cos(x.real), -x.dual * std::sin(x.real));
  }

  __host__ __device__ friend DT atan(const DT &x) {
    return DT(std::atan(x.real), x.dual / (1 + x.real * x.real));
  }

  __host__ __device__ friend DT acos(const DT &x) {
    return DT(std::acos(x.real), -x.dual / std::sqrt(1 - x.real * x.real));
  }

  __host__ __device__ friend DT exp(const DT &x) {
    T exp_real = std::exp(x.real);
    return DT(exp_real, x.dual * exp_real);
  }

  __host__ __device__ friend DT log(const DT &x) {
    return DT(std::log(x.real), x.dual / x.real);
  }

  __host__ __device__ friend DT sqrt(const DT &x) {
    T sqrt_real = std::sqrt(x.real);
    return DT(sqrt_real, x.dual / (2 * sqrt_real));
  }

  __host__ __device__ friend DT abs(const DT &x) {
    if (x.real < T(0.0)) {
      return -x;
    }
    return x;
  }

  __host__ __device__ bool operator<(const DT &other) const {
    return real < other.real;
  }

  __host__ __device__ bool operator>(const DT &other) const {
    return real > other.real;
  }

  __host__ __device__ bool operator<=(const DT &other) const {
    return real <= other.real;
  }

  __host__ __device__ bool operator>=(const DT &other) const {
    return real >= other.real;
  }
};

} // namespace graphite
