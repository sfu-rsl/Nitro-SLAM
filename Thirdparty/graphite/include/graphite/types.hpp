/// @file types.hpp
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace graphite {

// Type traits to determine if a type is a low precision type (half or bfloat16)

template <typename T> struct is_half_or_bfloat16 : std::false_type {};

template <> struct is_half_or_bfloat16<__half> : std::true_type {};

template <> struct is_half_or_bfloat16<__nv_bfloat16> : std::true_type {};

template <typename T> using is_low_precision = is_half_or_bfloat16<T>;

template <typename T, typename S>
using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

// Vec2 types for different precisions

template <typename T> struct vec2_type;

template <> struct vec2_type<float> { using type = float2; };

template <> struct vec2_type<double> { using type = double2; };

template <> struct vec2_type<__half> { using type = __half2; };

template <> struct vec2_type<__nv_bfloat16> { using type = __nv_bfloat162; };

// Conversion functions from higher precision vec2 to lower precision vec2
template <typename hp, typename lp>
__device__ lp convert_to_low_precision(const hp &value) {
  if constexpr (std::is_same_v<lp, __half2>) {
    return __float22half2_rn(value);
  } else if constexpr (std::is_same_v<lp, __nv_bfloat162>) {
    return __float22bfloat162_rn(value);
  } else {
    return static_cast<lp>(value);
  }
}

// For when we don't need storage
struct Empty {};

} // namespace graphite