/// @file common.hpp
#pragma once
#include <array>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda/std/tuple>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <graphite/dual.hpp>

namespace graphite {
#define hd_fn __host__ __device__
#define d_fn __device__
} // namespace graphite