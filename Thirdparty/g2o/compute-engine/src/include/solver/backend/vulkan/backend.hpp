#pragma once
#include <memory>
#include <vector>
#include <mutex>

#include <kompute/Manager.hpp>
#include <kompute/Algorithm.hpp>
#include <kompute/Tensor.hpp>

#include <solver/matrix_types.hpp>
#include <solver/gpu_buffer.hpp>

#include "compute_buffer.hpp"
#include "compute_engine.hpp"

namespace compute
{
    // Forward Declarations
    class VulkanComputeEngine;

    template <typename DataType>
    class VulkanComputeBuffer;

    class VulkanSolverSeq;

    // Aliases
    using ComputeEngine = VulkanComputeEngine;
    template <typename DataType>
    using GPUBuffer = VulkanComputeBuffer<DataType>;

    using SolverSeq = VulkanSolverSeq;
}