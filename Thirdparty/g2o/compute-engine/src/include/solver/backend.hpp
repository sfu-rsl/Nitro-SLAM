#pragma once

#include <memory>

// Default to vulkan backend
#include <solver/backend/vulkan/backend.hpp>

namespace compute
{

    template <typename DataType>
    using BufferPtr = std::shared_ptr<GPUBuffer<DataType>>;

}