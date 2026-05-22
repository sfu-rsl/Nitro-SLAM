#pragma once
#include <solver/gpu_buffer.hpp>
namespace compute
{

    template <typename DataType>
    class GPUBufferInterface
    {
    public:
        virtual ~GPUBufferInterface(){};

        virtual size_t size() = 0;

        virtual size_t element_size() = 0;

        virtual size_t mem_size() = 0;

        virtual DataType *map() = 0;

        virtual BufferType get_buffer_type() const = 0;
    };

}