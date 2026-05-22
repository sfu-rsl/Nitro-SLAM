#pragma once
#include <solver/interfaces.h>
#include <memory>
#include <kompute/Tensor.hpp>
namespace compute
{

    class VulkanComputeEngine;

    template <typename DataType>
    class VulkanComputeBuffer : GPUBufferInterface<DataType>
    {
        friend class VulkanComputeEngine;

    public:
        size_t size() override
        {
            return buffer->size();
        }

        size_t element_size() override
        {
            return sizeof(DataType);
        }

        size_t mem_size() override
        {
            return size() * element_size();
        }

        DataType *map() override
        {
            return buffer->data<DataType>();
        }

        std::shared_ptr<kp::Tensor> get_tensor() const
        {
            return buffer;
        }

        // BufferType get_buffer_type() const;

        BufferType get_buffer_type() const override
        {
            BufferType buffer_type = BufferType::Device;

            switch (buffer->tensorType())
            {
            case kp::Tensor::TensorTypes::eDevice:
                buffer_type = BufferType::Device;
                break;
            case kp::Tensor::TensorTypes::eHost:
                buffer_type = BufferType::Host;
                break;
            case kp::Tensor::TensorTypes::eStorage:
                buffer_type = BufferType::Storage;
                break;
            case kp::Tensor::TensorTypes::eDeviceCached:
                buffer_type = BufferType::DeviceCached;
                break;
            default:
                throw std::runtime_error("Invalid Buffer Type!");
            }

            // return buffer->tensorType() == kp::Tensor::TensorTypes::eHost ? BufferType::Host : BufferType::Device;
            return buffer_type;
        }

        // protected:
        // VulkanComputeBuffer(size_t n, VulkanComputeEngine& engine) {}

        // Ideally this should be a protected constructor
        VulkanComputeBuffer(std::shared_ptr<kp::Tensor> tensor, VulkanComputeEngine *engine) : buffer(tensor), owner(engine){};

    private:
        std::shared_ptr<kp::Tensor> buffer;
        VulkanComputeEngine *owner;
        // size_t size;
    };

    template <typename DataType>
    using VCBPtr = std::shared_ptr<VulkanComputeBuffer<DataType>>;

}