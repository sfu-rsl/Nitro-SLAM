#pragma once

#include <cmath>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>

#include <solver/interfaces.h>

#include <solver/matrix_types.hpp>
#include "compute_buffer.hpp"
#include <kompute/Tensor.hpp>
#include <kompute/Manager.hpp>

#define load_spv(SHADER) std::vector<uint32_t>(reinterpret_cast<const uint32_t *>(SHADER), reinterpret_cast<const uint32_t *>(SHADER) + (SHADER##_len / sizeof(uint32_t)));

namespace compute
{

    class VulkanSolverSeq;

    // Specific to Kompute
    template <typename DataType>
    constexpr kp::Tensor::TensorDataTypes get_data_type();

    template <>
    constexpr kp::Tensor::TensorDataTypes get_data_type<double>()
    {
        return kp::Tensor::TensorDataTypes::eDouble;
    }

    template <>
    constexpr kp::Tensor::TensorDataTypes get_data_type<float>()
    {
        return kp::Tensor::TensorDataTypes::eFloat;
    }

    template <>
    constexpr kp::Tensor::TensorDataTypes get_data_type<int32_t>()
    {
        return kp::Tensor::TensorDataTypes::eInt;
    }

    template <>
    constexpr kp::Tensor::TensorDataTypes get_data_type<uint32_t>()
    {
        return kp::Tensor::TensorDataTypes::eUnsignedInt;
    }

    template <>
    constexpr kp::Tensor::TensorDataTypes get_data_type<bool>()
    {
        return kp::Tensor::TensorDataTypes::eBool;
    }

    class VulkanComputeEngine
    {
    public:
        VulkanComputeEngine(uint32_t device_index = 0);

        template <typename DataType>
        VCBPtr<DataType> create_buffer(DataType *data, size_t size, BufferType buffer_type)
        {
            // create tensor and buffer wrapper
            if (size == 0)
            {
                throw std::runtime_error("Tried to create a zero-length buffer!");
            }
            std::shared_ptr<kp::Tensor> tensor = mgr.tensor(data, size, sizeof(DataType), get_data_type<DataType>(), get_tensor_type(buffer_type));
            return std::make_shared<VulkanComputeBuffer<DataType>>(tensor, this);
        }

        std::shared_ptr<vk::Device> get_device()
        {
            return mgr.get_device();
        }

        kp::Manager *get_manager() { return &mgr; }

        uint32_t get_subgroup_size() const
        {
            return subgroup_properties.subgroupSize;
        }

        VmaTotalStatistics get_vma_statistics()
        {
            return mgr.get_vma_statistics();
        }

        // For debugging.
        void print_stats()
        {
            mgr.print_stats();
        }

        // Create object for recording linear solver sequence
        std::shared_ptr<VulkanSolverSeq> create_op_sequence();
        void recycle_sequence(std::shared_ptr<VulkanSolverSeq> seq);
        

        kp::Manager &get_manager_ref()
        {
            return mgr;
        }

    private:
        // private members
        kp::Manager mgr;
        std::mutex seq_mut;
        std::vector<std::shared_ptr<VulkanSolverSeq>> sequences;

    public:
        std::vector<uint32_t> spv_sbm_multiply_subgroup;
        std::vector<uint32_t> spv_sbm_multiply_dist;
        std::vector<uint32_t> spv_sbm_multiply_dynamic;
        std::vector<uint32_t> spv_sbm_multiply_lite_dynamic;
        std::vector<uint32_t> spv_sbm_multiply_estimate_list;
        std::vector<uint32_t> spv_sbm_multiply_build_list;
        std::vector<uint32_t> spv_sbm_multiply_abat;
        std::vector<uint32_t> spv_sbm_multiply_packed;

        std::vector<uint32_t> spv_sbm_left_multiply_block_diagonal;

        std::vector<uint32_t> spv_sbm_bdi;
        std::vector<uint32_t> spv_sbm_multiply_block_diagonal;

        std::vector<uint32_t> spv_set_diagonal;
        std::vector<uint32_t> spv_copy_blocks;

        std::vector<uint32_t> spv_square_array;
        std::vector<uint32_t> spv_reduction;
        std::vector<uint32_t> spv_multiply_vec;
        std::vector<uint32_t> spv_copy_vec;
        std::vector<uint32_t> spv_fill_vec;
        std::vector<uint32_t> spv_add_vec;
        std::vector<uint32_t> spv_div_vec;
        std::vector<uint32_t> spv_add_vec_const_vec;

    private:
        vk::PhysicalDeviceSubgroupProperties subgroup_properties;

        // private functions
        kp::Tensor::TensorTypes get_tensor_type(BufferType buffer_type)
        {
            kp::Tensor::TensorTypes ttype;
            switch (buffer_type)
            {
            case BufferType::Device:
                ttype = kp::Tensor::TensorTypes::eDevice;
                break;
            case BufferType::Host:
                ttype = kp::Tensor::TensorTypes::eHost;
                break;
            case BufferType::Storage:
                ttype = kp::Tensor::TensorTypes::eStorage;
                break;
            case BufferType::DeviceCached:
                ttype = kp::Tensor::TensorTypes::eDeviceCached;
                break;
            default:
                throw std::runtime_error("Invalid buffer type!");
            }
            return ttype;
        }
    };
}

#include "solver.hpp"