#include <solver/backend/vulkan/compute_engine.hpp>

// Shader code

#include <solver/shaders/shadersbm_bdi.hpp>
#include <solver/shaders/shadersbm_multiply_block_diagonal.hpp>
#include <solver/shaders/shadersbm_left_multiply_block_diagonal.hpp>
#include <solver/shaders/shadersbm_multiply_lite.hpp>
#include <solver/shaders/shadersbm_multiply_dist.hpp>
#include <solver/shaders/shadersbm_multiply_dynamic.hpp>
#include <solver/shaders/shadersbm_multiply_lite_dynamic.hpp>
#include <solver/shaders/shadersbm_multiply_abat.hpp>
#include <solver/shaders/shadersbm_multiply_packed.hpp>

#include <solver/shaders/shadersbm_multiply_estimate_list.hpp>
#include <solver/shaders/shadersbm_multiply_build_list.hpp>

#include <solver/shaders/shaderset_diagonal.hpp>
#include <solver/shaders/shadercopy_blocks.hpp>

#include <solver/shaders/shadersquare_array.hpp>
#include <solver/shaders/shaderreduction.hpp>
#include <solver/shaders/shadermultiply_vec.hpp>
#include <solver/shaders/shadercopy_vec.hpp>
#include <solver/shaders/shaderfill_vec.hpp>
#include <solver/shaders/shaderadd_vec.hpp>
#include <solver/shaders/shaderdiv_vec.hpp>
#include <solver/shaders/shaderadd_vec_const_vec.hpp>

// End of shader code


namespace compute {


    VulkanComputeEngine::VulkanComputeEngine(uint32_t device_index): mgr(device_index)
        {
            // spv_sbm_multiply_subgroup = load_spv(kp::shader_data::shaders_sbm_multiply_subgroup_comp_spv);
            spv_sbm_multiply_subgroup = load_spv(kp::shader_data::shaders_sbm_multiply_lite_comp_spv);
            spv_sbm_multiply_dist = load_spv(kp::shader_data::shaders_sbm_multiply_dist_comp_spv);
            spv_sbm_multiply_dynamic = load_spv(kp::shader_data::shaders_sbm_multiply_dynamic_comp_spv);
            spv_sbm_multiply_lite_dynamic = load_spv(kp::shader_data::shaders_sbm_multiply_lite_dynamic_comp_spv);
            spv_sbm_multiply_abat = load_spv(kp::shader_data::shaders_sbm_multiply_abat_comp_spv);
            spv_sbm_multiply_packed = load_spv(kp::shader_data::shaders_sbm_multiply_packed_comp_spv);

            spv_sbm_multiply_estimate_list = load_spv(kp::shader_data::shaders_sbm_multiply_estimate_list_comp_spv);
            spv_sbm_multiply_build_list = load_spv(kp::shader_data::shaders_sbm_multiply_build_list_comp_spv);

            spv_sbm_bdi = load_spv(kp::shader_data::shaders_sbm_bdi_comp_spv);
            spv_sbm_multiply_block_diagonal = load_spv(kp::shader_data::shaders_sbm_multiply_block_diagonal_comp_spv);
            spv_sbm_left_multiply_block_diagonal = load_spv(kp::shader_data::shaders_sbm_left_multiply_block_diagonal_comp_spv);
            spv_set_diagonal = load_spv(kp::shader_data::shaders_set_diagonal_comp_spv);
            spv_copy_blocks = load_spv(kp::shader_data::shaders_copy_blocks_comp_spv);

            spv_square_array = load_spv(kp::shader_data::shaders_square_array_comp_spv);
            spv_reduction = load_spv(kp::shader_data::shaders_reduction_comp_spv);
            spv_multiply_vec = load_spv(kp::shader_data::shaders_multiply_vec_comp_spv);
            spv_copy_vec = load_spv(kp::shader_data::shaders_copy_vec_comp_spv);
            spv_fill_vec = load_spv(kp::shader_data::shaders_fill_vec_comp_spv);
            spv_add_vec = load_spv(kp::shader_data::shaders_add_vec_comp_spv);
            spv_div_vec = load_spv(kp::shader_data::shaders_div_vec_comp_spv);
            spv_add_vec_const_vec = load_spv(kp::shader_data::shaders_add_vec_const_vec_comp_spv);

            // store subgroup properties
            auto pd = mgr.get_physical_device();
            vk::PhysicalDeviceProperties2 props;
            props.pNext = &subgroup_properties;
            pd->getProperties2(&props);
        }


    std::shared_ptr<VulkanSolverSeq> VulkanComputeEngine::create_op_sequence()
    {
        seq_mut.lock();

        if (!sequences.empty()) {
            auto seq = sequences.back();
            sequences.pop_back();
            seq_mut.unlock();

            return seq;
        }
        seq_mut.unlock();
        
        return std::make_shared<VulkanSolverSeq>(mgr.sequence(), mgr.sequence(), this);
    }
    void VulkanComputeEngine::recycle_sequence(std::shared_ptr<VulkanSolverSeq> seq) {
        if (!seq) {
            return;
        }
        seq->clear();
        seq_mut.lock();
        sequences.push_back(seq);
        seq_mut.unlock();
    }

}