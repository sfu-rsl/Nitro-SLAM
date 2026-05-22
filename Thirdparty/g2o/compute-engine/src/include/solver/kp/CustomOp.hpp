#pragma once
#include <vulkan/vulkan.hpp>
#include <kompute/Tensor.hpp>
#include <kompute/operations/OpBase.hpp>
namespace kp
{
    class OpGlobalMemoryBarrier : public OpBase
    {
    public:
        OpGlobalMemoryBarrier(
            const std::vector<std::shared_ptr<Tensor>> &tensors,
            const vk::AccessFlagBits &srcAccessMask,
            const vk::AccessFlagBits &dstAccessMask,
            const vk::PipelineStageFlagBits &srcStageMask,
            const vk::PipelineStageFlagBits &dstStageMask);

        virtual ~OpGlobalMemoryBarrier() override;

        virtual void record(const vk::CommandBuffer &commandBuffer) override;

        virtual void preEval(const vk::CommandBuffer &commandBuffer) override;

        virtual void postEval(const vk::CommandBuffer &commandBuffer) override;

    private:
        const vk::AccessFlagBits mSrcAccessMask;
        const vk::AccessFlagBits mDstAccessMask;
        const vk::PipelineStageFlagBits mSrcStageMask;
        const vk::PipelineStageFlagBits mDstStageMask;
        const std::vector<std::shared_ptr<Tensor>> mTensors;
    };

}
