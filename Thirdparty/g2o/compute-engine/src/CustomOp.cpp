#include <solver/kp/CustomOp.hpp>
namespace kp
{

    OpGlobalMemoryBarrier::OpGlobalMemoryBarrier(
        const std::vector<std::shared_ptr<Tensor>> &tensors,
        const vk::AccessFlagBits &srcAccessMask,
        const vk::AccessFlagBits &dstAccessMask,
        const vk::PipelineStageFlagBits &srcStageMask,
        const vk::PipelineStageFlagBits &dstStageMask)
        : mSrcAccessMask(srcAccessMask), mDstAccessMask(dstAccessMask), mSrcStageMask(srcStageMask), mDstStageMask(dstStageMask), mTensors(tensors)
    {
        KP_LOG_DEBUG("Kompute OpGlobalMemoryBarrier constructor");
    }

    OpGlobalMemoryBarrier::~OpGlobalMemoryBarrier()
    {
        KP_LOG_DEBUG("Kompute OpGlobalMemoryBarrier destructor started");
    }

    void
    OpGlobalMemoryBarrier::record(const vk::CommandBuffer &commandBuffer)
    {
        KP_LOG_DEBUG("Kompute OpGlobalMemoryBarrier record called");

        vk::MemoryBarrier memoryBarrier;
        memoryBarrier.srcAccessMask = mSrcAccessMask;
        memoryBarrier.dstAccessMask = mDstAccessMask;

        commandBuffer.pipelineBarrier(
            mSrcStageMask,
            mDstStageMask,
            vk::DependencyFlags(),
            memoryBarrier,
            nullptr,
            nullptr);
    }

    void
    OpGlobalMemoryBarrier::preEval(const vk::CommandBuffer &commandBuffer)
    {
        KP_LOG_DEBUG("Kompute OpGlobalMemoryBarrier preEval called");
        (void)(commandBuffer);
    }

    void
    OpGlobalMemoryBarrier::postEval(const vk::CommandBuffer &commandBuffer)
    {
        KP_LOG_DEBUG("Kompute OpGlobalMemoryBarrier postSubmit called");
        (void)(commandBuffer);
    }

}
