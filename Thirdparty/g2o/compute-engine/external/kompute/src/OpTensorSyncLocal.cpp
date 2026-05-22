// SPDX-License-Identifier: Apache-2.0

#include "kompute/Tensor.hpp"

#include "kompute/operations/OpTensorSyncLocal.hpp"
namespace kp {

OpTensorSyncLocal::OpTensorSyncLocal(
  const std::vector<std::shared_ptr<Tensor>>& tensors, const std::vector<std::pair<uint32_t, uint32_t>>& ranges)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal constructor with params");

    if (tensors.size() < 1) {
        throw std::runtime_error(
          "Kompute OpTensorSyncLocal called with less than 1 tensor");
    }

    this->mTensors = tensors;
    this->mRanges = ranges;
}

OpTensorSyncLocal::~OpTensorSyncLocal()
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal destructor started");
}

void
OpTensorSyncLocal::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal record called");

    for (size_t i = 0; i < this->mTensors.size(); i++) {
        if (this->mTensors[i]->tensorType() == Tensor::TensorTypes::eDevice || this->mTensors[i]->tensorType() == Tensor::TensorTypes::eDeviceCached) {

            this->mTensors[i]->recordPrimaryBufferMemoryBarrier(
              commandBuffer,
              vk::AccessFlagBits::eShaderWrite,
              vk::AccessFlagBits::eTransferRead,
              vk::PipelineStageFlagBits::eComputeShader,
              vk::PipelineStageFlagBits::eTransfer);


            vk::BufferCopy copyRegion;
            vk::BufferCopy* pRegion = nullptr;
            if (mRanges.size() > 0) {
                auto dt_size = this->mTensors[i]->dataTypeMemorySize();
                copyRegion = vk::BufferCopy(mRanges.at(i).first*dt_size, mRanges.at(i).first*dt_size, mRanges.at(i).second*dt_size);
                pRegion = &copyRegion;
            }
            this->mTensors[i]->recordCopyFromDeviceToStaging(commandBuffer, pRegion);

            this->mTensors[i]->recordPrimaryBufferMemoryBarrier(
              commandBuffer,
              vk::AccessFlagBits::eTransferWrite,
              vk::AccessFlagBits::eHostRead,
              vk::PipelineStageFlagBits::eTransfer,
              vk::PipelineStageFlagBits::eHost);
        }
    }
}

void
OpTensorSyncLocal::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal preEval called");
}

void
OpTensorSyncLocal::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal postEval called");

    KP_LOG_DEBUG("Kompute OpTensorSyncLocal mapping data into tensor local");

    // ranges should be invalidated for the cached CPU visible buffers
    for (size_t i = 0; i < this->mTensors.size(); i++) {
        if (this->mTensors[i]->tensorType() == Tensor::TensorTypes::eDeviceCached) {
            this->mTensors[i]->invalidate();
        }
    }
}

}
