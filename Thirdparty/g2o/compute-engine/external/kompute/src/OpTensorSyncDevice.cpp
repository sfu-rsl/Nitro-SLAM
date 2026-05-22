// SPDX-License-Identifier: Apache-2.0

#include "kompute/operations/OpTensorSyncDevice.hpp"

namespace kp {

OpTensorSyncDevice::OpTensorSyncDevice(
  const std::vector<std::shared_ptr<Tensor>>& tensors, const std::vector<std::pair<uint32_t, uint32_t>>& ranges)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncDevice constructor with params");

    if (tensors.size() < 1) {
        throw std::runtime_error(
          "Kompute OpTensorSyncDevice called with less than 1 tensor");
    }

    this->mTensors = tensors;
    this->mRanges = ranges;
}

OpTensorSyncDevice::~OpTensorSyncDevice()
{
    KP_LOG_DEBUG("Kompute OpTensorSyncDevice destructor started");

    this->mTensors.clear();
}

void
OpTensorSyncDevice::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncDevice record called");

    for (size_t i = 0; i < this->mTensors.size(); i++) {
        if (this->mTensors[i]->tensorType() == Tensor::TensorTypes::eDevice || this->mTensors[i]->tensorType() == Tensor::TensorTypes::eDeviceCached) {
            vk::BufferCopy copyRegion;
            vk::BufferCopy* pRegion = nullptr;
            if (mRanges.size() > 0) {
                auto dt_size = this->mTensors[i]->dataTypeMemorySize();
                copyRegion = vk::BufferCopy(mRanges.at(i).first*dt_size, mRanges.at(i).first*dt_size, mRanges.at(i).second*dt_size);
                pRegion = &copyRegion;
            }            
            this->mTensors[i]->recordCopyFromStagingToDevice(commandBuffer, pRegion);
        }
    }
}

void
OpTensorSyncDevice::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncDevice preEval called");

    // Should flush memory before the copy is done
    for (size_t i = 0; i < this->mTensors.size(); i++) {
        if (this->mTensors[i]->tensorType() == Tensor::TensorTypes::eDeviceCached) {
            this->mTensors[i]->flush();
        }
    }
}

void
OpTensorSyncDevice::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncDevice postEval called");
}

}
