#ifndef CUDA_KEYFRAME_ALLOCATOR_H
#define CUDA_KEYFRAME_ALLOCATOR_H

#include "KeyFrame.h"
#include "Kernels/CudaWrappers/CudaKeyFrame.h"

namespace MAPPING_DATA_WRAPPER {
    class CudaKeyFrame;
}

namespace ORB_SLAM3 {
    class KeyFrame;
}

// Lifecycle helpers for the unified-memory CudaKeyFrame that mirrors an
// ORB_SLAM3::KeyFrame on the GPU. There is no id -> pointer table: each
// KeyFrame owns the pointer to its own CudaKeyFrame (KeyFrame::GetCudaKeyFrame),
// and the slots themselves come from the thread-safe
// UnifiedChunkAllocator<CudaKeyFrame> singleton.
namespace CudaKeyFrameAllocator {
    // Must be called once the CUDA context and CudaUtils are ready, before any
    // KeyFrame reaches the GPU. create() is fatal before it.
    void initialize();

    // The single way to get at KF's mirror: returns the one KF already has, or
    // allocates and fills a slot, publishes it on KF and returns that. Returns
    // nullptr once the allocator has been shut down.
    MAPPING_DATA_WRAPPER::CudaKeyFrame* create(ORB_SLAM3::KeyFrame* KF);

    // Detaches KF's mirror and returns its slot to the free list.
    void destroy(ORB_SLAM3::KeyFrame* KF);

    // Frees every slot. Safe to call from more than one controller.
    void shutdown();
}

#endif
