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
    // KeyFrame reaches the GPU. create()/getOrCreate() are no-ops before it.
    void initialize();

    // Allocates and fills the GPU mirror of KF, publishes it on KF and returns
    // it. Returns the existing mirror if KF already has one.
    MAPPING_DATA_WRAPPER::CudaKeyFrame* create(ORB_SLAM3::KeyFrame* KF);

    // KF's mirror, creating it on demand. Returns nullptr if the allocator is
    // not initialized or has already been shut down.
    MAPPING_DATA_WRAPPER::CudaKeyFrame* getOrCreate(ORB_SLAM3::KeyFrame* KF);

    // Detaches KF's mirror and returns its slot to the free list.
    void destroy(ORB_SLAM3::KeyFrame* KF);

    void addFeatureVector(ORB_SLAM3::KeyFrame* KF, const DBoW2::FeatureVector& featVec);

    void prefetchToDevice();

    // Frees every slot. Safe to call from more than one controller.
    void shutdown();

    int liveKeyFrames();
}

#endif
