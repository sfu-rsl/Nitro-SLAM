#ifndef CUDA_KEYFRAME_STORAGE_H
#define CUDA_KEYFRAME_STORAGE_H

#include <unordered_map>
#include <mutex>
#include "KeyFrame.h"
#include "Kernels/CudaUtils.h"
#include "Kernels/CudaWrappers/CudaKeyFrame.h"
#include "Kernels/UnifiedChunkAllocator.h"

namespace MAPPING_DATA_WRAPPER {
    class CudaKeyFrame;
}

namespace ORB_SLAM3 {
    class KeyFrame;
    class MapPoint;
}

class CudaKeyFrameStorage {
    public:
        static void initializeMemory();
        static MAPPING_DATA_WRAPPER::CudaKeyFrame* getCudaKeyFrame(long unsigned int mnId);
        static MAPPING_DATA_WRAPPER::CudaKeyFrame* addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static void eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static void printStorageKeyframes();
        static void addFeatureVector(long unsigned int KF_mnId, DBoW2::FeatureVector featVec);
        static void prefetchToDevice();
        static void shutdown();
    public:
        static UnifiedChunkAllocator<MAPPING_DATA_WRAPPER::CudaKeyFrame> allocator;
        static std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyFrame*> mnId_to_kf;
        static int num_keyframes;
        static bool memory_is_initialized, memory_is_free;
        static std::mutex mtx;
};

#endif
