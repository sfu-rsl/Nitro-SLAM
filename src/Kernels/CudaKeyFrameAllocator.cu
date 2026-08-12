#include "Kernels/CudaKeyFrameAllocator.h"
#include "Kernels/UnifiedChunkAllocator.h"
#include "Kernels/MappingKernelController.h"
#include "Stats/LocalMappingStats.h"

#include <atomic>

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyFrameAllocator::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

namespace {
    std::atomic<bool> memory_is_initialized(false);
    std::atomic<bool> memory_is_free(false);

    inline UnifiedChunkAllocator<MAPPING_DATA_WRAPPER::CudaKeyFrame>& keyFrameSlots() {
        return UnifiedChunkAllocator<MAPPING_DATA_WRAPPER::CudaKeyFrame>::instance();
    }

    inline bool usable() {
        return memory_is_initialized.load(std::memory_order_acquire) &&
               !memory_is_free.load(std::memory_order_acquire);
    }

    // Mirroring a keyframe before the CUDA side is up would fill the slot from
    // an unset CudaUtils configuration, so treat it as fatal like before.
    inline void abortIfNotInitialized(ORB_SLAM3::KeyFrame* KF) {
        if (memory_is_initialized.load(std::memory_order_acquire))
            return;
        cout << "[ERROR] CudaKeyFrameAllocator::create: memory not initialized (KF "
             << KF->mnId << ")!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }
}

namespace CudaKeyFrameAllocator {

void initialize() {
    memory_is_initialized.store(true, std::memory_order_release);
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* create(ORB_SLAM3::KeyFrame* KF) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif

    abortIfNotInitialized(KF);
    if (memory_is_free.load(std::memory_order_acquire))
        return nullptr;

    MAPPING_DATA_WRAPPER::CudaKeyFrame* existing = KF->GetCudaKeyFrame();
    if (existing != nullptr)
        return existing;

    MAPPING_DATA_WRAPPER::CudaKeyFrame* ptr = keyFrameSlots().allocate();
    ptr->setGPUAddress(ptr);
    ptr->setMemory(KF);

    int device_id;
    cudaGetDevice(&device_id);
    cudaMemPrefetchAsync(ptr, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), device_id);

    // Another thread may have mirrored the same KeyFrame while we were filling
    // ours in; the first one published wins and our slot goes back to the pool.
    if (!KF->SetCudaKeyFrameIfUnset(ptr)) {
        ptr->setAsEmpty();
        keyFrameSlots().deallocate(ptr);
        return KF->GetCudaKeyFrame();
    }

    DEBUG_PRINT("create: " << KF->mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().addCudaKeyFrame_time.push_back(time);
#endif

    return ptr;
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* getOrCreate(ORB_SLAM3::KeyFrame* KF) {
    MAPPING_DATA_WRAPPER::CudaKeyFrame* ptr = KF->GetCudaKeyFrame();
    if (ptr != nullptr)
        return ptr;
    return create(KF);
}

void destroy(ORB_SLAM3::KeyFrame* KF) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif

    MAPPING_DATA_WRAPPER::CudaKeyFrame* ptr = KF->TakeCudaKeyFrame();
    if (ptr == nullptr)
        return;

    // After shutdown() the chunks are gone, so the slot must not be touched.
    if (memory_is_free.load(std::memory_order_acquire))
        return;

    ptr->setAsEmpty();
    keyFrameSlots().deallocate(ptr);

    DEBUG_PRINT("destroy: " << KF->mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().eraseCudaKeyFrame_time.push_back(time);
#endif
}

void addFeatureVector(ORB_SLAM3::KeyFrame* KF, const DBoW2::FeatureVector& featVec) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif

    MAPPING_DATA_WRAPPER::CudaKeyFrame* ptr = getOrCreate(KF);
    if (ptr == nullptr) {
        cout << "[ERROR] CudaKeyFrameAllocator::addFeatureVector: KF " << KF->mnId << " has no GPU mirror!\n";
        return;
    }

    ptr->addFeatureVector(featVec);

    int device_id;
    cudaGetDevice(&device_id);
    cudaMemPrefetchAsync(ptr, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), device_id);

    DEBUG_PRINT("addFeatureVector: " << KF->mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().addFeatureVector_time.push_back(time);
#endif
}

void prefetchToDevice() {
    if (!usable())
        return;
    int device_id;
    cudaGetDevice(&device_id);
    keyFrameSlots().prefetchToDevice(device_id);
}

void shutdown() {
    if (!memory_is_initialized.load(std::memory_order_acquire))
        return;
    if (memory_is_free.exchange(true, std::memory_order_acq_rel))
        return;
    keyFrameSlots().shutdown();
}

int liveKeyFrames() {
    return keyFrameSlots().liveSlots();
}

}
