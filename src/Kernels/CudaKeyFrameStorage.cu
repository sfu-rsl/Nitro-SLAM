#include "Kernels/CudaKeyFrameStorage.h"
#include "Kernels/MappingKernelController.h"
#include "Stats/LocalMappingStats.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyFrameStorage::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

UnifiedChunkAllocator<MAPPING_DATA_WRAPPER::CudaKeyFrame> CudaKeyFrameStorage::allocator;
std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyFrame*> CudaKeyFrameStorage::mnId_to_kf;
int CudaKeyFrameStorage::num_keyframes = 0;
bool CudaKeyFrameStorage::memory_is_initialized = false;
bool CudaKeyFrameStorage::memory_is_free = false;
std::mutex CudaKeyFrameStorage::mtx;


void CudaKeyFrameStorage::initializeMemory() {
    if (memory_is_initialized) return;
    memory_is_initialized = true;
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrameStorage::addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized) {
        cout << "[ERROR] CudaKeyFrameStorage::addCudaKeyFrame: memory not initialized!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    auto it = mnId_to_kf.find(KF->mnId);
    if (it != mnId_to_kf.end()) {
        cout << "CudaKeyFrameStorage::addCudaKeyFrame: KF " << KF->mnId << " is already on GPU.\n";
        return it->second;
    }

    MAPPING_DATA_WRAPPER::CudaKeyFrame* ptr = allocator.allocate();
    ptr->setGPUAddress(ptr);
    ptr->setMemory(KF);

    int device_id;
    cudaGetDevice(&device_id);
    cudaMemPrefetchAsync(ptr, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), device_id);

    mnId_to_kf.emplace(KF->mnId, ptr);
    num_keyframes++;

    DEBUG_PRINT("addCudaKeyFrame: " << KF->mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().addCudaKeyFrame_time.push_back(time);
#endif

    return ptr;
}

void CudaKeyFrameStorage::eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif

    auto it = mnId_to_kf.find(KF->mnId);
    if (it == mnId_to_kf.end()) {
        cout << "CudaKeyFrameStorage::eraseCudaKeyFrame: KF " << KF->mnId << " not in GPU storage!\n";
        return;
    }

    MAPPING_DATA_WRAPPER::CudaKeyFrame* ptr = it->second;
    ptr->setAsEmpty();
    allocator.deallocate(ptr);
    mnId_to_kf.erase(it);
    num_keyframes--;

    DEBUG_PRINT("eraseCudaKeyFrame: " << KF->mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().eraseCudaKeyFrame_time.push_back(time);
#endif
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrameStorage::getCudaKeyFrame(long unsigned int mnId) {
    auto it = mnId_to_kf.find(mnId);
    if (it != mnId_to_kf.end())
        return it->second;
    return nullptr;
}

void CudaKeyFrameStorage::printStorageKeyframes() {
    cout << "[";
    for (const auto& pair : mnId_to_kf)
        std::cout << pair.first << ", ";
    cout << "]\n";
}

void CudaKeyFrameStorage::addFeatureVector(long unsigned int KF_mnId, DBoW2::FeatureVector featVec) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif

    auto it = mnId_to_kf.find(KF_mnId);
    if (it == mnId_to_kf.end()) {
        cout << "[ERROR] CudaKeyFrameStorage::addFeatureVector: KF not found!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    MAPPING_DATA_WRAPPER::CudaKeyFrame* ptr = it->second;
    ptr->addFeatureVector(featVec);

    int device_id;
    cudaGetDevice(&device_id);
    cudaMemPrefetchAsync(ptr, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), device_id);

    DEBUG_PRINT("addFeatureVector: " << KF_mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().addFeatureVector_time.push_back(time);
#endif
}

void CudaKeyFrameStorage::prefetchToDevice() {
    int device_id;
    cudaGetDevice(&device_id);
    allocator.prefetchToDevice(device_id);
}

void CudaKeyFrameStorage::shutdown() {
    if (!memory_is_initialized || memory_is_free)
        return;
    allocator.shutdown();
    memory_is_free = true;
}
