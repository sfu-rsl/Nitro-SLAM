#include "Kernels/CudaKeyFrameStorage.h"
#include "Kernels/MappingKernelController.h"
#include "Stats/LocalMappingStats.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyFrameStorage::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

MAPPING_DATA_WRAPPER::CudaKeyFrame *CudaKeyFrameStorage::d_keyframes, *CudaKeyFrameStorage::h_keyframes;
std::unordered_map<long unsigned int, ckd_buffer_index_t> CudaKeyFrameStorage::mnId_to_idx;
int CudaKeyFrameStorage::num_keyframes = 0;
bool CudaKeyFrameStorage::memory_is_initialized = false;
bool CudaKeyFrameStorage::memory_is_free = false;
ckd_buffer_index_t CudaKeyFrameStorage::first_free_idx = 0;
std::mutex CudaKeyFrameStorage::mtx;
std::queue<ckd_buffer_index_t> CudaKeyFrameStorage::free_idx;


void CudaKeyFrameStorage::initializeMemory(){   
    // std::unique_lock<std::mutex> lock(mtx);
    if (memory_is_initialized) return;
    checkCudaError(cudaMallocHost((void**)&h_keyframes, CUDA_KEYFRAME_STORAGE_SIZE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameStorage::] Failed to allocate memory for h_keyframes");  
    for (int i = 0; i < CUDA_KEYFRAME_STORAGE_SIZE; ++i) {
        h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyFrame();
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, CUDA_KEYFRAME_STORAGE_SIZE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameStorage::] Failed to allocate memory for d_keyframes");
    memory_is_initialized = true;
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrameStorage::addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif  
    // std::unique_lock<std::mutex> lock(mtx);
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaKeyFrameStorage::addCudaKeyFrame: ] memory not initialized!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }
    if (num_keyframes >= CUDA_KEYFRAME_STORAGE_SIZE) {
        cout << "[ERROR] CudaKeyFrameStorage::addCudaKeyFrame: ] number of keyframes: " << num_keyframes << " is greater than CUDA_KEYFRAME_STORAGE_SIZE: " << CUDA_KEYFRAME_STORAGE_SIZE << "\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    auto it = mnId_to_idx.find(KF->mnId);
    if (it != mnId_to_idx.end()) {
        cout << "CudaKeyFrameStorage::addCudaKeyFrame: ] KF " << KF->mnId << " is already on GPU.\n";
        return &d_keyframes[it->second];        
    }

    // Can we reuse old space?
    ckd_buffer_index_t new_kf_idx = first_free_idx;
    if (!free_idx.empty()) {
        new_kf_idx = free_idx.front();
        free_idx.pop();
    }
    else {
        first_free_idx++;
    }

    h_keyframes[new_kf_idx].setGPUAddress(&d_keyframes[new_kf_idx]);
    h_keyframes[new_kf_idx].setMemory(KF);
    checkCudaError(cudaMemcpy(&d_keyframes[new_kf_idx], &h_keyframes[new_kf_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameStorage::] Failed to copy individual element to d_keyframes");

    mnId_to_idx.emplace(KF->mnId, new_kf_idx);
    num_keyframes += 1;

    DEBUG_PRINT("addCudaKeyFrame: " << KF->mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().addCudaKeyFrame_time.push_back(time);
#endif

    return &d_keyframes[new_kf_idx];
}

void CudaKeyFrameStorage::eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif  
    // std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "CudaKeyFrameStorage::eraseCudaKeyFrame: ] KF " << KF->mnId << " not in GPU storage!\n";
        return;        
    }
    ckd_buffer_index_t idx = it->second;

    h_keyframes[idx].setAsEmpty();
    mnId_to_idx.erase(KF->mnId);
    free_idx.push(idx);
    num_keyframes--;

    DEBUG_PRINT("eraseCudaKeyFrame: " << KF->mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().eraseCudaKeyFrame_time.push_back(time);
#endif
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrameStorage::getCudaKeyFrame(long unsigned int mnId){
    // std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(mnId);
    if (it != mnId_to_idx.end()) {
        return &d_keyframes[it->second];
    }
    return nullptr;
}

void CudaKeyFrameStorage::printStorageKeyframes() {
    cout << "[";
    for (const auto& pair : mnId_to_idx) {
        std::cout << pair.first << ", ";
    }
    cout << "]\n";
}

void CudaKeyFrameStorage::addFeatureVector(long unsigned int KF_mnId, DBoW2::FeatureVector featVec) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif  
    // std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF_mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyFrameStorage::addFeatureVector: ] KF not found!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);        
    }
    int KF_idx = it->second;
    
    h_keyframes[KF_idx].addFeatureVector(featVec);
    checkCudaError(cudaMemcpy(&d_keyframes[KF_idx], &h_keyframes[KF_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameStorage::addFeatureVector: ] Failed to add feature vector");

    DEBUG_PRINT("addFeatureVector: " << KF_mnId << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().addFeatureVector_time.push_back(time);
#endif
}

void CudaKeyFrameStorage::shutdown() {
    if (!memory_is_initialized || memory_is_free) 
        return;

    for (int i = 0; i < CUDA_KEYFRAME_STORAGE_SIZE; ++i) {
        h_keyframes[i].freeMemory();
    }
    
    cudaFree(d_keyframes);
    cudaFreeHost(h_keyframes);
    memory_is_free = true;
}
