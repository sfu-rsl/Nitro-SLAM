#include "Kernels/LoopClosingKernelController.h"

// #define DEBUG


std::unique_ptr<SearchByProjectionKernel> LoopClosingKernelController::mpSearchByProjectionKernel = std::make_unique<SearchByProjectionKernel>();
// std::unique_ptr<SearchByBoWKernel> LoopClosingKernelController::mpSearchByBoWKernel = std::make_unique<SearchByBoWKernel>();
std::unique_ptr<SearchAndFuseKernel> LoopClosingKernelController::mpSearchAndFuseKernel = std::make_unique<SearchAndFuseKernel>();
bool LoopClosingKernelController::is_active = false;
bool LoopClosingKernelController::mergedSearchByProjectionOnGPU = false;
bool LoopClosingKernelController::merged3SearchByProjectionOnGPU = false;
bool LoopClosingKernelController::searchAndFuseOnGPU = false;
bool LoopClosingKernelController::singleSearchByProjectionOnGPU = false;
bool LoopClosingKernelController::graphOptimizationOnGPU = false;
bool LoopClosingKernelController::memory_is_initialized = false;
MAPPING_DATA_WRAPPER::CudaKeyFrame* LoopClosingKernelController::cudaKeyFramePtr;
std::mutex LoopClosingKernelController::shutDownMutex;
bool LoopClosingKernelController::localMappingFinished = false;
bool LoopClosingKernelController::loopClosingFinished = false;
bool LoopClosingKernelController::isShuttingDown = false;

__global__ void warmupKernel() {}


void LoopClosingKernelController::activate()
{
    is_active = true;
}


void LoopClosingKernelController::setGPURunMode(bool _mergedSearchByProjectionEnabled, bool _merged3SearchByProjectionEnabled, bool _searchAndFuseEnabled, bool _singleSearchByProjectionEnabled, bool _graphOptimizationEnabled)
{
    mergedSearchByProjectionOnGPU = _mergedSearchByProjectionEnabled;
    merged3SearchByProjectionOnGPU = _merged3SearchByProjectionEnabled;
    searchAndFuseOnGPU = _searchAndFuseEnabled;
    singleSearchByProjectionOnGPU = _singleSearchByProjectionEnabled;
    graphOptimizationOnGPU = _graphOptimizationEnabled;
}


void LoopClosingKernelController::initializeKernels(){
    cout << "Initializing Kernels...\n";
    
    CudaKeyFrameStorage::initializeMemory();

    cudaKeyFramePtr = new MAPPING_DATA_WRAPPER::CudaKeyFrame();

    if (mergedSearchByProjectionOnGPU || singleSearchByProjectionOnGPU || merged3SearchByProjectionOnGPU || graphOptimizationOnGPU)
        mpSearchByProjectionKernel->initialize();
    
    if (searchAndFuseOnGPU)
        mpSearchAndFuseKernel->initialize();

    checkCudaError(cudaDeviceSynchronize(), "[LoopClosing Kernel Controller:] Failed to initialize kernels.");
    memory_is_initialized = true;
}


void LoopClosingKernelController::shutdownKernels(bool _localMappingFinished, bool _loopClosingFinished)
{   
    unique_lock<mutex> lock(shutDownMutex);

    localMappingFinished = _localMappingFinished ? true : localMappingFinished;
    loopClosingFinished = _localMappingFinished ? true : loopClosingFinished;
    
    if (!localMappingFinished || !loopClosingFinished || isShuttingDown)
        return;

    isShuttingDown = true;

    cout << "Shutting kernels down...\n";

    if (memory_is_initialized) {
        CudaKeyFrameStorage::shutdown();
        cudaKeyFramePtr->freeMemory();
        delete cudaKeyFramePtr;
        if (mergedSearchByProjectionOnGPU || singleSearchByProjectionOnGPU || merged3SearchByProjectionOnGPU || graphOptimizationOnGPU)
            mpSearchByProjectionKernel->shutdown();
        if (searchAndFuseOnGPU)
            mpSearchAndFuseKernel->shutdown();
        // mpSearchByBoWKernel->shutdown();
    }
    CudaUtils::shutdown();
    cudaDeviceSynchronize();

}

void LoopClosingKernelController::saveKernelsStats(const std::string &file_path) {
    mpSearchByProjectionKernel->saveStats(file_path);
    mpSearchAndFuseKernel->saveStats(file_path);
}

int LoopClosingKernelController::launchSearchAndFuseKernel(vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, const float th,
                                            vector<ORB_SLAM3::MapPoint*> vpMapPoints, vector<ORB_SLAM3::MapPoint*> &vpReplacePoints)
{
    int nFused = 0;
    nFused = mpSearchAndFuseKernel->launch(connectedKFs, connectedScws, th,
                        vpMapPoints, vpReplacePoints);
    
    return nFused;
}


void LoopClosingKernelController::launchSearchByProjectionKernel(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, Sophus::Sim3<float> &Scw1,
                                const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                                std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                                int &numProjMatches, int &numProjOptMatches)
{
    mpSearchByProjectionKernel->mergedlaunch(pKF, vpPoints, Scw1,
                        vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming,
                        vpMatched1, th1, ratioHamming1,
                        numProjMatches, numProjOptMatches);
    
    return;
}

void LoopClosingKernelController::launch3SearchByProjectionKernel(vector<ORB_SLAM3::KeyFrame*> currentCovKFs, vector<Sophus::Sim3f> currentCovmScws, const std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,
                                    int th, float ratioHamming, int* num_matches, int covKFsSize)
{
    mpSearchByProjectionKernel->mergedlaunch(currentCovKFs, currentCovmScws, vpMapPoints,
                                    th, ratioHamming, num_matches, covKFsSize);
    
    return;
}

int LoopClosingKernelController::launchSingleSearchByProjectionKernel2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw,
                                const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming)
{
    return mpSearchByProjectionKernel->launch(pKF, Scw, vpPoints, vpMatched, th, ratioHamming);

}

int LoopClosingKernelController::launchSingleSearchByProjectionKernel(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw,
                                const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
    return mpSearchByProjectionKernel->launch(pKF, Scw, vpPoints, vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming);

}
    

int LoopClosingKernelController::launchSearchByBoWKernel(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12)
{
    
    // return mpSearchByBoWKernel->launch(pKF1, pKF2, vpMatches12);
}


void LoopClosingKernelController::launchWarmUp()
{
    // cudaFree(0);
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    std::cout << "[CUDA Warm-up] Completed GPU context initialization." << std::endl;
    return;
}