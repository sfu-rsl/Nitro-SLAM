#include "Kernels/MappingKernelController.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [Mapping KernelController::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

bool MappingKernelController::is_active = false;
bool MappingKernelController::searchForTriangulationOnGPU = false;
bool MappingKernelController::fuseOnGPU = false;
bool MappingKernelController::optimizeKeyframeCulling = false;
bool MappingKernelController::LBAOnGPU = false;
bool MappingKernelController::memory_is_initialized = false;
bool MappingKernelController::isShuttingDown = false;
bool MappingKernelController::localMappingFinished = false;
bool MappingKernelController::loopClosingFinished = false;
std::unique_ptr<SearchForTriangulationKernel> MappingKernelController::mpSearchForTriangulationKernel = std::make_unique<SearchForTriangulationKernel>();
std::unique_ptr<TriangulationMatchKernel> MappingKernelController::mpTriangulationMatchKernel = std::make_unique<TriangulationMatchKernel>();
bool MappingKernelController::useNewTriangulation = false;
bool MappingKernelController::useGPU2Pipeline = false;
std::unique_ptr<MapPointCandidateKernel> MappingKernelController::mpMapPointCandidateKernel = std::make_unique<MapPointCandidateKernel>();
std::unique_ptr<FuseKernel> MappingKernelController::mpFuseKernel = std::make_unique<FuseKernel>();
MAPPING_DATA_WRAPPER::CudaKeyFrame* MappingKernelController::cudaKeyFramePtr;
std::mutex MappingKernelController::shutDownMutex;

void MappingKernelController::setCUDADevice(int deviceID) {
    cudaSetDevice(deviceID);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    printf("Using device %d: %s\n", deviceID, deviceProp.name);
}

void MappingKernelController::activate() {
    is_active = true;
}

void MappingKernelController::setGPURunMode(bool _searchForTriangulation, bool _fuseStatus, bool _keyframeCulling, bool _LBA) {
    searchForTriangulationOnGPU = _searchForTriangulation;
    fuseOnGPU = _fuseStatus;
    optimizeKeyframeCulling = _keyframeCulling;
    LBAOnGPU = _LBA;
}

void MappingKernelController::initializeKernels(){

    DEBUG_PRINT("Initializing Kernels");
    
    CudaKeyFrameStorage::initializeMemory();

    cudaKeyFramePtr = new MAPPING_DATA_WRAPPER::CudaKeyFrame();

    if (searchForTriangulationOnGPU) {
        if (useNewTriangulation || useGPU2Pipeline)
            mpTriangulationMatchKernel->initialize();
        else
            mpSearchForTriangulationKernel->initialize();
        if (useGPU2Pipeline)
            mpMapPointCandidateKernel->initialize();
    }
    
    if (fuseOnGPU)
        mpFuseKernel->initialize();

    /*
    if (LBAOnGPU)
        ORB_SLAM3::initialize_compute_engine();
    */

    checkCudaError(cudaStreamSynchronize(cudaStreamPerThread), "[Mapping Kernel Controller:] Failed to initialize kernels.");
    memory_is_initialized = true;
}

void MappingKernelController::shutdownKernels(bool _localMappingFinished, bool _loopClosingFinished) {
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
        if (searchForTriangulationOnGPU == 1) {
            if (useNewTriangulation || useGPU2Pipeline)
                mpTriangulationMatchKernel->shutdown();
            else
                mpSearchForTriangulationKernel->shutdown();
            if (useGPU2Pipeline)
                mpMapPointCandidateKernel->shutdown();
        }
        if (fuseOnGPU == 1)
            mpFuseKernel->shutdown();
        //if (LBAOnGPU == 1)
        //    ORB_SLAM3::destroy_compute_engine();
    }

    CudaUtils::shutdown();
}

void MappingKernelController::saveKernelsStats(const std::string &file_path){

    DEBUG_PRINT("Saving Kernels Stats");
    
    mpSearchForTriangulationKernel->saveStats(file_path);
    mpFuseKernel->saveStats(file_path);
}

void MappingKernelController::launchMapPointCandidateKernel(
    ORB_SLAM3::KeyFrame* pKF1, const std::vector<ORB_SLAM3::KeyFrame*> &kept,
    const std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices,
    bool bInertial, bool bFarPoints, float thFarPoints, float ratioFactor,
    std::vector<MapPointCandidateKernel::Candidate> &outCandidates)
{
    mpMapPointCandidateKernel->launch(pKF1, kept, allvMatchedIndices, bInertial,
                                      bFarPoints, thFarPoints, ratioFactor, outCandidates);
}

void MappingKernelController::launchTriangulationMatchKernel(
    ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs,
    bool mbMonocular, bool bCoarse,
    std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, std::vector<size_t> &vpNeighKFsIndexes)
{
    mpTriangulationMatchKernel->launch(mpCurrentKeyFrame, vpNeighKFs, mbMonocular, bCoarse,
                                       allvMatchedIndices, vpNeighKFsIndexes);
}

void MappingKernelController::launchSearchForTriangulationKernel(
    ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
    bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2, 
    std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, 
    std::vector<size_t> &vpNeighKFsIndexes
) {
    mpSearchForTriangulationKernel->launch(
        mpCurrentKeyFrame, vpNeighKFs, mbMonocular, mbInertial, recentlyLost, mbIMU_BA2, 
        allvMatchedIndices, vpNeighKFsIndexes
    );
}

void MappingKernelController::launchFuseKernel(ORB_SLAM3::KeyFrame *neighKF, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints, const float th, 
                                               const bool bRight, ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow, 
                                               vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs) {

    DEBUG_PRINT("Launching Fuse Kernel");
    
    mpFuseKernel->launch(neighKF, vpMapPoints, th, bRight, pCamera, Tcw, Ow, validMapPoints, bestDists, bestIdxs);
}

void MappingKernelController::launchFuseKernelV2(
    std::vector<ORB_SLAM3::KeyFrame*> neighKFs, ORB_SLAM3::KeyFrame *currKF, const float th,  
    std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
) {

    DEBUG_PRINT("Launching Fuse Kernel V2");

    mpFuseKernel->launchV2(neighKFs, currKF, th, validMapPoints, bestDists, bestIdxs);
}