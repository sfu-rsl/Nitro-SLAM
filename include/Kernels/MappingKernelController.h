#ifndef MAPPING_KERNEL_CONTROLLER_H
#define MAPPING_KERNEL_CONTROLLER_H

#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameAllocator.h"
#include "CudaUtils.h"
#include "FuseKernel.h"
#include "SearchForTriangulationKernel.h"
#include "TriangulationMatchKernel.h"
#include "MapPointCandidateKernel.h"
#include "Optimizer.h"
#include <memory> 

using namespace std;

class MappingKernelController{
public:
    static void setCUDADevice(int deviceID);
    
    static bool is_active;
    
    static void activate();

    static bool searchForTriangulationOnGPU;
    static bool fuseOnGPU;
    static bool optimizeKeyframeCulling;
    // Selects the rewritten batched search (TriangulationMatchKernel) over the
    // original SearchForTriangulationKernel, which is preserved unchanged.
    static bool useNewTriangulation;
    // Selects CreateNewMapPointsGPU2, which also runs the geometry on the GPU.
    static bool useGPU2Pipeline;
    static bool LBAOnGPU;

    static void setGPURunMode(bool searchForTriangulation, bool fuse, bool keyframeCulling, bool LBA);

    static void initializeKernels();
    
    static void shutdownKernels(bool _localMappingFinished, bool _loopClosingFinished);
    
    static void saveKernelsStats(const std::string &file_path);


    static void launchMapPointCandidateKernel(
        ORB_SLAM3::KeyFrame* pKF1, const std::vector<ORB_SLAM3::KeyFrame*> &kept,
        const std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices,
        bool bInertial, bool bFarPoints, float thFarPoints, float ratioFactor,
        std::vector<MapPointCandidateKernel::Candidate> &outCandidates
    );

    static void launchTriangulationMatchKernel(
        ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs,
        bool mbMonocular, bool bCoarse,
        std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, std::vector<size_t> &vpNeighKFsIndexes
    );

    static void launchSearchForTriangulationKernel(
        ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
        bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2, 
        std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, std::vector<size_t> &vpNeighKFsIndexes
    );
    
    static void launchFuseKernel(
        ORB_SLAM3::KeyFrame *neighKF, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints, const float th, 
        const bool bRight, ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow, 
        std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
    );

    static void launchFuseKernelV2(
        std::vector<ORB_SLAM3::KeyFrame*> neighKFs, ORB_SLAM3::KeyFrame *currKF, const float th,  
        std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
    );

private:
    static bool memory_is_initialized, isShuttingDown, localMappingFinished, loopClosingFinished;
    static MAPPING_DATA_WRAPPER::CudaKeyFrame *cudaKeyFramePtr;
    static std::unique_ptr<SearchForTriangulationKernel> mpSearchForTriangulationKernel;
    static std::unique_ptr<TriangulationMatchKernel> mpTriangulationMatchKernel;
    static std::unique_ptr<MapPointCandidateKernel> mpMapPointCandidateKernel;
    static std::unique_ptr<FuseKernel> mpFuseKernel;
    static std::mutex shutDownMutex;
};

#endif
