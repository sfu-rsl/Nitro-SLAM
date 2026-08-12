#ifndef LOOP_CLOSING_KERNEL_CONTROLLER_H
#define LOOP_CLOSING_KERNEL_CONTROLLER_H

#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameAllocator.h"
#include "SearchAndFuseKernel.h"
#include "SearchByProjectionKernel.h"
// #include "SearchByBoWKernel.h"
#include <memory> 

using namespace std;

class LoopClosingKernelController{
public:
    static bool is_active;
    
    static void activate();

    static bool mergedSearchByProjectionOnGPU;
    static bool merged3SearchByProjectionOnGPU;
    static bool searchAndFuseOnGPU;
    static bool singleSearchByProjectionOnGPU;
    static bool graphOptimizationOnGPU;
    static bool globalBAOnGPU;

    static void setGPURunMode(bool mergedSearchByProjectionEnabled, bool merged3SearchByProjectionEnabled, bool searchAndFuseEnabled, bool singleSearchByProjectionEnabled, bool graphOptimizationEnabled, bool globalBAEnabled);

    static void initializeKernels();

    static void shutdownKernels(bool _localMappingFinished, bool _loopClosingFinished);

    static void launchSearchByProjectionKernel(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, Sophus::Sim3<float> &Scw1,
                                    const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                                    int &numProjMatches, int &numProjOptMatches);
    
    static void launch3SearchByProjectionKernel(vector<ORB_SLAM3::KeyFrame*> currentCovKFs, vector<Sophus::Sim3f> currentCovmScws, const std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,
                                    int th, float ratioHamming, int* num_matches, int covKFsSize);
    
    static int launchSingleSearchByProjectionKernel2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw,
                                const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming);
    
    static int launchSingleSearchByProjectionKernel(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw,
                                const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming);
    
    static int launchSearchByBoWKernel(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12);
    
    static int launchSearchAndFuseKernel(vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, const float th,
                                        vector<ORB_SLAM3::MapPoint*> vpMapPoints, vector<ORB_SLAM3::MapPoint*> &vpReplacePoints);
    static void launchWarmUp();

    static void saveKernelsStats(const std::string &file_path);

private:
    static bool memory_is_initialized, isShuttingDown, localMappingFinished, loopClosingFinished;
    static std::unique_ptr<SearchByProjectionKernel> mpSearchByProjectionKernel;
    // static std::unique_ptr<SearchByBoWKernel> mpSearchByBoWKernel;
    static std::unique_ptr<SearchAndFuseKernel> mpSearchAndFuseKernel;
    static MAPPING_DATA_WRAPPER::CudaKeyFrame *cudaKeyFramePtr;
    static std::mutex shutDownMutex;
};

#endif
