#ifndef SEARCH_BY_PROJECTION_KERNEL_H
#define SEARCH_BY_PROJECTION_KERNEL_H

#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameAllocator.h"
#include "KernelInterface.h"
#include "CudaUtils.h"
#include "CameraModels/GeometricCamera.h"
#include <Eigen/Core>
#include <csignal> 
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

class SearchByProjectionKernel : public KernelInterface {
public:
    SearchByProjectionKernel() { memory_is_initialized = false; };
    void initialize() override;
    void shutdown() override;
    void saveStats(const std::string &file_path) override;
    void launch() override { std::cout << "[SearchByProjectionKernel:] provide input for kernel launch.\n"; };
    int launch(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming);
    int launch(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming);
    void mergedlaunch(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, Sophus::Sim3<float> &Scw1,
                    const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                    int &numProjMatches, int &numProjOptMatches);
    void mergedlaunch(vector<ORB_SLAM3::KeyFrame*> currentCovKFs, vector<Sophus::Sim3f> currentCovmScws, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                    int th, float ratioHamming, int* num_matches, int covKFsSize);
    void origSearchByProjection(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming);
    void origSearchByProjection2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming);
    int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);

private:
    // Grow the map-point buffers to fit this launch. They were fixed at 4100 while
    // every launch path filled them with vpPoints.size() entries and only warned on
    // overflow -- the same defect that fired in SearchAndFuseKernel.
    void ensureMapPointCapacity(size_t numMapPoints);
    void freeMapPointBuffers();

    size_t mapPointCapacity = 0;

    bool memory_is_initialized;
    int *d_bestDists, *d_bestIdxs;
    int *bestDists, *bestIdxs;
    MAPPING_DATA_WRAPPER::CudaMapPoint *h_MapPoints, *d_MapPoints;
    MAPPING_DATA_WRAPPER::CudaKeyFrame *h_KeyFrame, *d_KeyFrame;
    MAPPING_DATA_WRAPPER::CudaKeyFrame **h_KeyFrames, **d_KeyFrames;
    Eigen::Vector3f *h_Ow, *d_Ow;
    Sophus::SE3f *h_Tcw, *d_Tcw;

    std::vector<double> launch_1_time, launch_2_time, merged_launch_1_time, merged_launch_2_time;
};

#endif