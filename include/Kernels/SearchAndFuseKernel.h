#ifndef SEARCH_AND_FUSE_KERNEL_H
#define SEARCH_AND_FUSE_KERNEL_H

#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameStorage.h"
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

#define MAX_CONNECTED_KF_COUNT 40

class SearchAndFuseKernel : public KernelInterface {
public:
    SearchAndFuseKernel() { memory_is_initialized = false; };
    void initialize() override;
    void shutdown() override;
    void saveStats(const std::string &file_path) override;
    void launch() override { std::cout << "[SearchAndFuseKernel:] provide input for kernel launch.\n"; };
    int launch(std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, float th,
               std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints, vector<ORB_SLAM3::MapPoint*> &vpReplacePoints);
    void origFuse(ORB_SLAM3::KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<ORB_SLAM3::MapPoint*> &vpPoints, const float th);
    int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);

private:
    bool memory_is_initialized;
    int *d_bestDists, *d_bestIdxs;
    int *bestDists, *bestIdxs;
    MAPPING_DATA_WRAPPER::CudaMapPoint *h_MapPoints, *d_MapPoints;
    MAPPING_DATA_WRAPPER::CudaKeyFrame **h_KeyFrames, **d_KeyFrames;
    Eigen::Vector3f *h_Ow, *d_Ow;
    Sophus::SE3f *h_Tcw, *d_Tcw;

    std::vector<double> input_data_wrap_time, input_data_transfer_time, kernel_exec_time, output_data_transfer_time, post_processing_time, total_exec_time;
};

#endif 