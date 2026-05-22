#ifndef FUSE_KERNEL_H
#define FUSE_KERNEL_H

#include "KernelInterface.h"
#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameStorage.h"
#include "CudaUtils.h"
#include "CameraModels/GeometricCamera.h"
#include <Eigen/Core>

#define MAX_NEIGHBOR_KF_COUNT 100

class FuseKernel: public KernelInterface {

    public:
        FuseKernel() { memory_is_initialized = false; 
                       frameCounter = 0; };
        void initialize() override;
        void shutdown() override;
        void launch() override { std::cout << "[FuseKernel:] provide input for kernel launch.\n"; };
        void launch(ORB_SLAM3::KeyFrame *neighKF, const vector<ORB_SLAM3::MapPoint*> &currKFMapPoints, const float th, 
                    const bool bRight, ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow, 
                    vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs);
        void launchV2(std::vector<ORB_SLAM3::KeyFrame*> neighKFs, ORB_SLAM3::KeyFrame *currKF, float th, 
                      vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs);
        void origFuse(ORB_SLAM3::KeyFrame *pKF, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints, const float th, const bool bRight);
        int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        void saveStats(const string &file_path) override;

    private:
        bool memory_is_initialized;
        int *d_bestDists, *d_bestIdxs;
        MAPPING_DATA_WRAPPER::CudaKeyFrame **d_neighKFs;
        MAPPING_DATA_WRAPPER::CudaMapPoint *d_currKFMapPoints;
        Sophus::SE3f *d_Tcw, *d_TcwRight;
        Eigen::Vector3f *d_Ow, *d_OwRight;

        std::vector<std::pair<long unsigned int, double>> input_data_wrap_time;
        std::vector<std::pair<long unsigned int, double>> input_data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> kernel_exec_time;
        std::vector<std::pair<long unsigned int, double>> output_data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> total_exec_time;
        long unsigned int frameCounter;
};

#endif 