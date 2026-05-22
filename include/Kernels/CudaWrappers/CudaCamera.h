#ifndef CUDA_CAMERA_H
#define CUDA_CAMERA_H

#include <Eigen/Core>

namespace MAPPING_DATA_WRAPPER {
    struct CudaCamera {
        bool isAvailable;
        float mvParameters[8];
        Eigen::Matrix3f toK;
    };
}

#endif