#ifndef CUDA_KEYPOINT_H
#define CUDA_KEYPOINT_H

#include <opencv2/opencv.hpp>

namespace TRACKING_DATA_WRAPPER {

    struct CudaKeyPoint {
        float ptx;
        float pty;
        int octave;
    };
}


namespace MAPPING_DATA_WRAPPER {

    struct CudaKeyPoint {
        float ptx;
        float pty;
        int octave;
    };
}

#endif // CUDA_KEYPOINT_H