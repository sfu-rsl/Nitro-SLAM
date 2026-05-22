#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "../KeyFrame.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaWrappers/CudaMapPoint.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

// Flag to activate the measurement of time in each kernel.
#define REGISTER_TRACKING_STATS
#define REGISTER_LOCAL_MAPPING_STATS
#define REGISTER_LOOP_CLOSING_STATS

// #define DEBUG
#define N_FEATURES_TH 20
#define DESCRIPTOR_SIZE 32

class CudaUtils {
    public:
        static void loadSetting(int _nFeatures, int _nLevels, bool _isMonocular, float _scaleFactor, int _nCols, int _nRows, bool _cameraIsFisheye);
        static void shutdown();

    public:
        static int nFeatures_with_th;
        static int nLevels; 
        static bool isMonocular; 
        static float scaleFactor; 
        static int nCols;
        static int nRows;
        static int keypointsPerCell;
        static int maxNumOfMapPoints;
        static int ORBmatcher_TH_HIGH;
        static int ORBmatcher_TH_LOW;
        static int ORBmatcher_HISTO_LENGTH;
        static float* d_mvScaleFactors;
        static bool cameraIsFisheye;
};

void checkCudaError(cudaError_t err, const char* msg);
__device__ int DescriptorDistance(const uint8_t *a, const uint8_t *b);

void printKeyframeCPU(ORB_SLAM3::KeyFrame* KF);
__global__ void printKFSingleGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame* KF);
__global__ void printKFSingleGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame** d_keyframes, int idx);
__global__ void printKFListGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame** d_keyframes, int size);
__device__ void printKeyframeGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame* KF);

void printMPCPU(ORB_SLAM3::MapPoint* mp);
__device__ void printMPGPU(MAPPING_DATA_WRAPPER::CudaMapPoint* mp);
__global__ void printMPSingleGPU(MAPPING_DATA_WRAPPER::CudaMapPoint* mp);
__global__ void printMPListGPU(MAPPING_DATA_WRAPPER::CudaMapPoint* d_mapPoints, int idx);

#endif // CUDA_UTILS_H