#include "Kernels/CudaUtils.h"
#include "Kernels/TrackingKernelController.h"
#include "Kernels/MappingKernelController.h"
#include "Kernels/LoopClosingKernelController.h"

int CudaUtils::nFeatures_with_th;
int CudaUtils::nLevels; 
bool CudaUtils::isMonocular;  
float CudaUtils::scaleFactor;  
int CudaUtils::nCols;
int CudaUtils::nRows;
int CudaUtils::keypointsPerCell = 20;
int CudaUtils::maxNumOfMapPoints = 16000;
float* CudaUtils::d_mvScaleFactors;
int CudaUtils::ORBmatcher_TH_HIGH = 100;
int CudaUtils::ORBmatcher_TH_LOW = 50;
int CudaUtils::ORBmatcher_HISTO_LENGTH = 30;
bool CudaUtils::cameraIsFisheye;

void printKeyframeCPU(ORB_SLAM3::KeyFrame* KF) {
    printf("[*CPU*] KF mnId: %d\n", KF->mnId);
    vector<ORB_SLAM3::MapPoint*> h_mapPoints = KF->GetMapPointMatches();
    int mvpMapPoints_size = h_mapPoints.size();
    for (int i = 0; i < mvpMapPoints_size; ++i) {
        if(h_mapPoints[i]) {
            printf("    i:%d, mp mnId: %d\n", i, h_mapPoints[i]->mnId);
            std::map<ORB_SLAM3::KeyFrame*,std::tuple<int,int>> observations = h_mapPoints[i]->GetObservations();
            for(map<ORB_SLAM3::KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                ORB_SLAM3::KeyFrame* pKFi = mit->first;
                printf("        pKFi mnId: %d\n", pKFi->mnId);
            }           
        }
    }
}

__device__ void printKeyframeGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame* KF) {
    printf("KF mnId: %lu\n", KF->mnId);
}

__global__ void printKFSingleGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame* KF) {
    printKeyframeGPU(KF);
}

__global__ void printKFSingleGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame** d_keyframes, int idx) {
    MAPPING_DATA_WRAPPER::CudaKeyFrame* KF = d_keyframes[idx];
    printKeyframeGPU(KF);
}

__global__ void printKFListGPU(MAPPING_DATA_WRAPPER::CudaKeyFrame** d_keyframes, int size) {
    for (int i = 0; i < size; i++) {
        MAPPING_DATA_WRAPPER::CudaKeyFrame* KF = d_keyframes[i];
        if (KF == nullptr) {
            printf("d_keyframes[%d]: \nnullptr\n", i);
            continue;
        }
        printf("d_keyframes[%d]: \n", i);
        printKeyframeGPU(KF);
    }
}

void printMPCPU(ORB_SLAM3::MapPoint* mp) {
    printf("[*CPU*] mp mnId: %lu\n", mp->mnId);
    cv::Mat mDescriptor = mp->GetDescriptor();
    cout << mDescriptor.rows << ", " << mDescriptor.cols << std::endl;
    for (int i = 0; i < mDescriptor.rows; i++) {
        for (int j = 0; j < mDescriptor.cols; j++) {
            printf("%u,", mDescriptor.at<uint8_t>(i, j));
        }
        printf("\n");
    }  
}

__device__ void printMPGPU(MAPPING_DATA_WRAPPER::CudaMapPoint* mp) {
    printf("[=GPU=] mp mnId: %lu\n", mp->mnId);
    printf("mDescriptor: ");
    for (int i = 0; i < 32; i++) {
        printf("%u,", mp->mDescriptor[i]);
    }
    printf("\n");
}

__global__ void printMPSingleGPU(MAPPING_DATA_WRAPPER::CudaMapPoint* mp) {
    printMPGPU(mp);
}

__global__ void printMPListGPU(MAPPING_DATA_WRAPPER::CudaMapPoint* d_mapPoints, int idx) {
    MAPPING_DATA_WRAPPER::CudaMapPoint mp = d_mapPoints[idx];
    printMPGPU(&mp);
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << ", status code: " << err << std::endl;
        
        if (TrackingKernelController::is_active)
            TrackingKernelController::shutdownKernels();
        if (MappingKernelController::is_active)
            MappingKernelController::shutdownKernels(true, true);
        if (LoopClosingKernelController::is_active)
            LoopClosingKernelController::shutdownKernels(true, true);
        // CudaUtils::shutdown();

        exit(EXIT_FAILURE);
    }
}

cudaError_t allocateSharedMemory(void** ptr, size_t size) {
#ifdef DEVICE_JETSON
    return cudaHostAlloc(ptr, size, cudaHostAllocMapped);
#else
    return cudaMallocManaged(ptr, size);
#endif
}

cudaError_t freeSharedMemory(void* ptr) {
#ifdef DEVICE_JETSON
    return cudaFreeHost(ptr);
#else
    return cudaFree(ptr);
#endif
}

void CudaUtils::loadSetting(int _nFeatures, int _nLevels, bool _isMonocular, float _scaleFactor, int _nCols, int _nRows, bool _cameraIsFisheye){
    nFeatures_with_th = _nFeatures + N_FEATURES_TH;
    nLevels = _nLevels;
    isMonocular = _isMonocular;
    scaleFactor = _scaleFactor;
    nCols = _nCols;
    nRows = _nRows;
    cameraIsFisheye = _cameraIsFisheye;
    std::vector<float> h_mvScaleFactors(nLevels);
    h_mvScaleFactors[0]=1.0f;
    for(int i=1; i<nLevels; i++)
    {
        h_mvScaleFactors[i]=h_mvScaleFactors[i-1]*scaleFactor;
    }
    checkCudaError(cudaMalloc(&d_mvScaleFactors, h_mvScaleFactors.size() * sizeof(float)), "CudaUtils:: Failed to allocate memory for d_mvScaleFactors");
    checkCudaError(cudaMemcpy(d_mvScaleFactors, h_mvScaleFactors.data(), h_mvScaleFactors.size() * sizeof(float), cudaMemcpyHostToDevice),"CudaUtils:: Failed to initialize d_mvScaleFactors"); 
}

void CudaUtils::shutdown(){
    if (d_mvScaleFactors != nullptr) {
        checkCudaError(cudaFree(d_mvScaleFactors), "[CudaUtils::] Failed to free frame memory: d_mvScaleFactors");
        d_mvScaleFactors = nullptr;
    }
}

__device__ int DescriptorDistance(const uint8_t *a, const uint8_t *b) {
    const int32_t *pa = reinterpret_cast<const int32_t*>(a);
    const int32_t *pb = reinterpret_cast<const int32_t*>(b);

    int dist = 0;

    for (int i = 0; i < DESCRIPTOR_SIZE / 4; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
