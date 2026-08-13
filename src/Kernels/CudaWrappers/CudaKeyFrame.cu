#include "Kernels/CudaWrappers/CudaKeyFrame.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyFrame]: " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

namespace MAPPING_DATA_WRAPPER
{
    void CudaKeyFrame::initializeMemory(){
        DEBUG_PRINT("Allocating GPU memory For CudaKeyFrame...");

        int nFeatures = CudaUtils::nFeatures_with_th;
        
        bool cameraIsFisheye = CudaUtils::cameraIsFisheye;

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mvDepth, 2 * nFeatures * sizeof(float)), "Frame::failed to allocate memory for mvDepth");
        } else {
            checkCudaError(cudaMalloc((void**)&mvDepth, nFeatures * sizeof(float)), "Frame::failed to allocate memory for mvDepth");
        }

        checkCudaError(cudaMalloc((void**)&mvScaleFactors, nFeatures * sizeof(float)), "KeyFrame::failed to allocate memory for mvScaleFactors");

        checkCudaError(cudaMalloc((void**)&mvuRight, nFeatures * sizeof(float)), "KeyFrame::failed to allocate memory for mvuRight");

        checkCudaError(cudaMalloc((void**)&mvKeys, nFeatures * sizeof(CudaKeyPoint)), "KeyFrame::failed to allocate memory for mvKeys");
        
        checkCudaError(cudaMalloc((void**)&mvKeysRight, nFeatures * sizeof(CudaKeyPoint)), "KeyFrame::failed to allocate memory for mvKeysRight");
        
        checkCudaError(cudaMalloc((void**)&mvKeysUn, nFeatures * sizeof(CudaKeyPoint)), "KeyFrame::failed to allocate memory for mvKeysUn"); 
        
        checkCudaError(cudaMalloc((void**)&mvInvLevelSigma2, nFeatures * sizeof(float)), "KeyFrame::failed to allocate memory for mvInvLevelSigma2");

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mDescriptors, 2 * nFeatures * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Frame::failed to allocate memory for mDescriptors");
        } else {
            checkCudaError(cudaMalloc((void**)&mDescriptors, nFeatures * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Frame::failed to allocate memory for mDescriptors");
        }

        checkCudaError(cudaMalloc((void**)&mFeatVec, MAX_FEAT_PER_WORD*MAX_FEAT_VEC_SIZE*sizeof(unsigned int)), "KeyFrame::failed to allocate memory for mFeatVec");
        checkCudaError(cudaMalloc((void**)&mFeatVecStartIndexes, MAX_FEAT_VEC_SIZE*sizeof(int)), "KeyFrame::failed to allocate memory for mFeatVecStartIndexes");
    }

    CudaKeyFrame::CudaKeyFrame() {
        initializeMemory();
    }

    void CudaKeyFrame::setGPUAddress(CudaKeyFrame* ptr) {
        gpuAddr = ptr;
    }

    void CudaKeyFrame::setMemory(ORB_SLAM3::KeyFrame* KF) {
        DEBUG_PRINT("Filling CudaKeyFrame Memory With KeyFrame Data...");

        mnId = KF->mnId;
        Nleft = KF->NLeft;
        mThDepth = KF->mThDepth;
        mfLogScaleFactor = KF->mfLogScaleFactor;
        mnScaleLevels = KF->mnScaleLevels;
        mnMinX = KF->mnMinX;
        mnMaxX = KF->mnMaxX;
        mnMinY = KF->mnMinY;
        mnMaxY = KF->mnMaxY;
        mfGridElementWidthInv = KF->mfGridElementWidthInv;
        mfGridElementHeightInv = KF->mfGridElementHeightInv;
        mnGridCols = KF->mnGridCols;
        mnGridRows = KF->mnGridRows;
        mbf = KF->mbf;

        fx = KF->fx;
        fy = KF->fy;
        cx = KF->cx;
        cy = KF->cy;

        checkCudaError(cudaMemcpy(mvDepth, KF->mvDepth.data(), KF->mvDepth.size() * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvDepth to gpu");
        
        mvScaleFactors_size = KF->mvScaleFactors.size();
        checkCudaError(cudaMemcpy(mvScaleFactors, KF->mvScaleFactors.data(), mvScaleFactors_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvScaleFactors to gpu");
        
        mvInvLevelSigma2_size = KF->mvInvLevelSigma2.size();
        checkCudaError(cudaMemcpy(mvInvLevelSigma2, KF->mvInvLevelSigma2.data(), mvInvLevelSigma2_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvInvLevelSigma2 to gpu");
        
        mvuRight_size = KF->mvuRight.size();
        checkCudaError(cudaMemcpy(mvuRight, KF->mvuRight.data(), mvuRight_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvuRight to gpu");
        
        mDescriptor_rows = KF->mDescriptors.rows;
        checkCudaError(cudaMemcpy((void*) mDescriptors, KF->mDescriptors.data,  KF->mDescriptors.rows * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mDescriptors to gpu"); 
        
        mvKeys_size = KF->mvKeys.size();
        std::vector<CudaKeyPoint> tmp_mvKeys(mvKeys_size);
        for (int i = 0; i < mvKeys_size; ++i){
            tmp_mvKeys[i].ptx = KF->mvKeys[i].pt.x;
            tmp_mvKeys[i].pty = KF->mvKeys[i].pt.y;
            tmp_mvKeys[i].octave = KF->mvKeys[i].octave;
        }
        checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeys to gpu");
        
        mvKeysRight_size = KF->mvKeysRight.size();
        std::vector<CudaKeyPoint> tmp_mvKeysRight(mvKeysRight_size);        
        for (int i = 0; i < mvKeysRight_size; ++i){
            tmp_mvKeysRight[i].ptx = KF->mvKeysRight[i].pt.x;
            tmp_mvKeysRight[i].pty = KF->mvKeysRight[i].pt.y;
            tmp_mvKeysRight[i].octave = KF->mvKeysRight[i].octave;
        }
        checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysRight to gpu");
        
        mvKeysUn_size = KF->mvKeysUn.size();
        std::vector<CudaKeyPoint> tmp_mvKeysUn(mvKeysUn_size);   
        for (int i = 0; i < mvKeysUn_size; ++i){
            tmp_mvKeysUn[i].ptx = KF->mvKeysUn[i].pt.x;
            tmp_mvKeysUn[i].pty = KF->mvKeysUn[i].pt.y;
            tmp_mvKeysUn[i].octave = KF->mvKeysUn[i].octave;
        }
        checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysUn to gpu");

        int keypoints_per_cell = CudaUtils::keypointsPerCell;
        for (int i = 0; i < mnGridCols; ++i) {
            for (int j = 0; j < mnGridRows; ++j) {
                size_t num_keypoints = KF->getMGrid()[i][j].size();
                if (num_keypoints > 0) {
                    std::memcpy(&flatMGrid[(i * mnGridRows + j) * keypoints_per_cell], KF->getMGrid()[i][j].data(), num_keypoints * sizeof(std::size_t));
                }
                flatMGrid_size[i * mnGridRows + j] = num_keypoints;
            }
        }

        if (!KF->mGridRight.empty()) {
            for (int i = 0; i < mnGridCols; ++i) {
                for (int j = 0; j < mnGridRows; ++j) {
                    size_t num_keypoints = KF->mGridRight[i][j].size();
                    if (num_keypoints > 0) {
                        std::memcpy(&flatMGridRight[(i * mnGridRows + j) * KEYPOINTS_PER_CELL], KF->mGridRight[i][j].data(), num_keypoints * sizeof(std::size_t));
                    }
                    flatMGridRight_size[i * mnGridRows + j] = num_keypoints;
                }
            }
        }

        copyGPUCamera(&camera1, KF->mpCamera);
        copyGPUCamera(&camera2, KF->mpCamera2);
    }

    void CudaKeyFrame::addFeatureVector(const DBoW2::FeatureVector &featVec) {
        mFeatCount = featVec.size();
        unsigned int tmp_mFeatVec[mFeatCount * MAX_FEAT_PER_WORD];
        int tmp_mFeatVecStartIndexes[mFeatCount];
        copyFeatVec(tmp_mFeatVec, tmp_mFeatVecStartIndexes, featVec);
        int mFeatVecSize = tmp_mFeatVecStartIndexes[mFeatCount-1];

        checkCudaError(cudaMemcpy(mFeatVec, tmp_mFeatVec, mFeatVecSize*sizeof(unsigned int), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mFeatVec to gpu");
        checkCudaError(cudaMemcpy(mFeatVecStartIndexes, tmp_mFeatVecStartIndexes, mFeatCount*sizeof(int), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mFeatVecStartIndexes to gpu");
    }

    void CudaKeyFrame::copyGPUCamera(CudaCamera *out, ORB_SLAM3::GeometricCamera *camera) {
        out->isAvailable = (bool) camera;
        if (!out->isAvailable)
            return;
    
        memcpy(out->mvParameters, camera->getParameters().data(), sizeof(float)*camera->getParameters().size());
        out->toK = camera->toK_();
    }

    void CudaKeyFrame::copyFeatVec(unsigned int *out, int *outIndexes, const DBoW2::FeatureVector &inp) {
        DBoW2::FeatureVector::const_iterator f1it = inp.begin();
        DBoW2::FeatureVector::const_iterator f1end = inp.end();
        int outFeatureVecSize = 0, counter = 0;

        while (f1it != f1end) {
            memcpy(out + outFeatureVecSize, f1it->second.data(), f1it->second.size()*sizeof(unsigned int));
            outFeatureVecSize += f1it->second.size();
            outIndexes[counter] = outFeatureVecSize;
            counter++;
            f1it++;
        }
    }

    void CudaKeyFrame::freeMemory(){
        DEBUG_PRINT("Freeing GPU Memory For KeyFrame...");
        checkCudaError(cudaFree((void*)mvScaleFactors),"Failed to free keyframe memory: mvScaleFactors");
        checkCudaError(cudaFree((void*)mvInvLevelSigma2),"Failed to free keyframe memory: mvInvLevelSigma2");
        checkCudaError(cudaFree((void*)mvuRight),"Failed to free keyframe memory: mvuRight");
        checkCudaError(cudaFree((void*)mDescriptors),"Failed to free keyframe memory: mDescriptors");
        checkCudaError(cudaFree((void*)mvKeys),"Failed to free keyframe memory: mvKeys");
        checkCudaError(cudaFree((void*)mvKeysRight),"Failed to free keyframe memory: mvKeysRight");
        checkCudaError(cudaFree((void*)mvKeysUn),"Failed to free keyframe memory: mvKeysUn");
        checkCudaError(cudaFree((void*)mFeatVec),"Failed to free keyframe memory: mFeatVec");
        checkCudaError(cudaFree((void*)mFeatVecStartIndexes),"Failed to free keyframe memory: mFeatVecStartIndexes");
    }
}