#include "Kernels/CudaWrappers/CudaKeyFrame.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyFrame]: " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

#include <algorithm>

#include <vector>
#include <sstream>

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

        // The grid stores one entry per keypoint that landed in a cell, so it is bounded
        // by the same feature count as the keypoint arrays it indexes into.
        checkCudaError(cudaMalloc((void**)&gridIndices, nFeatures * sizeof(uint16_t)), "KeyFrame::failed to allocate memory for gridIndices");
        checkCudaError(cudaMalloc((void**)&gridIndicesRight, nFeatures * sizeof(uint16_t)), "KeyFrame::failed to allocate memory for gridIndicesRight");

        // mFeatVec holds one entry per feature (~N), not MAX_FEAT_PER_WORD per BoW node;
        // the old guess over-allocated it by ~5x on every keyframe. It grows on demand, so
        // starting from the real expected size costs nothing when a keyframe exceeds it.
        featVecCapacity = 2 * nFeatures;
        featStartIdxCapacity = MAX_FEAT_VEC_SIZE;
        checkCudaError(cudaMalloc((void**)&mFeatVec, featVecCapacity*sizeof(unsigned int)), "KeyFrame::failed to allocate memory for mFeatVec");
        checkCudaError(cudaMalloc((void**)&mFeatVecStartIndexes, featStartIdxCapacity*sizeof(int)), "KeyFrame::failed to allocate memory for mFeatVecStartIndexes");
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

        // Build both grids in CSR form. The previous fixed-stride layout had to clamp
        // each cell to KEYPOINTS_PER_CELL and silently drop the overflow; a packed layout
        // has no per-cell bound to exceed, so nothing is dropped and the stride mismatch
        // between writer and readers is gone with it.
        //
        // Note this is an element-wise narrowing conversion, NOT a memcpy: the source
        // cells are std::vector<size_t>, and copying their bytes into a uint16_t buffer
        // would interleave indices with zeros and feed out-of-bounds reads into
        // mvKeysUn[] and mDescriptors[].
        buildGrid(KF->getMGrid(), gridOffsets, gridIndices, "gridIndices");
        buildGrid(KF->mGridRight, gridOffsetsRight, gridIndicesRight, "gridIndicesRight");

        copyGPUCamera(&camera1, KF->mpCamera);
        copyGPUCamera(&camera2, KF->mpCamera2);
    }

    // mFeatVecStartIndexes held MAX_FEAT_VEC_SIZE (100) ints while this copied mFeatCount
    // of them, and mFeatCount is the number of distinct BoW nodes for the keyframe, which
    // for the feature counts these sequences produce runs well past 100. cudaMemcpy does
    // not reliably fault on a modest overrun inside the same arena, so the excess quietly
    // landed on whatever device allocation followed. Both buffers now grow to fit.
    //
    // The enclosing CudaKeyFrame lives in managed memory and the GPU dereferences these
    // pointers straight out of the struct, so replacing them is visible device-side with
    // no re-upload. Slots are recycled without re-running the constructor, so the
    // capacities persist and a grown keyframe stays grown.
    void CudaKeyFrame::addFeatureVector(const DBoW2::FeatureVector &featVec) {
        mFeatCount = featVec.size();
        if (mFeatCount <= 0)
            return;

        size_t totalFeatures = 0;
        for (const auto &node : featVec)
            totalFeatures += node.second.size();

        std::vector<unsigned int> tmp_mFeatVec(totalFeatures);
        std::vector<int> tmp_mFeatVecStartIndexes(mFeatCount);
        copyFeatVec(tmp_mFeatVec.data(), tmp_mFeatVecStartIndexes.data(), featVec);
        int mFeatVecSize = tmp_mFeatVecStartIndexes[mFeatCount-1];

        if ((size_t)mFeatVecSize > featVecCapacity) {
            cudaFree(mFeatVec);
            featVecCapacity = std::max((size_t)mFeatVecSize, featVecCapacity * 2);
            checkCudaError(cudaMalloc((void**)&mFeatVec, featVecCapacity*sizeof(unsigned int)), "CudaKeyFrame:: failed to grow mFeatVec");
        }
        if ((size_t)mFeatCount > featStartIdxCapacity) {
            cudaFree(mFeatVecStartIndexes);
            featStartIdxCapacity = std::max((size_t)mFeatCount, featStartIdxCapacity * 2);
            checkCudaError(cudaMalloc((void**)&mFeatVecStartIndexes, featStartIdxCapacity*sizeof(int)), "CudaKeyFrame:: failed to grow mFeatVecStartIndexes");
        }

        checkCudaError(cudaMemcpy(mFeatVec, tmp_mFeatVec.data(), mFeatVecSize*sizeof(unsigned int), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mFeatVec to gpu");
        checkCudaError(cudaMemcpy(mFeatVecStartIndexes, tmp_mFeatVecStartIndexes.data(), mFeatCount*sizeof(int), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mFeatVecStartIndexes to gpu");
    }

    // Pack one grid into offsets + indices and upload the indices. An empty grid (the
    // right grid on any non-fisheye camera, where KeyFrame leaves mGridRight unsized)
    // still has to zero its offsets: slots are recycled without re-running the
    // constructor, so stale offsets from a previous keyframe would otherwise be read.
    void CudaKeyFrame::buildGrid(const std::vector<std::vector<std::vector<size_t>>> &grid,
                                 uint16_t *offsets, uint16_t *deviceIndices, const char *name) {
        const int nCells = mnGridCols * mnGridRows;

        if (grid.empty()) {
            std::memset(offsets, 0, (nCells + 1) * sizeof(uint16_t));
            return;
        }

        std::vector<uint16_t> staging;
        staging.reserve(CudaUtils::nFeatures_with_th);

        int total = 0;
        for (int i = 0; i < mnGridCols; ++i) {
            for (int j = 0; j < mnGridRows; ++j) {
                offsets[i * mnGridRows + j] = (uint16_t) total;
                for (size_t idx : grid[i][j])
                    staging.push_back((uint16_t) idx);
                total += (int) grid[i][j].size();
            }
        }
        offsets[nCells] = (uint16_t) total;

        if (total > CudaUtils::nFeatures_with_th) {
            std::ostringstream out;
            out << "CudaKeyFrame::buildGrid: " << name << " needs " << total
                << " entries but is sized " << CudaUtils::nFeatures_with_th;
            fatalError(out.str().c_str());
        }

        if (total > 0)
            checkCudaError(cudaMemcpy(deviceIndices, staging.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice),
                           "CudaKeyFrame:: Failed to copy grid indices to gpu");
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
        checkCudaError(cudaFree((void*)gridIndices),"Failed to free keyframe memory: gridIndices");
        checkCudaError(cudaFree((void*)gridIndicesRight),"Failed to free keyframe memory: gridIndicesRight");
        // Both pointers dangle now; clear the capacities so a reused slot cannot mistake
        // them for live buffers.
        featVecCapacity = 0;
        featStartIdxCapacity = 0;
    }
}