#ifndef CUDA_KEYFRAME_H
#define CUDA_KEYFRAME_H

#include "CudaKeyPoint.h"
#include "CudaCamera.h"
#include "KeyFrame.h"
#include "../CudaUtils.h"


#define MAX_FEAT_VEC_SIZE 100
#define MAX_FEAT_PER_WORD 100

// This header is reached from host-only translation units (KeyFrame.cc pulls it in
// via CudaKeyFrameAllocator.h and is compiled by the CXX compiler), so the CUDA
// qualifiers cannot appear unguarded on a function definition.
#ifdef __CUDACC__
#define CUDAKF_HD __host__ __device__
#else
#define CUDAKF_HD
#endif

namespace MAPPING_DATA_WRAPPER {

class CudaKeyFrame {
    private:
        void initializeMemory();
        void copyGPUCamera(CudaCamera *out, ORB_SLAM3::GeometricCamera *camera);
        void copyFeatVec(unsigned int *out, int *outIndexes, const DBoW2::FeatureVector &inp);
        void buildGrid(const std::vector<std::vector<std::vector<size_t>>> &grid,
                       uint16_t *offsets, uint16_t *deviceIndices, const char *name);

    public:
        CudaKeyFrame();
        void setGPUAddress(CudaKeyFrame* ptr);
        void setMemory(ORB_SLAM3::KeyFrame* KF);
        void setMemory(ORB_SLAM3::KeyFrame &KF);
        void addFeatureVector(const DBoW2::FeatureVector &featVec);
        void setAsEmpty() { isEmpty = true; };
        void freeMemory();

        // Candidates in grid cell (ix, iy). Writes the cell's count to n and returns a
        // pointer to its first feature index. The returned range is valid until the
        // owning keyframe's slot is recycled.
        CUDAKF_HD inline const uint16_t* cellPtr(int ix, int iy, bool bRight, int &n) const {
            const int c = ix * mnGridRows + iy;
            const uint16_t* off = bRight ? gridOffsetsRight : gridOffsets;
            n = (int)(off[c + 1] - off[c]);
            return (bRight ? gridIndicesRight : gridIndices) + off[c];
        }

    public:
        bool isEmpty;
        long unsigned int mnId;
        int Nleft;
        float mfLogScaleFactor;
        int mnScaleLevels;
        float mnMinX;
        float mnMaxX;
        float mnMinY;
        float mnMaxY;
        float mfGridElementWidthInv;
        float mfGridElementHeightInv;
        int mnGridCols;
        int mnGridRows;
        float mThDepth;
        float* mvDepth;
        float mbf;

        float fx;
        float fy;
        float cx;
        float cy;

        CudaKeyFrame* gpuAddr;

        size_t mvScaleFactors_size;
        float* mvScaleFactors;

        size_t mvKeys_size, mvKeysRight_size, mvKeysUn_size;
        const CudaKeyPoint *mvKeys, *mvKeysRight;
        CudaKeyPoint *mvKeysUn;

        size_t mvuRight_size;
        float* mvuRight;

        size_t mvInvLevelSigma2_size;
        float* mvInvLevelSigma2;

        int mDescriptor_rows;
        const uint8_t* mDescriptors;

        // The feature grid, stored CSR-style: cell c holds the indices in
        // [gridOffsets[c], gridOffsets[c+1]) of gridIndices. The previous layout gave
        // every one of the 3072 cells a fixed 20 slots of std::size_t, which at the ~0.33
        // keypoints per cell these sequences actually produce made the two grids ~1 MB of
        // almost pure padding -- 86% of the whole keyframe mirror, multiplied by every
        // keyframe in the map.
        //
        // uint16_t is the natural index width: Frame::AssignFeaturesToGrid only ever
        // stores i < Nleft, i - Nleft < NRight, or i < N, so a grid entry is always below
        // the frame's feature count. CudaUtils::loadSetting refuses a configuration whose
        // feature count could reach UINT16_MAX, so the bound cannot be crossed silently.
        //
        // Offsets stay inline (fixed size, one per cell); the index arrays are device
        // buffers sized like the keypoint arrays they index into.
        uint16_t gridOffsets[FRAME_GRID_COLS * FRAME_GRID_ROWS + 1];
        uint16_t gridOffsetsRight[FRAME_GRID_COLS * FRAME_GRID_ROWS + 1];
        uint16_t *gridIndices, *gridIndicesRight;

        CudaCamera camera1, camera2;

        int mFeatCount;
        unsigned int *mFeatVec;
        int *mFeatVecStartIndexes;
        // Capacities of the two buffers above, in elements. The BoW node count per
        // keyframe is vocabulary- and scene-dependent and routinely exceeds the old
        // fixed MAX_FEAT_VEC_SIZE, so addFeatureVector() grows them on demand.
        size_t featVecCapacity;
        size_t featStartIdxCapacity;
};
}

#endif // CUDA_KEYFRAME_H