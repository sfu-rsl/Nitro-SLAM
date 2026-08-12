#ifndef TRIANGULATION_MATCH_KERNEL_H
#define TRIANGULATION_MATCH_KERNEL_H

#include <vector>
#include <utility>
#include <cstddef>

#include "KeyFrame.h"
#include "KernelInterface.h"
#include "CudaWrappers/CudaKeyFrame.h"

// Batched correspondence search for LocalMapping::CreateNewMapPoints.
//
// Rewrite of SearchForTriangulationKernel, which is kept intact alongside it. The
// intent is unchanged - decouple the correspondence search from triangulation so the
// search can be batched across all covisible neighbours on the GPU, while triangulation
// stays on the host - but this version is a direct port of the CPU inner loop in
// ORBmatcher::SearchForTriangulation rather than a restructuring of it.
//
// Differences from the original implementation, all deliberate:
//
//  * One thread per (neighbour, idx1) work item instead of one thread per BoW node.
//    The old mapping put a whole node's features on a single thread and sized the block
//    as dim3(MAX_FEAT_VEC_SIZE), which serialised the inner loop and hard-capped the
//    launch geometry. Work items are flattened on the host, so the grid scales with the
//    real amount of work and nothing is capped.
//  * Each (neighbour, idx1) pair is unique - a feature belongs to exactly one BoW node
//    at a given level - so results are written without atomics and without a
//    fixed-capacity scratch array.
//  * Camera model is dispatched on the actual camera type rather than inferred from
//    Nleft, which is only a proxy and is wrong for monocular fisheye.
//  * bCoarse is supplied by the caller as a real predicate.
//
// The caller is still responsible for re-applying the "skip keypoints that already have
// a MapPoint" rule while triangulating: the batched search cannot observe MapPoints
// created during the same call, so it works from a pre-batch snapshot.
class TriangulationMatchKernel : public KernelInterface {
public:
    TriangulationMatchKernel() { memory_is_initialized = false; }

    void initialize() override;
    void shutdown() override;
    void saveStats(const std::string &file_path) override;
    void launch() override {}

    // Returns, for each neighbour that passed the baseline check, the list of
    // (idx1, idx2) correspondences. vpNeighKFsIndexes indexes into vpNeighKFs and is
    // filled in covisibility order, matching the CPU traversal.
    void launch(ORB_SLAM3::KeyFrame* pCurrentKF,
                const std::vector<ORB_SLAM3::KeyFrame*> &vpNeighKFs,
                bool bMonocular, bool bCoarse,
                std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices,
                std::vector<size_t> &vpNeighKFsIndexes);

private:
    bool memory_is_initialized;

    // Device scratch, grown on demand rather than fixed at a compile-time capacity.
    void ensureCapacity(size_t nWorkItems, size_t nNeighbours, size_t nMatchSlots);

    int    *d_workIdx1        = nullptr;   // per work item: keypoint in the current KF
    int    *d_workNeighbour   = nullptr;   // per work item: neighbour slot
    int    *d_workNode2       = nullptr;   // per work item: BoW node ordinal in KF2
    size_t  workCapacity      = 0;

    MAPPING_DATA_WRAPPER::CudaKeyFrame **d_neighKFs = nullptr;
    Eigen::Matrix3f *d_Rll = nullptr, *d_Rlr = nullptr, *d_Rrl = nullptr, *d_Rrr = nullptr;
    Eigen::Vector3f *d_tll = nullptr, *d_tlr = nullptr, *d_trl = nullptr, *d_trr = nullptr;
    Eigen::Vector2f *d_ep   = nullptr;
    size_t  neighbourCapacity = 0;

    int  *d_matches   = nullptr;           // nNeighbours * maxFeatures, -1 == no match
    bool *d_mpExists1 = nullptr;           // maxFeatures
    bool *d_mpExists2 = nullptr;           // nNeighbours * maxFeatures
    size_t matchCapacity = 0;

    unsigned long nCalls = 0, nWorkItemsTotal = 0, nMatchesTotal = 0;
};

#endif // TRIANGULATION_MATCH_KERNEL_H
