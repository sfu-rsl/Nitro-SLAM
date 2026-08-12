#ifndef MAP_POINT_CANDIDATE_KERNEL_H
#define MAP_POINT_CANDIDATE_KERNEL_H

#include <vector>
#include <utility>

#include "KeyFrame.h"
#include "KernelInterface.h"
#include "CudaWrappers/CudaKeyFrame.h"

// Second stage of CreateNewMapPointsGPU2: given the correspondences produced by
// TriangulationMatchKernel, evaluate every candidate's geometry in parallel and hand
// the host a compacted list of the ones that survive.
//
// The host is left with only the work that genuinely needs to be serial - allocating
// MapPoint objects and wiring the observation graph - because that is also the part
// that carries the ordering dependency: a keypoint may be claimed by only one
// neighbour, and which one wins must follow covisibility order to match the CPU.
//
// Everything before that is per-candidate independent and runs on the GPU: ray
// unprojection, parallax, linear or stereo triangulation, cheirality, both reprojection
// error tests, and the scale-consistency ratio. Results are emitted in
// (neighbour, idx1) order so the host can apply the ordering rule by a single pass.
class MapPointCandidateKernel : public KernelInterface {
public:
    MapPointCandidateKernel() { memory_is_initialized = false; }

    void initialize() override;
    void shutdown() override;
    void saveStats(const std::string &file_path) override;
    void launch() override {}

    struct Candidate {
        int   neighbourSlot;
        int   idx1;
        int   idx2;
        float x, y, z;      // world coordinates
        bool  bStereoPoint; // came from a stereo unprojection rather than triangulation
    };

    // Per-keyframe scalars the kernel needs that CudaKeyFrame does not carry.
    struct FrameInfo {
        float mb;
        float mbf;
    };

    void launch(ORB_SLAM3::KeyFrame* pKF1,
                const std::vector<ORB_SLAM3::KeyFrame*> &vpNeighKFs,       // the kept ones
                const std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices,
                bool bInertial, bool bFarPoints, float thFarPoints, float ratioFactor,
                std::vector<Candidate> &outCandidates);

private:
    bool memory_is_initialized;
    unsigned long nCalls = 0, nEvaluated = 0, nAccepted = 0;
};

#endif // MAP_POINT_CANDIDATE_KERNEL_H
