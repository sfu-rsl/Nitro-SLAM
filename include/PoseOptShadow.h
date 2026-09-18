#ifndef POSE_OPT_SHADOW_H
#define POSE_OPT_SHADOW_H

// Shadow comparison of the two pose-optimizer backends on real sequence data.
//
// The synthetic differential tests in Examples/Tests/test_pose_optimization.cc show
// the fused GPU solver reproducing g2o to ~1e-5 m on every scene they build, yet the
// two backends still produce different trajectories over a sequence. Something the
// synthetic scenes do not reproduce is therefore driving them apart, and the way to
// find it is to compare the two on the frames the sequence actually presents.
//
// With NITRO_PO_SHADOW=1 the tracker runs *both* solvers on every pose-optimization
// call: the frame's optimizer-visible state is snapshotted, g2o runs and its answer
// is recorded, the snapshot is restored, the fused kernel runs on the identical
// input, and the two answers are compared. One of them is then kept so the sequence
// continues -- NITRO_PO_SHADOW_KEEP picks which, defaulting to the backend the run
// was configured with, so a shadow run still follows the same trajectory it would
// have followed otherwise.
//
// This costs a second solve per call and so must be run with NITRO_NO_PACING=1: the
// point is per-call divergence, not timing or ATE.
//
// Divergence is written to NITRO_PO_SHADOW_OUT (default pose_opt_shadow.csv), one
// row per call, so the corpus can be sorted for the worst cases and those inputs
// replayed offline rather than re-running the sequence for each experiment.

#include <string>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "ImuTypes.h"

namespace ORB_SLAM3 {

class Frame;
class ConstraintPoseImu;

namespace PoseOptShadow {

// Is shadow mode on for this process?
bool Enabled();
// Which backend's answer the sequence carries forward ("gpu" or "cpu").
bool KeepGpu();

// The part of a Frame the pose optimizers write. Enough to run one solver, put the
// frame back exactly as it was, and run the other on the same input.
struct FrameState {
    Sophus::SE3f Tcw;
    bool hasPose = false;
    Eigen::Vector3f Vw = Eigen::Vector3f::Zero();
    bool hasVelocity = false;
    IMU::Bias bias;
    std::vector<bool> outlier;
    ConstraintPoseImu *cpi = nullptr;

    // PoseInertialOptimizationLastFrame *consumes* the previous frame's prior --
    // "delete pFp->mpcpi; pFp->mpcpi = NULL;" at the end of the g2o version. So the
    // first solver to run destroys part of the second one's input, and without
    // reinstating it the comparison would be between one solver given a prior and
    // another given nothing. The prior's contents are copied out here and a fresh
    // one is rebuilt from them on restore.
    Frame *prev = nullptr;
    bool hasPrevPrior = false;
    Eigen::Matrix3d pRwb;
    Eigen::Vector3d ptwb, pvwb, pbg, pba;
    Eigen::Matrix<double, 15, 15> pH;
};

FrameState Capture(const Frame &F);

// Total visual chi2 at `pose`, over the observations `mask` marks as inliers.
//
// Both solvers are scored on the *same* factor set and the same residual so the
// number answers one question: which of the two actually drove the cost lower on
// this input. A solver that reaches a lower chi2 will also carry more observations
// under a fixed gate, which would make an inlier surplus a consequence of fitting
// better rather than of gating wrongly.
double VisualChi2(const Frame &F, const Sophus::SE3f &Tcw,
                  const std::vector<bool> &mask);
// Restores `s` onto `F`. Any ConstraintPoseImu the solver allocated in the meantime
// is released, since the frame is going back to the prior it had before the call.
void Restore(Frame &F, const FrameState &s);

// Record one call: the two backends' results on identical input.
// `F` is only read, to classify each disagreeing observation: a surplus that is
// concentrated in one kind of factor (close vs far mono, stereo, right camera)
// points at that branch of the gate, where a surplus spread evenly over all of them
// is just the two solvers stopping at slightly different poses.
void Record(int variant, long frame_id, int n_factors, const FrameState &cpu,
            int cpu_inliers, const FrameState &gpu, int gpu_inliers, const Frame &F);

// Flush the corpus and print a summary. Safe to call when disabled.
void Finish();

}  // namespace PoseOptShadow
}  // namespace ORB_SLAM3

#endif
