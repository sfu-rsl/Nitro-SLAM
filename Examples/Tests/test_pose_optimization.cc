// Differential test for the fused GPU pose optimizer.
//
// OptimizerGPU::PoseOptimizationFused replaces g2o for the pose-only solve inside
// TrackLocalMap. The two are meant to be interchangeable: same initial pose, same
// observations, same chi2 gates, so the same pose and the same inlier set out.
// This builds synthetic scenes where the answer is known, runs both solvers from an
// identical starting state, and reports where they disagree -- with each other and
// with the truth they were both given enough information to recover.
//
// Scenes are synthetic on purpose. A capture from a real sequence tells you that
// something drifted; a scene with a known pose tells you which of the two solvers
// was wrong about it, and lets outlier fraction and noise be dialled independently.
//
// Covers both camera geometries the sweep uses, because they take different paths
// through the factor builder: pinhole (EuRoC, mono and stereo observations, one
// camera) and Kannala-Brandt fisheye stereo (TUM-VI, two cameras, every observation
// monocular in its own camera). outdoors6 is the fisheye case.
//
// Build:  make test_pose_optimization
// Run:    ./test_pose_optimization [--seeds N] [--verbose]

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

#include "Frame.h"
#include "G2oTypes.h"
#include "ImuTypes.h"
#include "MapPoint.h"
#include "Optimizer.h"
#include "CameraModels/KannalaBrandt8.h"
#include "CameraModels/Pinhole.h"
#include "Kernels/TrackingKernelController.h"

using namespace ORB_SLAM3;

namespace {

// Tolerances. The two solvers minimise the same cost but not with the same
// arithmetic -- g2o works in double, the fused kernel in the compile-time FT -- so
// they are not expected to agree bit for bit. They are expected to land on the same
// minimum: a millimetre and a hundredth of a degree is far tighter than the
// difference between a right and a wrong answer here, and far looser than the noise
// floor between two correct ones.
constexpr double kPosTolM = 1e-3;
// On a scene the reference cannot solve either, the minimum is a valley rather than
// a point, so the two solvers stop at different places along it. This is the
// agreement bar there -- still two orders of magnitude below the divergence the
// Jacobi scaling used to produce at these depths.
constexpr double kPosTolUnobservableM = 2e-2;
constexpr double kRotTolDeg = 5e-2;
// Recovering the planted pose is a weaker claim than agreeing, since both solvers
// start perturbed and stop on their own criteria; this is the "did it converge at
// all" gate rather than an accuracy claim.
constexpr double kTruthPosTolM = 0.05;
constexpr double kTruthRotTolDeg = 1.0;

constexpr int kLevels = 8;
constexpr float kScaleFactor = 1.2f;

struct Stats {
    int checks = 0, failures = 0;
};

void report(Stats &s, bool ok, const std::string &what, const std::string &detail) {
    s.checks++;
    if (!ok) {
        s.failures++;
        std::cout << "    FAIL  " << what << "  " << detail << "\n";
    } else {
        std::cout << "    ok    " << what << "  " << detail << "\n";
    }
}

double rot_deg(const Sophus::SE3f &a, const Sophus::SE3f &b) {
    const Eigen::Matrix3d Ra = a.rotationMatrix().cast<double>();
    const Eigen::Matrix3d Rb = b.rotationMatrix().cast<double>();
    const double c = ((Ra.transpose() * Rb).trace() - 1.0) * 0.5;
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

double pos_m(const Sophus::SE3f &a, const Sophus::SE3f &b) {
    return (a.translation() - b.translation()).cast<double>().norm();
}

// A synthetic scene: the pose that generated the observations, the map points, and
// which of them were deliberately corrupted.
struct Scene {
    Sophus::SE3f Tcw_true;
    Sophus::SE3f Tcw_init;
    std::vector<MapPoint *> points;
    std::vector<bool> planted_outlier;
    int n_outliers = 0;
};

// Frame owns none of this; the test holds it so it outlives the frame.
struct Arena {
    std::vector<MapPoint *> points;
    ~Arena() {
        for (MapPoint *p : points) delete p;
    }
};

Sophus::SE3f perturb(const Sophus::SE3f &T, double dpos, double ddeg,
                     std::mt19937 &rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::Vector3d axis(g(rng), g(rng), g(rng));
    if (axis.norm() < 1e-9) axis = Eigen::Vector3d::UnitX();
    axis.normalize();
    const Eigen::Matrix3d dR =
        Eigen::AngleAxisd(ddeg * M_PI / 180.0, axis).toRotationMatrix();
    Eigen::Vector3d dt(g(rng), g(rng), g(rng));
    if (dt.norm() > 1e-9) dt = dt.normalized() * dpos;
    return Sophus::SE3f((dR.cast<float>() * T.rotationMatrix()).eval(),
                        (T.translation() + dt.cast<float>()).eval());
}

// Build a frame whose observations are the projections of `points` under
// Tcw_true, with pixel noise and a planted outlier fraction.
//
// `stereo` gives every point a right-eye u so the stereo chi2 gate is exercised;
// `fisheye` builds the two-camera layout instead, where the second half of the
// observations belong to the right camera and are monocular in it.
// `spread` is the lateral extent of the point cloud and `zmin`/`zmax` its depth
// range. Shrinking the spread while pushing the depth out narrows the cone the
// points subtend, which is what makes the pose weakly observable and the Hessian
// ill-conditioned -- the regime where a single-precision solve would be expected to
// part company with a double-precision one, and the regime a well-spread synthetic
// scene never enters.
void build_scene(Scene &scene, Arena &arena, Frame &frame, GeometricCamera *cam1,
                 GeometricCamera *cam2, int n_points, double noise_px,
                 double outlier_frac, bool stereo, std::mt19937 &rng,
                 double spread = 2.5, double zmin = 2.0, double zmax = 12.0) {
    std::normal_distribution<double> gauss(0.0, noise_px);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<double> ux(-spread, spread);
    std::uniform_real_distribution<double> uz(zmin, zmax);
    std::uniform_int_distribution<int> uoct(0, kLevels - 1);
    std::uniform_real_distribution<double> ubad(-90.0, 90.0);

    // A pose that is not near identity, so a bug in the rotation update cannot hide.
    Eigen::Matrix3d R = (Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(-0.20, Eigen::Vector3d::UnitY()) *
                         Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitX()))
                            .toRotationMatrix();
    Eigen::Vector3d t(0.4, -0.25, 0.15);
    scene.Tcw_true = Sophus::SE3f(R.cast<float>(), t.cast<float>());

    const bool fisheye = (cam2 != nullptr);
    const int nleft = fisheye ? n_points / 2 : n_points;

    frame.N = n_points;
    frame.mpCamera = cam1;
    frame.mpCamera2 = cam2;
    frame.Nleft = fisheye ? nleft : -1;
    frame.Nright = fisheye ? (n_points - nleft) : -1;
    frame.mbf = 40.0f;   // baseline * fx, in pixels
    // g2o's stereo edge reprojects from these scalars rather than through the
    // camera model, so a frame that only carries mpCamera silently produces
    // meaningless stereo residuals. The fused kernel uses the camera model for
    // both, which is why only the reference path noticed.
    frame.fx = cam1->getParameter(0); frame.fy = cam1->getParameter(1);
    frame.cx = cam1->getParameter(2); frame.cy = cam1->getParameter(3);
    frame.invfx = 1.0f / frame.fx; frame.invfy = 1.0f / frame.fy;
    frame.mvpMapPoints.assign(n_points, nullptr);
    frame.mvbOutlier.assign(n_points, false);
    frame.mvuRight.assign(n_points, -1.0f);
    frame.mvKeys.clear();
    frame.mvKeysUn.clear();
    frame.mvKeysRight.clear();
    frame.mvInvLevelSigma2.resize(kLevels);
    for (int l = 0; l < kLevels; l++) {
        const float s = std::pow(kScaleFactor, l);
        frame.mvInvLevelSigma2[l] = 1.0f / (s * s);
    }
    // Two-camera rig: the right camera sits 10 cm to the right of the left one and
    // is yawed slightly inward, so a bug that drops or misapplies the rig rotation
    // shows up rather than cancelling against an identity.
    const Sophus::SE3f Tlr(
        Sophus::SO3f::exp(Eigen::Vector3f(0.0f, 0.02f, 0.0f)).matrix(),
        Eigen::Vector3f(0.10f, 0.0f, 0.0f));
    const Sophus::SE3f Trl = Tlr.inverse();
    frame.SetRelativePoseTlr(Tlr);

    scene.points.assign(n_points, nullptr);
    scene.planted_outlier.assign(n_points, false);

    std::vector<cv::KeyPoint> left, right;
    for (int i = 0; i < n_points; i++) {
        // A point comfortably in front of whichever camera observes it.
        Eigen::Vector3f Xc(static_cast<float>(ux(rng)), static_cast<float>(ux(rng)),
                           static_cast<float>(uz(rng)));
        const bool on_right = fisheye && i >= nleft;
        Eigen::Vector3f Xc_obs = on_right ? (Trl * Xc).eval() : Xc;

        // World position consistent with the true pose.
        const Eigen::Vector3f Xw =
            scene.Tcw_true.inverse() * Xc;

        MapPoint *mp = new MapPoint();
        mp->SetWorldPos(Xw);
        arena.points.push_back(mp);
        scene.points[i] = mp;
        frame.mvpMapPoints[i] = mp;

        GeometricCamera *cam = on_right ? cam2 : cam1;
        const cv::Point2f uv = cam->project(cv::Point3f(Xc_obs.x(), Xc_obs.y(), Xc_obs.z()));

        const bool bad = u01(rng) < outlier_frac;
        scene.planted_outlier[i] = bad;
        if (bad) scene.n_outliers++;

        cv::KeyPoint kp;
        kp.pt.x = uv.x + static_cast<float>(bad ? ubad(rng) : gauss(rng));
        kp.pt.y = uv.y + static_cast<float>(bad ? ubad(rng) : gauss(rng));
        kp.octave = uoct(rng);

        if (on_right) {
            right.push_back(kp);
        } else {
            left.push_back(kp);
            if (stereo && !fisheye) {
                // u_right = u_left - bf/Z, corrupted alongside the left observation.
                const float ur = uv.x - frame.mbf / Xc.z();
                frame.mvuRight[i] =
                    ur + static_cast<float>(bad ? ubad(rng) : gauss(rng));
            }
        }
    }

    frame.mvKeys = left;
    frame.mvKeysRight = right;
    // The pinhole path reads undistorted keys; the fisheye path reads raw ones.
    frame.mvKeysUn = left;
    if (fisheye) {
        // mvKeysUn is indexed by i over the whole frame in the fisheye branch's
        // sibling code paths, so keep it the full length to avoid a short read.
        frame.mvKeysUn.insert(frame.mvKeysUn.end(), right.begin(), right.end());
    }

    scene.Tcw_init = perturb(scene.Tcw_true, 0.08, 1.5, rng);
}

struct Outcome {
    Sophus::SE3f pose;
    int inliers = 0;
    std::vector<bool> outlier;
};

Outcome run_solver(Frame &frame, const Scene &scene, bool gpu) {
    frame.SetPose(scene.Tcw_init);
    std::fill(frame.mvbOutlier.begin(), frame.mvbOutlier.end(), false);
    Outcome o;
    o.inliers = gpu ? OptimizerGPU::PoseOptimizationFused(&frame)
                    : Optimizer::PoseOptimization(&frame);
    o.pose = frame.GetPose();
    o.outlier = frame.mvbOutlier;
    return o;
}

int classification_diff(const Outcome &a, const Outcome &b) {
    int n = 0;
    for (size_t i = 0; i < a.outlier.size() && i < b.outlier.size(); i++)
        if (a.outlier[i] != b.outlier[i]) n++;
    return n;
}

void run_case(Stats &s, const std::string &name, GeometricCamera *cam1,
              GeometricCamera *cam2, int n_points, double noise_px,
              double outlier_frac, bool stereo, unsigned seed, bool verbose,
              double spread = 2.5, double zmin = 2.0, double zmax = 12.0) {
    std::mt19937 rng(seed);
    Arena arena;
    Frame frame;
    Scene scene;
    build_scene(scene, arena, frame, cam1, cam2, n_points, noise_px, outlier_frac,
                stereo, rng, spread, zmin, zmax);

    const Outcome g2o = run_solver(frame, scene, false);
    const Outcome gpu = run_solver(frame, scene, true);

    char buf[512];
    std::cout << "  " << name << "  (seed " << seed << ", " << n_points
              << " pts, " << scene.n_outliers << " planted outliers)\n";
    if (verbose) {
        std::snprintf(buf, sizeof(buf),
                      "g2o inliers=%d  gpu inliers=%d  init err %.4f m / %.3f deg",
                      g2o.inliers, gpu.inliers, pos_m(scene.Tcw_init, scene.Tcw_true),
                      rot_deg(scene.Tcw_init, scene.Tcw_true));
        std::cout << "      " << buf << "\n";
    }

    // 1. Does g2o -- the reference -- recover the planted pose? If not, the scene
    //    is unreasonable and nothing downstream of it means anything.
    const double g_pos = pos_m(g2o.pose, scene.Tcw_true);
    const double g_rot = rot_deg(g2o.pose, scene.Tcw_true);
    const bool observable = g_pos < kTruthPosTolM && g_rot < kTruthRotTolDeg;
    std::snprintf(buf, sizeof(buf), "g2o vs truth: %.5f m, %.4f deg", g_pos, g_rot);
    // Whether the scene is observable at all is a property of the geometry, not of
    // either solver. The stress cases are deliberately built past that edge -- far
    // points in a narrow cone, where depth is barely constrained -- so the reference
    // failing to recover the planted pose is the expected outcome there, and
    // demanding it of the contender would only be asserting that the scene is easy.
    // What still has to hold on an unobservable scene is that the two solvers agree.
    if (observable) {
        report(s, true, "reference recovers pose", buf);
    } else {
        std::cout << "    --    scene not observable by the reference either  " << buf
                  << "\n";
    }

    // 2. The claim under test: the fused kernel lands where g2o lands.
    const double d_pos = pos_m(gpu.pose, g2o.pose);
    const double d_rot = rot_deg(gpu.pose, g2o.pose);
    std::snprintf(buf, sizeof(buf), "gpu vs g2o: %.5f m, %.4f deg", d_pos, d_rot);
    report(s, d_pos < (observable ? kPosTolM : kPosTolUnobservableM) &&
                  d_rot < kRotTolDeg, "gpu agrees with g2o", buf);

    // 3. And recovers the pose in its own right, which is what tracking consumes.
    const double u_pos = pos_m(gpu.pose, scene.Tcw_true);
    const double u_rot = rot_deg(gpu.pose, scene.Tcw_true);
    std::snprintf(buf, sizeof(buf), "gpu vs truth: %.5f m, %.4f deg", u_pos, u_rot);
    if (observable) {
        report(s, u_pos < kTruthPosTolM && u_rot < kTruthRotTolDeg,
               "gpu recovers pose", buf);
    }

    // 4. Inlier count and outlier set drive which points survive into the next
    //    frame, so a solver that finds the right pose but rejects the wrong points
    //    still degrades tracking.
    std::snprintf(buf, sizeof(buf), "g2o=%d gpu=%d, %d classifications differ",
                  g2o.inliers, gpu.inliers, classification_diff(g2o, gpu));
    report(s, std::abs(g2o.inliers - gpu.inliers) <= std::max(1, n_points / 50),
           "inlier counts agree", buf);
}


// ---------------------------------------------------------------------------
// Inertial variant
//
// PoseOptimizationFused above is only 45 of the ~29,000 pose-optimization calls a
// long sequence makes; the other 99.8% go through the two inertial solvers, which
// carry velocity, gyro and accelerometer biases, an IMU preintegration between the
// two frames and a marginalisation prior on the previous one. That is where a
// discrepancy would both hide and matter, so it gets its own scene.
//
// The motion is deliberately simple -- constant world velocity, no rotation -- so
// the IMU measurements have a closed form: with zero angular rate the gyro reads
// its bias, and with zero world acceleration the accelerometer reads the specific
// force R_wb^T * (-g). Anything the preintegration produces from that is checkable
// by hand, which matters when the thing under test is itself an optimizer.
// ---------------------------------------------------------------------------

struct InertialScene {
    Sophus::SE3f Tcw_true;
    Eigen::Matrix3f Rwb_true;
    Eigen::Vector3f twb_true, vel_true;
};

// Prior on the previous frame's 15-dof state. Tight enough to anchor the problem,
// loose enough that the visual and inertial residuals still decide the answer.
Matrix15d prior_information(double sigma_pose, double sigma_vel, double sigma_bias) {
    Matrix15d H = Matrix15d::Zero();
    for (int i = 0; i < 6; i++) H(i, i) = 1.0 / (sigma_pose * sigma_pose);
    for (int i = 6; i < 9; i++) H(i, i) = 1.0 / (sigma_vel * sigma_vel);
    for (int i = 9; i < 15; i++) H(i, i) = 1.0 / (sigma_bias * sigma_bias);
    return H;
}

void build_inertial_scene(InertialScene &scene, Arena &arena, Frame &prev,
                          Frame &curr, GeometricCamera *cam, GeometricCamera *cam2,
                          IMU::Preintegrated *&preint, int n_points,
                          double noise_px, double outlier_frac, std::mt19937 &rng,
                          const Eigen::Vector3f &omega = Eigen::Vector3f::Zero(),
                          const IMU::Bias &true_bias = IMU::Bias()) {
    const float dt = 0.05f;                       // 20 Hz, as the sequences run
    const Eigen::Vector3f gravity(0.0f, 0.0f, -IMU::GRAVITY_VALUE);

    // Body frame and camera frame coincide, so the visual scene can be built in
    // camera coordinates without a second transform to reason about.
    IMU::Calib calib(Sophus::SE3f(), 1.7e-4f, 2.0e-3f, 1.9e-5f, 3.0e-3f);

    scene.Rwb_true = Eigen::Matrix3f::Identity();
    scene.twb_true = Eigen::Vector3f(0.4f, -0.25f, 0.15f);
    scene.vel_true = Eigen::Vector3f(1.2f, 0.3f, -0.1f);

    const Eigen::Matrix3f Rwb0 = scene.Rwb_true;
    const Eigen::Vector3f twb0 = scene.twb_true;
    const Eigen::Vector3f twb1 = twb0 + scene.vel_true * dt;

    // Previous frame: true state, and a prior centred on it.
    prev.mImuCalib = calib;
    prev.mImuBias = IMU::Bias();
    prev.SetImuPoseVelocity(Rwb0, twb0, scene.vel_true);
    prev.mpcpi = new ConstraintPoseImu(
        Rwb0.cast<double>(), twb0.cast<double>(), scene.vel_true.cast<double>(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        prior_information(1e-3, 1e-3, 1e-4));

    // World acceleration stays zero, so the accelerometer reads the specific force
    // R_wb(t)^T * (-g); with a body-frame angular rate the rotation moves through
    // the interval and each substep gets its own R. Measurements carry the true
    // bias, while the preintegration is built at the *estimated* bias (zero), which
    // is the situation the solver is actually asked to handle: it has to work out
    // the correction rather than being handed it.
    const Eigen::Vector3f ba_true(true_bias.bax, true_bias.bay, true_bias.baz);
    const Eigen::Vector3f bg_true(true_bias.bwx, true_bias.bwy, true_bias.bwz);
    preint = new IMU::Preintegrated(IMU::Bias(), calib);
    const int substeps = 10;
    const float sub_dt = dt / substeps;
    for (int k = 0; k < substeps; k++) {
        const Eigen::Vector3f w = omega * (static_cast<float>(k) + 0.5f) * sub_dt;
        const Eigen::Matrix3f Rk = Rwb0 * Sophus::SO3f::exp(w).matrix();
        const Eigen::Vector3f acc = Rk.transpose() * (-gravity) + ba_true;
        const Eigen::Vector3f gyr = omega + bg_true;
        preint->IntegrateNewMeasurement(acc, gyr, sub_dt);
    }
    const Eigen::Matrix3f Rwb1 = Rwb0 * Sophus::SO3f::exp(omega * dt).matrix();

    // Current frame: same rotation, advanced by the velocity.
    curr.mImuCalib = calib;
    curr.mImuBias = IMU::Bias();
    curr.mpPrevFrame = &prev;
    // The frame-to-frame solver reads mpImuPreintegratedFrame for the motion
    // constraint and mpImuPreintegrated for the bias random-walk covariance. In the
    // system they differ (previous frame vs last keyframe); here the previous frame
    // is the reference for both, which is the configuration where they coincide.
    curr.mpImuPreintegratedFrame = preint;
    curr.mpImuPreintegrated = preint;
    curr.mpcpi = nullptr;

    scene.Tcw_true = Sophus::SE3f(Rwb1.transpose(), (-Rwb1.transpose() * twb1).eval());

    // Visual observations of the same kind the pose-only scene uses.
    std::normal_distribution<double> gauss(0.0, noise_px);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<double> ux(-2.5, 2.5);
    std::uniform_real_distribution<double> uz(2.0, 12.0);
    std::uniform_int_distribution<int> uoct(0, kLevels - 1);
    std::uniform_real_distribution<double> ubad(-90.0, 90.0);

    curr.N = n_points;
    curr.mpCamera = cam;
    curr.mpCamera2 = cam2;
    curr.Nleft = cam2 ? n_points / 2 : -1;
    curr.Nright = cam2 ? (n_points - n_points / 2) : -1;
    const Sophus::SE3f Tlr_i(
        Sophus::SO3f::exp(Eigen::Vector3f(0.0f, 0.02f, 0.0f)).matrix(),
        Eigen::Vector3f(0.10f, 0.0f, 0.0f));
    curr.SetRelativePoseTlr(Tlr_i);
    curr.mbf = 40.0f;
    curr.fx = cam->getParameter(0); curr.fy = cam->getParameter(1);
    curr.cx = cam->getParameter(2); curr.cy = cam->getParameter(3);
    curr.invfx = 1.0f / curr.fx;    curr.invfy = 1.0f / curr.fy;
    curr.mvpMapPoints.assign(n_points, nullptr);
    curr.mvbOutlier.assign(n_points, false);
    curr.mvuRight.assign(n_points, -1.0f);
    curr.mvInvLevelSigma2.resize(kLevels);
    for (int l = 0; l < kLevels; l++) {
        const float sc = std::pow(kScaleFactor, l);
        curr.mvInvLevelSigma2[l] = 1.0f / (sc * sc);
    }

    std::vector<cv::KeyPoint> keys, keys_right;
    const int nleft_i = cam2 ? n_points / 2 : n_points;
    for (int i = 0; i < n_points; i++) {
        Eigen::Vector3f Xc(static_cast<float>(ux(rng)), static_cast<float>(ux(rng)),
                           static_cast<float>(uz(rng)));
        const Eigen::Vector3f Xw = scene.Tcw_true.inverse() * Xc;
        MapPoint *mp = new MapPoint();
        mp->SetWorldPos(Xw);
        mp->mTrackDepth = Xc.z();
        arena.points.push_back(mp);
        curr.mvpMapPoints[i] = mp;

        const bool on_right = cam2 && i >= nleft_i;
        const Eigen::Vector3f Xc_obs = on_right ? (Tlr_i.inverse() * Xc).eval() : Xc;
        GeometricCamera *c = on_right ? cam2 : cam;
        const cv::Point2f uv =
            c->project(cv::Point3f(Xc_obs.x(), Xc_obs.y(), Xc_obs.z()));
        const bool bad = u01(rng) < outlier_frac;
        cv::KeyPoint kp;
        kp.pt.x = uv.x + static_cast<float>(bad ? ubad(rng) : gauss(rng));
        kp.pt.y = uv.y + static_cast<float>(bad ? ubad(rng) : gauss(rng));
        kp.octave = uoct(rng);
        (on_right ? keys_right : keys).push_back(kp);
    }
    curr.mvKeys = keys;
    curr.mvKeysUn = keys;
    curr.mvKeysRight = keys_right;
    if (cam2) curr.mvKeysUn.insert(curr.mvKeysUn.end(), keys_right.begin(),
                                   keys_right.end());
}

void run_inertial_case(Stats &s, const std::string &name, GeometricCamera *cam,
                       int n_points, double noise_px, double outlier_frac,
                       unsigned seed, bool verbose, bool hard = false,
                       GeometricCamera *cam2 = nullptr) {
    char buf[512];
    std::cout << "  " << name << "  (seed " << seed << ", " << n_points << " pts)\n";

    // Each solver gets its own scene: both mutate the frame's prior and its state,
    // so replaying one frame through the second solver would not be a fair start.
    Sophus::SE3f pose[2];
    Eigen::Vector3f vel[2];
    int inliers[2] = {0, 0};
    InertialScene truth;

    for (int which = 0; which < 2; which++) {
        std::mt19937 rng(seed);
        Arena arena;
        Frame prev, curr;
        IMU::Preintegrated *preint = nullptr;
        InertialScene scene;
        // Hard mode turns on the two things a fused solver is most likely to get
        // subtly wrong: a rotation inside the preintegration interval, and a bias
        // the optimizer has to estimate rather than being given.
        const Eigen::Vector3f omega =
            hard ? Eigen::Vector3f(0.30f, -0.45f, 0.20f) : Eigen::Vector3f::Zero();
        const IMU::Bias bias =
            hard ? IMU::Bias(0.03f, -0.02f, 0.04f, 0.006f, -0.004f, 0.005f)
                 : IMU::Bias();
        build_inertial_scene(scene, arena, prev, curr, cam, cam2, preint, n_points,
                             noise_px, outlier_frac, rng, omega, bias);
        truth = scene;

        // Start both from the same perturbed state.
        const Sophus::SE3f Tcw_init = perturb(scene.Tcw_true, 0.05, 1.0, rng);
        const Eigen::Matrix3f Rwb_init = Tcw_init.rotationMatrix().transpose();
        const Eigen::Vector3f twb_init = -Rwb_init * Tcw_init.translation();
        curr.SetImuPoseVelocity(Rwb_init, twb_init,
                                scene.vel_true + Eigen::Vector3f(0.05f, -0.03f, 0.02f));

        inliers[which] = which == 0
            ? Optimizer::PoseInertialOptimizationLastFrame(&curr, false)
            : OptimizerGPU::PoseInertialOptimizationLastFrameFused(&curr, false);
        pose[which] = curr.GetPose();
        vel[which] = curr.GetVelocity();

        if (curr.mpcpi) { DeleteConstraintPoseImu(curr.mpcpi); curr.mpcpi = nullptr; }
        if (prev.mpcpi) { DeleteConstraintPoseImu(prev.mpcpi); prev.mpcpi = nullptr; }
        delete preint;
    }

    if (verbose) {
        std::snprintf(buf, sizeof(buf), "g2o inliers=%d  gpu inliers=%d", inliers[0],
                      inliers[1]);
        std::cout << "      " << buf << "\n";
    }

    const double g_pos = pos_m(pose[0], truth.Tcw_true);
    const double g_rot = rot_deg(pose[0], truth.Tcw_true);
    std::snprintf(buf, sizeof(buf), "g2o vs truth: %.5f m, %.4f deg", g_pos, g_rot);
    // A single frame pair barely observes the bias, so in hard mode the reference
    // is not expected to land on the truth -- only to land somewhere definite that
    // the fused kernel also lands on. Agreement is the claim under test either way.
    if (!hard) {
        report(s, g_pos < kTruthPosTolM && g_rot < kTruthRotTolDeg,
               "reference recovers pose", buf);
    } else {
        std::cout << "    --    reference vs truth (not asserted)  " << buf << "\n";
    }

    const bool observable = true;   // the prior constrains these scenes fully
    const double d_pos = pos_m(pose[1], pose[0]);
    const double d_rot = rot_deg(pose[1], pose[0]);
    std::snprintf(buf, sizeof(buf), "gpu vs g2o: %.5f m, %.4f deg", d_pos, d_rot);
    report(s, d_pos < (observable ? kPosTolM : kPosTolUnobservableM) &&
                  d_rot < kRotTolDeg, "gpu agrees with g2o", buf);

    const double dv = (vel[1] - vel[0]).cast<double>().norm();
    std::snprintf(buf, sizeof(buf), "|dv| = %.6f m/s  (g2o %.4f, gpu %.4f)", dv,
                  vel[0].norm(), vel[1].norm());
    report(s, dv < 1e-2, "gpu velocity agrees with g2o", buf);

    std::snprintf(buf, sizeof(buf), "g2o=%d gpu=%d", inliers[0], inliers[1]);
    report(s, std::abs(inliers[0] - inliers[1]) <= std::max(1, n_points / 50),
           "inlier counts agree", buf);
}


// Divergence as a function of conditioning.
//
// The stress cases show the fused kernel and g2o parting company on weakly
// observable geometry while matching to 1e-5 m on well-conditioned scenes. This
// sweeps the depth of the point cloud at a fixed narrow lateral spread, which walks
// the problem from strongly to weakly observable, and reports how far apart the two
// solvers end up at each step. It is a characterisation rather than an assertion:
// what it should show is whether the disagreement is a threshold effect or grows
// smoothly with conditioning.
void sweep_conditioning(GeometricCamera *cam, int seeds) {
    std::cout << "\nconditioning sweep: narrow cone (spread 0.6 m), varying depth\n";
    std::cout << "  depth(m)   pts   median |dt| gpu-vs-g2o   max |dt|   median |dR| deg\n";
    const double depths[] = {5, 10, 20, 40, 80, 160};
    for (double z : depths) {
        std::vector<double> dts, drs;
        for (int k = 0; k < seeds; k++) {
            std::mt19937 rng(4000 + k);
            Arena arena;
            Frame frame;
            Scene scene;
            build_scene(scene, arena, frame, cam, nullptr, 200, 0.5, 0.10, false,
                        rng, 0.6, z, z * 1.5);
            const Outcome a = run_solver(frame, scene, false);
            const Outcome b = run_solver(frame, scene, true);
            dts.push_back(pos_m(b.pose, a.pose));
            drs.push_back(rot_deg(b.pose, a.pose));
        }
        std::sort(dts.begin(), dts.end());
        std::sort(drs.begin(), drs.end());
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  %7.0f   200   %16.6f   %8.6f   %14.6f",
                      z, dts[dts.size() / 2], dts.back(), drs[drs.size() / 2]);
        std::cout << buf << "\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    int seeds = 3;
    bool verbose = false, sweep = false;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--sweep")) sweep = true;
        else if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = std::atoi(argv[++i]);
    }

    // The fused path is only taken when the controller says so; the test drives the
    // two entry points directly, but the flags gate internals further down.
    TrackingKernelController::setGPURunMode(true, true, true, true, true, true);

    // EuRoC-like pinhole intrinsics, and TUM-VI-like fisheye ones.
    Pinhole pinhole(std::vector<float>{458.654f, 457.296f, 367.215f, 248.375f});
    KannalaBrandt8 fish1(std::vector<float>{190.978f, 190.973f, 254.932f, 256.897f,
                                            0.0034f, 0.0007f, -0.0009f, 0.00003f});
    KannalaBrandt8 fish2(std::vector<float>{190.443f, 190.442f, 252.590f, 254.195f,
                                            0.0034f, 0.0007f, -0.0009f, 0.00003f});

    Stats s;
    if (sweep) {
        sweep_conditioning(&pinhole, seeds);
        return 0;
    }
    std::cout << "fused GPU pose optimizer vs g2o\n\n";
    for (int k = 0; k < seeds; k++) {
        const unsigned seed = 1000u + k;
        std::cout << "seed " << seed << "\n";
        run_case(s, "pinhole, mono obs, clean", &pinhole, nullptr, 200, 0.5, 0.00,
                 false, seed, verbose);
        run_case(s, "pinhole, mono obs, 20% outliers", &pinhole, nullptr, 200, 0.5,
                 0.20, false, seed + 1, verbose);
        run_case(s, "pinhole, stereo obs, 20% outliers", &pinhole, nullptr, 200, 0.5,
                 0.20, true, seed + 2, verbose);
        run_case(s, "fisheye (KB8), mono obs, clean", &fish1, nullptr, 200, 0.5,
                 0.00, false, seed + 3, verbose);
        run_case(s, "fisheye (KB8), mono obs, 20% outliers", &fish1, nullptr, 200,
                 0.5, 0.20, false, seed + 4, verbose);
        run_inertial_case(s, "inertial (last frame), pinhole, clean", &pinhole, 200,
                          0.5, 0.00, seed + 5, verbose);
        run_inertial_case(s, "inertial (last frame), pinhole, 20% outliers", &pinhole,
                          200, 0.5, 0.20, seed + 6, verbose);
        run_inertial_case(s, "inertial (last frame), fisheye, 20% outliers", &fish1,
                          200, 0.5, 0.20, seed + 7, verbose);
        run_inertial_case(s, "inertial HARD (rotation + bias), pinhole", &pinhole, 200,
                          0.5, 0.20, seed + 8, verbose, true);
        run_inertial_case(s, "inertial HARD (rotation + bias), fisheye", &fish1, 200,
                          0.5, 0.20, seed + 9, verbose, true);
        // The two-camera fisheye rig -- the configuration every TUM-VI sequence
        // runs, and the one the regressions cluster in.
        run_case(s, "RIG fisheye stereo (2 cams), clean", &fish1, &fish2, 200, 0.5,
                 0.00, false, seed + 10, verbose);
        run_case(s, "RIG fisheye stereo (2 cams), 20% outliers", &fish1, &fish2, 200,
                 0.5, 0.20, false, seed + 11, verbose);
        run_inertial_case(s, "RIG inertial, 2 cams, 20% outliers", &fish1, 200, 0.5,
                          0.20, seed + 12, verbose, false, &fish2);
        run_inertial_case(s, "RIG inertial HARD, 2 cams", &fish1, 200, 0.5, 0.20,
                          seed + 13, verbose, true, &fish2);
        // Ill-conditioned regimes. The fused kernel solves in single precision by
        // design (see the note on FT in PoseOptimizationFused.cu); these are the
        // shapes of problem where that choice would start to show against g2o's
        // double, and where a long outdoor sequence actually spends some of its
        // frames -- few surviving observations, distant points, little parallax.
        run_case(s, "STRESS few points (20)", &pinhole, nullptr, 20, 0.5, 0.10,
                 false, seed + 14, verbose);
        run_case(s, "STRESS far + narrow cone", &pinhole, nullptr, 200, 0.5, 0.10,
                 false, seed + 15, verbose, 0.6, 40.0, 60.0);
        run_case(s, "STRESS far + narrow, few points", &pinhole, nullptr, 25, 0.5,
                 0.10, false, seed + 16, verbose, 0.5, 45.0, 70.0);
        run_case(s, "STRESS far + narrow, fisheye rig", &fish1, &fish2, 60, 0.5,
                 0.10, false, seed + 17, verbose, 0.8, 35.0, 55.0);
        run_case(s, "STRESS high noise (3 px)", &pinhole, nullptr, 200, 3.0, 0.25,
                 false, seed + 18, verbose);
        std::cout << "\n";
    }

    std::cout << "checks " << s.checks << ", failures " << s.failures << "\n";
    return s.failures == 0 ? 0 : 1;
}
