#pragma once
#include "Optimizer.h"

#include <graphite/vector.hpp>
#include <graphite/loss.hpp>
#include "GPUTypes.h"
#include <graphite/solver/eigen.hpp>
#include <graphite/optimizer/levenberg_marquardt.hpp>
#include "G2oTypes.h"
#include "OptimizableTypes.h"



namespace ORB_SLAM3 {

namespace OptimizerGPU {


template <typename FP, typename Camera, size_t max_cameras>
static std::array<Camera*, max_cameras> get_cameras(Frame* pF, std::unordered_map<GeometricCamera*, Camera*>& cameras) {
    std::array<Camera*, max_cameras> cams = {nullptr, nullptr};
    constexpr size_t num_params = Camera::parameter_size;
    if (cameras.find(pF->mpCamera) == cameras.end()) {
        Camera* cam;
        cudaMallocManaged(&cam, sizeof(Camera));
        cudaDeviceSynchronize();
        std::array<FP, num_params> cam_params;
        for (size_t i = 0; i < cam_params.size(); i++) cam_params[i] = pF->mpCamera->getParameter(i);
        *cam = Camera(cam_params);
        cameras[pF->mpCamera] = cam;
        cams[0] = cam;
    } else {
        cams[0] = cameras[pF->mpCamera];
    }
    if (pF->mpCamera2) {
        if (cameras.find(pF->mpCamera2) == cameras.end()) {
            Camera* cam;
            cudaMallocManaged(&cam, sizeof(Camera));
            cudaDeviceSynchronize();
            std::array<FP, num_params> cam_params;
            for (size_t i = 0; i < cam_params.size(); i++) cam_params[i] = pF->mpCamera2->getParameter(i);
            *cam = Camera(cam_params);
            cameras[pF->mpCamera2] = cam;
            cams[1] = cam;
        } else {
            cams[1] = cameras[pF->mpCamera2];
        }
    }
    return cams;
}


template <typename Camera, size_t max_cameras>
int PoseOptimizationInternal(Frame *pFrame)
{
    using namespace graphite;
    using namespace gpu;

    using FP = double;
    using SP = double;
    using Pose = gpu::ImuCamPose<FP, Camera>;

    std::unordered_map<GeometricCamera*, Camera*> cameras;
    auto cams = get_cameras<FP, Camera, max_cameras>(pFrame, cameras);

    graphite::managed_vector<Pose> frame_pose(1);
    frame_pose[0] = Pose(pFrame, cams.data());
    const Pose initial_pose = frame_pose[0];

    struct MonoEntry {
        int frame_idx;
        Vec2<FP> obs;
        PoseOnlyData<FP> data;
        SP invSigma2;
    };
    struct StereoEntry {
        int frame_idx;
        Vec3<FP> obs;
        PoseOnlyData<FP> data;
        SP invSigma2;
    };

    const int N = pFrame->N;
    vector<MonoEntry>   mono_entries;
    vector<StereoEntry> stereo_entries;
    mono_entries.reserve(N);
    stereo_entries.reserve(N);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);
        for (int i = 0; i < N; i++) {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (!pMP) continue;

            pFrame->mvbOutlier[i] = false;

            PoseOnlyData<FP> data;
            data.Xw = pMP->GetWorldPos().cast<FP>();

            if (!pFrame->mpCamera2) {
                if (pFrame->mvuRight[i] < 0) {
                    const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                    data.cam_idx = 0;
                    MonoEntry e;
                    e.frame_idx = i;
                    e.obs << kpUn.pt.x, kpUn.pt.y;
                    e.data = data;
                    e.invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    mono_entries.push_back(e);
                } else {
                    const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                    data.cam_idx = 0;
                    StereoEntry e;
                    e.frame_idx = i;
                    e.obs << kpUn.pt.x, kpUn.pt.y, pFrame->mvuRight[i];
                    e.data = data;
                    e.invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    stereo_entries.push_back(e);
                }
            } else {
                cv::KeyPoint kpUn;
                if (i < pFrame->Nleft) {
                    kpUn = pFrame->mvKeys[i];
                    data.cam_idx = 0;
                } else {
                    kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];
                    data.cam_idx = 1;
                }
                MonoEntry e;
                e.frame_idx = i;
                e.obs << kpUn.pt.x, kpUn.pt.y;
                e.data = data;
                e.invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                mono_entries.push_back(e);
            }
        }
    }

    const int nInitialCorrespondences = (int)(mono_entries.size() + stereo_entries.size());
    if (nInitialCorrespondences < 3) {
        for (auto& [cam_ptr, cam] : cameras) cudaFree(cam);
        return 0;
    }

    const float chi2Mono   = 5.991f;
    const float chi2Stereo = 7.815f;
    const int its[4]       = {10, 10, 10, 10};
    const FP thHuberMono   = sqrt(FP(5.991));
    const FP thHuberStereo = sqrt(FP(7.815));

    Graph<FP, SP> graph;
    EigenLDLTSolver<FP, SP> solver;
    StreamPool streams(1);

    auto pose_desc = PoseDescriptor<FP, SP, Camera>();
    pose_desc.reserve(1);
    graph.add_vertex_descriptor(&pose_desc);
    pose_desc.add_vertex(0, &frame_pose[0], false);

    using MonoDesc   = MonoConstraintOnlyPoseDescriptor<FP, SP, HuberLoss<FP, 2>, Camera>;
    using StereoDesc = StereoConstraintOnlyPoseDescriptor<FP, SP, HuberLoss<FP, 3>, Camera>;

    MonoDesc   mono_desc(&pose_desc);
    StereoDesc stereo_desc(&pose_desc);

    vector<size_t> mono_factor_ids(mono_entries.size());
    vector<size_t> stereo_factor_ids(stereo_entries.size());

    mono_desc.reserve(mono_entries.size());
    for (size_t j = 0; j < mono_entries.size(); j++) {
        const auto& e = mono_entries[j];
        Mat2<SP> info = Mat2<SP>::Identity() * e.invSigma2;
        mono_factor_ids[j] = mono_desc.add_factor(
            {0}, e.obs, info.data(), e.data, HuberLoss<FP, 2>(thHuberMono));
    }

    stereo_desc.reserve(stereo_entries.size());
    for (size_t j = 0; j < stereo_entries.size(); j++) {
        const auto& e = stereo_entries[j];
        Mat3<SP> info = Mat3<SP>::Identity() * e.invSigma2;
        stereo_factor_ids[j] = stereo_desc.add_factor(
            {0}, e.obs, info.data(), e.data, HuberLoss<FP, 3>(thHuberStereo));
    }

    if (mono_desc.internal_count() > 0)   graph.add_factor_descriptor(&mono_desc);
    if (stereo_desc.internal_count() > 0) graph.add_factor_descriptor(&stereo_desc);

    int nBad = 0;

    for (size_t it = 0; it < 4; it++) {
        frame_pose[0] = initial_pose;

        for (size_t j = 0; j < mono_desc.internal_count(); j++)
            mono_desc.set_active(mono_factor_ids[j], pFrame->mvbOutlier[mono_entries[j].frame_idx] ? 1 : 0);
        for (size_t j = 0; j < stereo_desc.internal_count(); j++)
            stereo_desc.set_active(stereo_factor_ids[j], pFrame->mvbOutlier[stereo_entries[j].frame_idx] ? 1 : 0);

        optimizer::LevenbergMarquardtOptions<FP, SP> options;
        options.solver = &solver;
        options.iterations = its[it];
        options.initial_damping = 1e0;
        options.optimization_level = 0;
        options.streams = &streams;
        options.stop_flag = nullptr;
        options.verbose = false;

        optimizer::levenberg_marquardt2<FP, SP>(&graph, &options);

        nBad = 0;
        const Pose& pose = frame_pose[0];

        for (size_t j = 0; j < mono_desc.internal_count(); j++) {
            const auto& e = mono_entries[j];
            float chi2_val;
            if (!pFrame->mvbOutlier[e.frame_idx]) {
                chi2_val = mono_desc.chi2(mono_factor_ids[j]);
            } else {
                Vec2<FP> proj = pose.Project(e.data.Xw, e.data.cam_idx);
                chi2_val = float((e.obs - proj).squaredNorm() * e.invSigma2);
            }
            if (chi2_val > chi2Mono) {
                pFrame->mvbOutlier[e.frame_idx] = true;
                nBad++;
            } else {
                pFrame->mvbOutlier[e.frame_idx] = false;
            }
        }

        for (size_t j = 0; j < stereo_desc.internal_count(); j++) {
            const auto& e = stereo_entries[j];
            float chi2_val;
            if (!pFrame->mvbOutlier[e.frame_idx]) {
                chi2_val = stereo_desc.chi2(stereo_factor_ids[j]);
            } else {
                Vec3<FP> proj = pose.ProjectStereo(e.data.Xw, e.data.cam_idx);
                chi2_val = float((e.obs - proj).squaredNorm() * e.invSigma2);
            }
            if (chi2_val > chi2Stereo) {
                pFrame->mvbOutlier[e.frame_idx] = true;
                nBad++;
            } else {
                pFrame->mvbOutlier[e.frame_idx] = false;
            }
        }

        if (nInitialCorrespondences < 10) break;
    }

    Sophus::SE3f final_pose(
        frame_pose[0].Rcw[0].template cast<float>(),
        frame_pose[0].tcw[0].template cast<float>()
    );
    pFrame->SetPose(final_pose);

    for (auto& [cam_ptr, cam] : cameras) cudaFree(cam);

    return nInitialCorrespondences - nBad;
}



int PoseOptimization(Frame *pFrame) {
    using namespace gpu;
    if (pFrame->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        return OptimizerGPU::PoseOptimizationInternal<PinholeCamera<double>, 2>(pFrame);
    }
    else {
        return OptimizerGPU::PoseOptimizationInternal<KannalaBrandt8Camera<double>, 2>(pFrame);
    }
}


template <typename Camera, size_t max_cameras>
int PoseInertialOptimizationLastKFInternal(Frame *pFrame, bool bRecInit)
{
    using namespace graphite;
    using namespace gpu;

    using FP = double;
    using SP = double;
    using Pose = gpu::ImuCamPose<FP, Camera>;
    using PoseDesc = PoseDescriptor<FP, SP, Camera>;

    std::unordered_map<GeometricCamera*, Camera*> cameras;
    auto cams = get_cameras<FP, Camera, max_cameras>(pFrame, cameras);

    KeyFrame* pKF = pFrame->mpLastKeyFrame;

    // ---- Vertex data (managed memory, accessible from CPU+GPU) ----
    managed_vector<Pose>          frame_pose(1), kf_pose(1);
    managed_vector<Velocity<FP>>  frame_vel(1),  kf_vel(1);
    managed_vector<GyroBias<FP>>  frame_gyrobias(1), kf_gyrobias(1);
    managed_vector<AccBias<FP>>   frame_accbias(1),  kf_accbias(1);

    frame_pose[0]     = Pose(pFrame, cams.data());
    kf_pose[0]        = Pose(pKF,    cams.data());
    frame_vel[0]      = pFrame->GetVelocity().cast<FP>();
    kf_vel[0]         = pKF->GetVelocity().cast<FP>();
    frame_gyrobias[0] << pFrame->mImuBias.bwx, pFrame->mImuBias.bwy, pFrame->mImuBias.bwz;
    kf_gyrobias[0]    = pKF->GetGyroBias().cast<FP>();
    frame_accbias[0]  << pFrame->mImuBias.bax, pFrame->mImuBias.bay, pFrame->mImuBias.baz;
    kf_accbias[0]     = pKF->GetAccBias().cast<FP>();

    // ---- Correspondence building ----
    const int N      = pFrame->N;
    const int Nleft  = pFrame->Nleft;
    const bool bRight = (Nleft != -1);

    int nInitialMonoCorrespondences  = 0;
    int nInitialStereoCorrespondences = 0;

    struct MonoEntry {
        int frame_idx;
        Vec2<FP> obs;
        PoseOnlyData<FP> data;
        SP invSigma2;
        bool bClose;
    };
    struct StereoEntry {
        int frame_idx;
        Vec3<FP> obs;
        PoseOnlyData<FP> data;
        SP invSigma2;
    };

    vector<MonoEntry>   mono_entries;
    vector<StereoEntry> stereo_entries;
    mono_entries.reserve(N);
    stereo_entries.reserve(N);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);
        for (int i = 0; i < N; i++) {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (!pMP) continue;

            PoseOnlyData<FP> data;
            data.Xw = pMP->GetWorldPos().cast<FP>();

            // Left monocular observation
            if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft) {
                cv::KeyPoint kpUn = (i < Nleft) ? pFrame->mvKeys[i] : pFrame->mvKeysUn[i];
                nInitialMonoCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Vec2<FP> obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                const float unc2 = pFrame->mpCamera->uncertainty2(obs.template cast<double>());
                const SP invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;

                data.cam_idx = 0;
                mono_entries.push_back({i, obs, data, invSigma2, pMP->mTrackDepth < 10.f});
            }
            // Stereo observation
            else if (!bRight) {
                cv::KeyPoint kpUn = pFrame->mvKeysUn[i];
                nInitialStereoCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Vec2<FP> obs2d;
                obs2d << kpUn.pt.x, kpUn.pt.y;
                const float unc2 = pFrame->mpCamera->uncertainty2(obs2d.template cast<double>());
                const SP invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;

                Vec3<FP> obs;
                obs << kpUn.pt.x, kpUn.pt.y, pFrame->mvuRight[i];
                data.cam_idx = 0;
                stereo_entries.push_back({i, obs, data, invSigma2});
            }

            // Right monocular observation
            if (bRight && i >= Nleft) {
                cv::KeyPoint kpUn = pFrame->mvKeysRight[i - Nleft];
                nInitialMonoCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Vec2<FP> obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                const float unc2 = pFrame->mpCamera->uncertainty2(obs.template cast<double>());
                const SP invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;

                data.cam_idx = 1;
                mono_entries.push_back({i, obs, data, invSigma2, pMP->mTrackDepth < 10.f});
            }
        }
    }

    const int nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // ---- Graphite graph setup ----
    const FP thHuberMono   = sqrt(FP(5.991));
    const FP thHuberStereo = sqrt(FP(7.815));

    Graph<FP, SP> graph;
    EigenLDLTSolver<FP, SP> solver;
    StreamPool streams(1);

    // 4 shared vertex descriptors. Each holds 2 vertices:
    //   id=0: KF vertex (fixed), id=1: frame vertex (unfixed)
    PoseDesc pose_desc;
    VelocityDescriptor<FP, SP>  vel_desc;
    GyroBiasDescriptor<FP, SP>  gyrobias_desc;
    AccBiasDescriptor<FP, SP>   accbias_desc;

    pose_desc.reserve(2);
    vel_desc.reserve(2);
    gyrobias_desc.reserve(2);
    accbias_desc.reserve(2);

    graph.add_vertex_descriptor(&pose_desc);
    graph.add_vertex_descriptor(&vel_desc);
    graph.add_vertex_descriptor(&gyrobias_desc);
    graph.add_vertex_descriptor(&accbias_desc);

    // KF vertices (fixed)
    pose_desc.add_vertex(0, &kf_pose[0], true);
    vel_desc.add_vertex(0, &kf_vel[0], true);
    gyrobias_desc.add_vertex(0, &kf_gyrobias[0], true);
    accbias_desc.add_vertex(0, &kf_accbias[0], true);

    // Frame vertices (unfixed, optimized)
    pose_desc.add_vertex(1, &frame_pose[0], false);
    vel_desc.add_vertex(1, &frame_vel[0], false);
    gyrobias_desc.add_vertex(1, &frame_gyrobias[0], false);
    accbias_desc.add_vertex(1, &frame_accbias[0], false);

    // Visual factor descriptors (only connect to frame pose, id=1)
    using MonoDesc   = MonoConstraintOnlyPoseDescriptor<FP, SP, HuberLoss<FP, 2>, Camera>;
    using StereoDesc = StereoConstraintOnlyPoseDescriptor<FP, SP, HuberLoss<FP, 3>, Camera>;
    MonoDesc   mono_desc(&pose_desc);
    StereoDesc stereo_desc(&pose_desc);

    vector<size_t> mono_factor_ids(mono_entries.size());
    vector<size_t> stereo_factor_ids(stereo_entries.size());

    mono_desc.reserve(mono_entries.size());
    for (size_t j = 0; j < mono_entries.size(); j++) {
        const auto& e = mono_entries[j];
        Mat2<SP> info = Mat2<SP>::Identity() * e.invSigma2;
        mono_factor_ids[j] = mono_desc.add_factor({1}, e.obs, info.data(), e.data, HuberLoss<FP, 2>(thHuberMono));
    }

    stereo_desc.reserve(stereo_entries.size());
    for (size_t j = 0; j < stereo_entries.size(); j++) {
        const auto& e = stereo_entries[j];
        Mat3<SP> info = Mat3<SP>::Identity() * e.invSigma2;
        stereo_factor_ids[j] = stereo_desc.add_factor({1}, e.obs, info.data(), e.data, HuberLoss<FP, 3>(thHuberStereo));
    }

    // Inertial factor: [KF pose, KF vel, KF gyrobias, KF accbias, frame pose, frame vel]
    //   = vertex ids {0, 0, 0, 0, 1, 1} from the respective descriptors
    using InertialDesc = InertialConstraintDescriptor<FP, SP, DefaultLoss<FP, 9>, PoseDesc>;
    InertialDesc inertial_desc(&pose_desc, &vel_desc, &gyrobias_desc, &accbias_desc,
                               &pose_desc, &vel_desc);

    InertialConstraintData<FP> imu_data(pFrame->mpImuPreintegrated);
    auto Omega = imu_data.get_information_matrix<SP>(pFrame->mpImuPreintegrated);
    inertial_desc.add_factor({0, 0, 0, 0, 1, 1}, Empty(), Omega.data(), imu_data, DefaultLoss<FP, 9>());

    // Gyro and acc random walk: [KF bias (id=0), frame bias (id=1)]
    using GyroRWDesc = GyroRWConstraintDescriptor<FP, SP, DefaultLoss<FP, 3>>;
    using AccRWDesc  = AccRWConstraintDescriptor<FP, SP, DefaultLoss<FP, 3>>;
    GyroRWDesc gyrorw_desc(&gyrobias_desc, &gyrobias_desc);
    AccRWDesc  accrw_desc(&accbias_desc,  &accbias_desc);

    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    gyrorw_desc.add_factor({0, 1}, Empty(), InfoG.data(), Empty(), DefaultLoss<FP, 3>());
    accrw_desc.add_factor( {0, 1}, Empty(), InfoA.data(), Empty(), DefaultLoss<FP, 3>());

    if (mono_desc.internal_count() > 0)   graph.add_factor_descriptor(&mono_desc);
    if (stereo_desc.internal_count() > 0) graph.add_factor_descriptor(&stereo_desc);
    graph.add_factor_descriptor(&inertial_desc);
    graph.add_factor_descriptor(&gyrorw_desc);
    graph.add_factor_descriptor(&accrw_desc);

    // ---- Optimization loop (4 iterations, decreasing chi2 thresholds) ----
    const float chi2Mono[4]   = {12.f, 7.5f, 5.991f, 5.991f};
    const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
    const int   its[4]        = {10, 10, 10, 10};

    int nBad = 0, nInliers = 0;
    int nInliersMono = 0, nInliersStereo = 0;

    for (size_t it = 0; it < 4; it++) {
        // Activate/deactivate visual factors based on current outlier status
        for (size_t j = 0; j < mono_desc.internal_count(); j++)
            mono_desc.set_active(mono_factor_ids[j], pFrame->mvbOutlier[mono_entries[j].frame_idx] ? 1 : 0);
        for (size_t j = 0; j < stereo_desc.internal_count(); j++)
            stereo_desc.set_active(stereo_factor_ids[j], pFrame->mvbOutlier[stereo_entries[j].frame_idx] ? 1 : 0);

        optimizer::LevenbergMarquardtOptions<FP, SP> options;
        options.solver           = &solver;
        options.iterations       = its[it];
        options.initial_damping  = 1e0;
        options.optimization_level = 0;
        options.streams          = &streams;
        options.stop_flag        = nullptr;
        options.verbose          = false;
        optimizer::levenberg_marquardt2<FP, SP>(&graph, &options);

        nBad = 0;
        nInliers = nInliersMono = nInliersStereo = 0;
        const float chi2close = 1.5f * chi2Mono[it];
        const Pose& pose = frame_pose[0];

        for (size_t j = 0; j < mono_desc.internal_count(); j++) {
            const auto& e = mono_entries[j];
            float chi2_val;
            if (!pFrame->mvbOutlier[e.frame_idx]) {
                chi2_val = mono_desc.chi2(mono_factor_ids[j]);
            } else {
                Vec2<FP> proj = pose.Project(e.data.Xw, e.data.cam_idx);
                chi2_val = float((e.obs - proj).squaredNorm() * e.invSigma2);
            }
            bool is_outlier = (e.bClose ? chi2_val > chi2close : chi2_val > chi2Mono[it])
                              || !pose.isDepthPositive(e.data.Xw, e.data.cam_idx);
            if (is_outlier) {
                pFrame->mvbOutlier[e.frame_idx] = true;
                nBad++;
            } else {
                pFrame->mvbOutlier[e.frame_idx] = false;
                nInliersMono++;
            }
        }

        for (size_t j = 0; j < stereo_desc.internal_count(); j++) {
            const auto& e = stereo_entries[j];
            float chi2_val;
            if (!pFrame->mvbOutlier[e.frame_idx]) {
                chi2_val = stereo_desc.chi2(stereo_factor_ids[j]);
            } else {
                Vec3<FP> proj = pose.ProjectStereo(e.data.Xw, e.data.cam_idx);
                chi2_val = float((e.obs - proj).squaredNorm() * e.invSigma2);
            }
            if (chi2_val > chi2Stereo[it]) {
                pFrame->mvbOutlier[e.frame_idx] = true;
                nBad++;
            } else {
                pFrame->mvbOutlier[e.frame_idx] = false;
                nInliersStereo++;
            }
        }

        nInliers = nInliersMono + nInliersStereo;

        // +3 for the always-active inertial, gyroRW, and accRW factors
        if (nInliers + 3 < 10) break;
    }

    // ---- Recovery: relax thresholds if too few inliers ----
    if (nInliers < 30 && !bRecInit) {
        nBad = 0;
        const float chi2MonoOut   = 18.f;
        const float chi2StereoOut = 24.f;
        const Pose& pose = frame_pose[0];
        for (size_t j = 0; j < mono_entries.size(); j++) {
            const auto& e = mono_entries[j];
            Vec2<FP> proj = pose.Project(e.data.Xw, e.data.cam_idx);
            float chi2_val = float((e.obs - proj).squaredNorm() * e.invSigma2);
            if (chi2_val < chi2MonoOut)
                pFrame->mvbOutlier[e.frame_idx] = false;
            else
                nBad++;
        }
        for (size_t j = 0; j < stereo_entries.size(); j++) {
            const auto& e = stereo_entries[j];
            Vec3<FP> proj = pose.ProjectStereo(e.data.Xw, e.data.cam_idx);
            float chi2_val = float((e.obs - proj).squaredNorm() * e.invSigma2);
            if (chi2_val < chi2StereoOut)
                pFrame->mvbOutlier[e.frame_idx] = false;
            else
                nBad++;
        }
    }

    // ---- Extract optimized pose, velocity, biases ----
    cudaDeviceSynchronize();

    pFrame->SetImuPoseVelocity(
        frame_pose[0].Rwb.template cast<float>(),
        frame_pose[0].twb.template cast<float>(),
        frame_vel[0].template cast<float>());
    pFrame->mImuBias = IMU::Bias(
        frame_accbias[0][0],  frame_accbias[0][1],  frame_accbias[0][2],
        frame_gyrobias[0][0], frame_gyrobias[0][1], frame_gyrobias[0][2]);

    // ---- Hessian for marginalization prior (mpcpi) ----
    // Evaluate Jacobians on CPU at the final optimized state using the hd_fn
    // static methods from GPUTypes.h, then accumulate H = J^T * Omega * J.
    using InerConstraint   = InertialConstraint<FP, SP, DefaultLoss<FP, 9>, PoseDesc>;
    using MonoConstraint   = MonoConstraintOnlyPose<FP, SP, HuberLoss<FP, 2>, Camera>;
    using StereoConstraint = StereoConstraintOnlyPose<FP, SP, HuberLoss<FP, 3>, Camera>;

    Eigen::Matrix<double, 15, 15> H;
    H.setZero();

    // 1. Inertial contribution — H[0:9, 0:9] from [J4|J5]^T * Omega * [J4|J5]
    {
        InertialConstraintData<FP> cpu_imu(pFrame->mpImuPreintegrated);
        Eigen::Matrix<double, 9, 9> Omega9 =
            cpu_imu.get_information_matrix<double>(pFrame->mpImuPreintegrated);

        double J4[9 * 6], J5[9 * 3];
        InerConstraint::template jacobian<double, 4>(
            kf_pose[0], kf_vel[0], kf_gyrobias[0], kf_accbias[0],
            frame_pose[0], frame_vel[0], cpu_imu, J4);
        InerConstraint::template jacobian<double, 5>(
            kf_pose[0], kf_vel[0], kf_gyrobias[0], kf_accbias[0],
            frame_pose[0], frame_vel[0], cpu_imu, J5);

        Eigen::Map<const Eigen::Matrix<double, 9, 6, Eigen::ColMajor>> J4m(J4);
        Eigen::Map<const Eigen::Matrix<double, 9, 3, Eigen::ColMajor>> J5m(J5);
        Eigen::Matrix<double, 9, 9> Jstack;
        Jstack.block<9, 6>(0, 0) = J4m;
        Jstack.block<9, 3>(0, 6) = J5m;
        H.block<9, 9>(0, 0) += Jstack.transpose() * Omega9 * Jstack;
    }

    // 2. GyroRW — H[9:12, 9:12] += InfoG  (second vertex Jacobian = I)
    H.block<3, 3>(9, 9) += InfoG;

    // 3. AccRW — H[12:15, 12:15] += InfoA  (second vertex Jacobian = I)
    H.block<3, 3>(12, 12) += InfoA;

    // 4. Visual inliers — H[0:6, 0:6] += J^T * invSigma2 * J
    for (size_t j = 0; j < mono_entries.size(); j++) {
        if (!pFrame->mvbOutlier[mono_entries[j].frame_idx]) {
            const auto& e = mono_entries[j];
            double Jm[2 * 6];
            MonoConstraint::template jacobian<double, 0>(frame_pose[0], e.obs, e.data, Jm);
            Eigen::Map<const Eigen::Matrix<double, 2, 6, Eigen::ColMajor>> J(Jm);
            H.block<6, 6>(0, 0) += J.transpose() * double(e.invSigma2) * J;
        }
    }
    for (size_t j = 0; j < stereo_entries.size(); j++) {
        if (!pFrame->mvbOutlier[stereo_entries[j].frame_idx]) {
            const auto& e = stereo_entries[j];
            double Jm[3 * 6];
            StereoConstraint::template jacobian<double, 0>(frame_pose[0], e.obs, e.data, Jm);
            Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::ColMajor>> J(Jm);
            H.block<6, 6>(0, 0) += J.transpose() * double(e.invSigma2) * J;
        }
    }

    pFrame->mpcpi = new ConstraintPoseImu(
        frame_pose[0].Rwb.template cast<double>(),
        frame_pose[0].twb.template cast<double>(),
        frame_vel[0].template cast<double>(),
        frame_gyrobias[0].template cast<double>(),
        frame_accbias[0].template cast<double>(),
        H);

    for (auto& [cam_ptr, cam] : cameras) cudaFree(cam);
    return nInitialCorrespondences - nBad;
}


int PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit) {
    using namespace gpu;
    if (pFrame->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        return OptimizerGPU::PoseInertialOptimizationLastKFInternal<PinholeCamera<double>, 2>(pFrame, bRecInit);
    } else {
        return OptimizerGPU::PoseInertialOptimizationLastKFInternal<KannalaBrandt8Camera<double>, 2>(pFrame, bRecInit);
    }
}

}

}
