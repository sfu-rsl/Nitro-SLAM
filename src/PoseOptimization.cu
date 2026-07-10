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
    StreamPool streams(4);

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
        // options.initial_damping = 1e0;
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

    auto tpre0 = std::chrono::steady_clock::now();

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

    auto tpre1 = std::chrono::steady_clock::now();
    std::cout << "PO Pre-Setup (vertices+correspondences) took " << std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(tpre1 - tpre0).count() << " ms" << std::endl;

    // ---- Graphite graph setup ----
    auto tsetup0 = std::chrono::steady_clock::now();

    const FP thHuberMono   = sqrt(FP(5.991));
    const FP thHuberStereo = sqrt(FP(7.815));

    Graph<FP, SP> graph;
    EigenLDLTSolver<FP, SP> solver;
    StreamPool streams(4);

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

    auto tsetup1 = std::chrono::steady_clock::now();
    std::cout << "PO Graph Setup took " << std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(tsetup1 - tsetup0).count() << " ms" << std::endl;

    // ---- Optimization loop (4 iterations, decreasing chi2 thresholds) ----
    auto topt0 = std::chrono::steady_clock::now();

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
        // options.initial_damping  = 1e0;
        options.optimization_level = 0;
        options.streams          = &streams;
        options.stop_flag        = nullptr;
        options.verbose          = false;

        auto tg0 = std::chrono::steady_clock::now();
        optimizer::levenberg_marquardt2<FP, SP>(&graph, &options);
        auto tg1 = std::chrono::steady_clock::now();
        std::cout << "PO lm took " << std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(tg1 - tg0).count() << " ms" << std::endl;

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

    auto topt1 = std::chrono::steady_clock::now();
    std::cout << "PO Graph Optimization took " << std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(topt1 - topt0).count() << " ms" << std::endl;

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

    auto th0 =  std::chrono::steady_clock::now();
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
        auto th1 =  std::chrono::steady_clock::now();

         std::cout << "PO Hessian took " <<   std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(th1 - th0).count() << " ms" << std::endl;


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


template <typename Camera, size_t max_cameras>
int PoseInertialOptimizationLastFInternal(Frame *pFrame, bool bRecInit)
{
    using namespace graphite;
    using namespace gpu;

    using FP = double;
    using SP = double;
    using Pose = gpu::ImuCamPose<FP, Camera>;
    using PoseDesc = PoseDescriptor<FP, SP, Camera>;

    std::unordered_map<GeometricCamera*, Camera*> cameras;
    auto cams = get_cameras<FP, Camera, max_cameras>(pFrame, cameras);

    Frame* pFp = pFrame->mpPrevFrame;

    // ---- Vertex data ----
    managed_vector<Pose>          curr_pose(1), pf_pose(1);
    managed_vector<Velocity<FP>>  curr_vel(1),  pf_vel(1);
    managed_vector<GyroBias<FP>>  curr_gyrobias(1), pf_gyrobias(1);
    managed_vector<AccBias<FP>>   curr_accbias(1),  pf_accbias(1);

    curr_pose[0]     = Pose(pFrame, cams.data());
    pf_pose[0]       = Pose(pFp,    cams.data());
    curr_vel[0]      = pFrame->GetVelocity().cast<FP>();
    pf_vel[0]        = pFp->GetVelocity().cast<FP>();
    curr_gyrobias[0] << pFrame->mImuBias.bwx, pFrame->mImuBias.bwy, pFrame->mImuBias.bwz;
    pf_gyrobias[0]   << pFp->mImuBias.bwx, pFp->mImuBias.bwy, pFp->mImuBias.bwz;
    curr_accbias[0]  << pFrame->mImuBias.bax, pFrame->mImuBias.bay, pFrame->mImuBias.baz;
    pf_accbias[0]    << pFp->mImuBias.bax, pFp->mImuBias.bay, pFp->mImuBias.baz;

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
    StreamPool streams(4);

    // 4 shared vertex descriptors. id=0: prev frame (unfixed), id=1: curr frame (unfixed)
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

    // Prev frame vertices (id=0, unfixed — both frames are optimized)
    pose_desc.add_vertex(0, &pf_pose[0], false);
    vel_desc.add_vertex(0, &pf_vel[0], false);
    gyrobias_desc.add_vertex(0, &pf_gyrobias[0], false);
    accbias_desc.add_vertex(0, &pf_accbias[0], false);

    // Current frame vertices (id=1, unfixed)
    pose_desc.add_vertex(1, &curr_pose[0], false);
    vel_desc.add_vertex(1, &curr_vel[0], false);
    gyrobias_desc.add_vertex(1, &curr_gyrobias[0], false);
    accbias_desc.add_vertex(1, &curr_accbias[0], false);

    // Visual factors connect to current frame pose (id=1)
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

    // Inertial factor: [prev pose, prev vel, prev gyrobias, prev accbias, curr pose, curr vel]
    using InertialDesc = InertialConstraintDescriptor<FP, SP, DefaultLoss<FP, 9>, PoseDesc>;
    InertialDesc inertial_desc(&pose_desc, &vel_desc, &gyrobias_desc, &accbias_desc,
                               &pose_desc, &vel_desc);

    InertialConstraintData<FP> imu_data(pFrame->mpImuPreintegratedFrame);
    auto Omega = imu_data.get_information_matrix<SP>(pFrame->mpImuPreintegratedFrame);
    inertial_desc.add_factor({0, 0, 0, 0, 1, 1}, Empty(), Omega.data(), imu_data, DefaultLoss<FP, 9>());

    // Gyro/acc random walk
    using GyroRWDesc = GyroRWConstraintDescriptor<FP, SP, DefaultLoss<FP, 3>>;
    using AccRWDesc  = AccRWConstraintDescriptor<FP, SP, DefaultLoss<FP, 3>>;
    GyroRWDesc gyrorw_desc(&gyrobias_desc, &gyrobias_desc);
    AccRWDesc  accrw_desc(&accbias_desc,  &accbias_desc);

    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    gyrorw_desc.add_factor({0, 1}, Empty(), InfoG.data(), Empty(), DefaultLoss<FP, 3>());
    accrw_desc.add_factor( {0, 1}, Empty(), InfoA.data(), Empty(), DefaultLoss<FP, 3>());

    // Prior on prev frame: EdgePriorPoseImu equivalent with Huber kernel (delta=5)
    using ImuPriorDesc = ImuPriorConstraintDescriptor<FP, SP, HuberLoss<FP, 15>, PoseDesc>;
    ImuPriorDesc prior_desc(&pose_desc, &vel_desc, &gyrobias_desc, &accbias_desc);

    if (!pFp->mpcpi)
        Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);

    typename ImuPriorConstraint<FP, SP, HuberLoss<FP, 15>, PoseDesc>::Data prior_data;
    prior_data.Rwb = pFp->mpcpi->Rwb.cast<FP>();
    prior_data.twb = pFp->mpcpi->twb.cast<FP>();
    prior_data.vwb = pFp->mpcpi->vwb.cast<FP>();
    prior_data.bg  = pFp->mpcpi->bg.cast<FP>();
    prior_data.ba  = pFp->mpcpi->ba.cast<FP>();
    Eigen::Matrix<SP, 15, 15> H_prior = pFp->mpcpi->H.cast<SP>();
    prior_desc.add_factor({0, 0, 0, 0}, Empty(), H_prior.data(), prior_data, HuberLoss<FP, 15>(FP(5)));

    if (mono_desc.internal_count() > 0)   graph.add_factor_descriptor(&mono_desc);
    if (stereo_desc.internal_count() > 0) graph.add_factor_descriptor(&stereo_desc);
    graph.add_factor_descriptor(&inertial_desc);
    graph.add_factor_descriptor(&gyrorw_desc);
    graph.add_factor_descriptor(&accrw_desc);
    graph.add_factor_descriptor(&prior_desc);

    // ---- Optimization loop ----
    // chi2Mono is constant at 5.991 for all 4 iterations (unlike LastKF)
    const float chi2Mono[4]   = {5.991f, 5.991f, 5.991f, 5.991f};
    const float chi2Stereo[4] = {15.6f,  9.8f,   7.815f, 7.815f};
    const int   its[4]        = {10, 10, 10, 10};

    int nBad = 0, nInliers = 0;
    int nInliersMono = 0, nInliersStereo = 0;

    for (size_t it = 0; it < 4; it++) {
        for (size_t j = 0; j < mono_desc.internal_count(); j++)
            mono_desc.set_active(mono_factor_ids[j], pFrame->mvbOutlier[mono_entries[j].frame_idx] ? 1 : 0);
        for (size_t j = 0; j < stereo_desc.internal_count(); j++)
            stereo_desc.set_active(stereo_factor_ids[j], pFrame->mvbOutlier[stereo_entries[j].frame_idx] ? 1 : 0);

        optimizer::LevenbergMarquardtOptions<FP, SP> options;
        options.solver           = &solver;
        options.iterations       = its[it];
        // options.initial_damping  = 1e0;
        options.optimization_level = 0;
        options.streams          = &streams;
        options.stop_flag        = nullptr;
        options.verbose          = false;
        optimizer::levenberg_marquardt2<FP, SP>(&graph, &options);

        nBad = 0;
        nInliers = nInliersMono = nInliersStereo = 0;
        const float chi2close = 1.5f * chi2Mono[it];
        const Pose& pose = curr_pose[0];

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

        // +4 for always-active inertial, gyroRW, accRW, prior factors
        if (nInliers + 4 < 10) break;
    }

    // ---- Recovery: relax thresholds if too few inliers ----
    if (nInliers < 30 && !bRecInit) {
        nBad = 0;
        const float chi2MonoOut   = 18.f;
        const float chi2StereoOut = 24.f;
        const Pose& pose = curr_pose[0];
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

    // ---- Extract optimized state ----

    pFrame->SetImuPoseVelocity(
        curr_pose[0].Rwb.template cast<float>(),
        curr_pose[0].twb.template cast<float>(),
        curr_vel[0].template cast<float>());
    pFrame->mImuBias = IMU::Bias(
        curr_accbias[0][0], curr_accbias[0][1], curr_accbias[0][2],
        curr_gyrobias[0][0], curr_gyrobias[0][1], curr_gyrobias[0][2]);

    // ---- 30x30 Hessian for marginalization prior (mpcpi) ----
    // Layout: [0:6]=prev_pose, [6:9]=prev_vel, [9:12]=prev_gyrobias, [12:15]=prev_accbias,
    //         [15:21]=curr_pose, [21:24]=curr_vel, [24:27]=curr_gyrobias, [27:30]=curr_accbias
    using InerConstraint   = InertialConstraint<FP, SP, DefaultLoss<FP, 9>, PoseDesc>;
    using ImuPrior         = ImuPriorConstraint<FP, SP, HuberLoss<FP, 15>, PoseDesc>;
    using MonoConstraint   = MonoConstraintOnlyPose<FP, SP, HuberLoss<FP, 2>, Camera>;
    using StereoConstraint = StereoConstraintOnlyPose<FP, SP, HuberLoss<FP, 3>, Camera>;

    Eigen::Matrix<double, 30, 30> H30;
    H30.setZero();

    // 1. Inertial edge — J[9x24] spanning [0:24, 0:24]
    {
        InertialConstraintData<FP> cpu_imu(pFrame->mpImuPreintegratedFrame);
        Eigen::Matrix<double, 9, 9> Omega9 =
            cpu_imu.get_information_matrix<double>(pFrame->mpImuPreintegratedFrame);

        // Jacobians for all 6 slots mapped into a 9x24 block Jacobian
        Eigen::Matrix<double, 9, 24> Jfull;
        Jfull.setZero();
        {
            double J0[9*6];
            InerConstraint::template jacobian<double, 0>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0],
                curr_pose[0], curr_vel[0], cpu_imu, J0);
            Jfull.block<9,6>(0,0) = Eigen::Map<const Eigen::Matrix<double,9,6>>(J0);
        }
        {
            double J1[9*3];
            InerConstraint::template jacobian<double, 1>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0],
                curr_pose[0], curr_vel[0], cpu_imu, J1);
            Jfull.block<9,3>(0,6) = Eigen::Map<const Eigen::Matrix<double,9,3>>(J1);
        }
        {
            double J2[9*3];
            InerConstraint::template jacobian<double, 2>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0],
                curr_pose[0], curr_vel[0], cpu_imu, J2);
            Jfull.block<9,3>(0,9) = Eigen::Map<const Eigen::Matrix<double,9,3>>(J2);
        }
        {
            double J3[9*3];
            InerConstraint::template jacobian<double, 3>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0],
                curr_pose[0], curr_vel[0], cpu_imu, J3);
            Jfull.block<9,3>(0,12) = Eigen::Map<const Eigen::Matrix<double,9,3>>(J3);
        }
        {
            double J4[9*6];
            InerConstraint::template jacobian<double, 4>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0],
                curr_pose[0], curr_vel[0], cpu_imu, J4);
            Jfull.block<9,6>(0,15) = Eigen::Map<const Eigen::Matrix<double,9,6>>(J4);
        }
        {
            double J5[9*3];
            InerConstraint::template jacobian<double, 5>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0],
                curr_pose[0], curr_vel[0], cpu_imu, J5);
            Jfull.block<9,3>(0,21) = Eigen::Map<const Eigen::Matrix<double,9,3>>(J5);
        }
        H30.block<24,24>(0,0) += Jfull.transpose() * Omega9 * Jfull;
    }

    // 2. GyroRW — cross-terms between prev [9:12] and curr [24:27]
    H30.block<3,3>(9,9)   += InfoG;
    H30.block<3,3>(9,24)  -= InfoG;
    H30.block<3,3>(24,9)  -= InfoG;
    H30.block<3,3>(24,24) += InfoG;

    // 3. AccRW — cross-terms between prev [12:15] and curr [27:30]
    H30.block<3,3>(12,12) += InfoA;
    H30.block<3,3>(12,27) -= InfoA;
    H30.block<3,3>(27,12) -= InfoA;
    H30.block<3,3>(27,27) += InfoA;

    // 4. Prior on prev frame — J[15x15] at H30[0:15, 0:15]
    {
        Eigen::Matrix<double, 15, 15> Jp;
        Jp.setZero();
        {
            double J0[15*6];
            ImuPrior::template jacobian<double, 0>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0], prior_data, J0);
            Jp.block<15,6>(0,0) = Eigen::Map<const Eigen::Matrix<double,15,6>>(J0);
        }
        {
            double J1[15*3];
            ImuPrior::template jacobian<double, 1>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0], prior_data, J1);
            Jp.block<15,3>(0,6) = Eigen::Map<const Eigen::Matrix<double,15,3>>(J1);
        }
        {
            double J2[15*3];
            ImuPrior::template jacobian<double, 2>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0], prior_data, J2);
            Jp.block<15,3>(0,9) = Eigen::Map<const Eigen::Matrix<double,15,3>>(J2);
        }
        {
            double J3[15*3];
            ImuPrior::template jacobian<double, 3>(
                pf_pose[0], pf_vel[0], pf_gyrobias[0], pf_accbias[0], prior_data, J3);
            Jp.block<15,3>(0,12) = Eigen::Map<const Eigen::Matrix<double,15,3>>(J3);
        }
        Eigen::Matrix<double,15,15> H_prior_d = pFp->mpcpi->H;
        H30.block<15,15>(0,0) += Jp.transpose() * H_prior_d * Jp;
    }

    // 5. Visual inliers — J[2x6] or J[3x6] wrt curr pose at H30[15:21, 15:21]
    for (size_t j = 0; j < mono_entries.size(); j++) {
        if (!pFrame->mvbOutlier[mono_entries[j].frame_idx]) {
            const auto& e = mono_entries[j];
            double Jm[2 * 6];
            MonoConstraint::template jacobian<double, 0>(curr_pose[0], e.obs, e.data, Jm);
            Eigen::Map<const Eigen::Matrix<double, 2, 6>> J(Jm);
            H30.block<6,6>(15,15) += J.transpose() * double(e.invSigma2) * J;
        }
    }
    for (size_t j = 0; j < stereo_entries.size(); j++) {
        if (!pFrame->mvbOutlier[stereo_entries[j].frame_idx]) {
            const auto& e = stereo_entries[j];
            double Js[3 * 6];
            StereoConstraint::template jacobian<double, 0>(curr_pose[0], e.obs, e.data, Js);
            Eigen::Map<const Eigen::Matrix<double, 3, 6>> J(Js);
            H30.block<6,6>(15,15) += J.transpose() * double(e.invSigma2) * J;
        }
    }

    // Marginalize prev frame states (0..14) → 15x15 prior for current frame
    H30 = Optimizer::Marginalize(H30, 0, 14);

    pFrame->mpcpi = new ConstraintPoseImu(
        curr_pose[0].Rwb.template cast<double>(),
        curr_pose[0].twb.template cast<double>(),
        curr_vel[0].template cast<double>(),
        curr_gyrobias[0].template cast<double>(),
        curr_accbias[0].template cast<double>(),
        H30.block<15,15>(15,15));

    delete pFp->mpcpi;
    pFp->mpcpi = nullptr;

    for (auto& [cam_ptr, cam] : cameras) cudaFree(cam);
    return nInitialCorrespondences - nBad;
}


int PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit) {
    using namespace gpu;
    if (pFrame->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        return OptimizerGPU::PoseInertialOptimizationLastFInternal<PinholeCamera<double>, 2>(pFrame, bRecInit);
    } else {
        return OptimizerGPU::PoseInertialOptimizationLastFInternal<KannalaBrandt8Camera<double>, 2>(pFrame, bRecInit);
    }
}

}

}
