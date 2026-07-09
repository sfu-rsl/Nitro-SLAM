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

}

}
