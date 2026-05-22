#pragma once

// This file contains types used for 4DoF pose graph optimization in Graphite
#include <graphite/optimizer/levenberg_marquardt.hpp>
#include <graphite/types.hpp>
#include <graphite/utils.hpp>

#include "GPUPose.h"

namespace gpu {

    using namespace graphite;


    template <typename T> struct Pose4DoFTraits {
        static constexpr size_t dimension = 4;
        // using Vertex = Pose4DoF<T>;
        using Vertex = gpu::ImuCamPose<T, gpu::PinholeCamera<T>>;

        template <typename P>
        hd_fn static void parameters(const Vertex &vertex, P* params) {
            Eigen::Map<Eigen::Matrix<P, 4, 1>> p(params);

            const auto tvec = vertex.twb;
            const auto rvec = LogSO3(vertex.DR);
            p(0) = rvec(2);
            p(1) = tvec(0);
            p(2) = tvec(1);
            p(3) = tvec(2);
        }

        hd_fn static void update(Vertex &vertex, const T *delta) {

            Eigen::Matrix<T, 6, 1> update;

            // rotation part
            update(0) = 0.0;
            update(1) = 0.0;
            update(2) = delta[0];

            // translation part
            update(3) = delta[1];
            update(4) = delta[2];
            update(5) = delta[3];

            vertex.UpdateW(update.data());
        }
    };

    template <typename T, typename S>
    using Pose4DoFDescriptor = VertexDescriptor<T, S, Pose4DoFTraits<T>>;








template <typename T, typename S, typename L> struct Factor4DoFTraits {
    static constexpr size_t dimension = 6; // must be 6 because the residual still relies on entire pose
    using VertexDescriptors = std::tuple<Pose4DoFDescriptor<T, S>, Pose4DoFDescriptor<T, S>>;
    using Observation = Sophus::SE3<T>;
    using Data = Empty;
    using Loss = L;
    using Differentiation = DifferentiationMode::Auto;

    using Pose = typename Pose4DoFDescriptor<T, S>::VertexType;

    template <typename D>
    hd_fn static void get_cam_pose(const D* p, const Pose& pose, 
            Mat3<D> &Rcw, Vec3<D> &tcw) {
        const auto tr = LogSO3(pose.DR);

        Vec3<D> rvec;
        rvec << D(tr(0)), D(tr(1)), p[0];
        // rvec << D(0), D(0), p[0];
        const Mat3<D> DR = ExpSO3(rvec);
        Vec3<D> twb;
        twb << p[1], p[2], p[3];

        const Mat3<D> Rwb = DR * pose.Rwb0.template cast<D>();

        // no need to normalize

        // Compute world to body
        const Mat3<D> Rbw = Rwb.transpose();
        const Vec3<D> tbw = -Rbw * twb;

        // Compute camera pose (cam 0 / left cam)
        const auto Rcb = pose.Rcb[0].template cast<D>();
        const auto tcb = pose.tcb[0].template cast<D>();
        
        Rcw = Rcb * Rbw;
        tcw = Rcb * tbw + tcb;

    }


    template <typename D>
    hd_fn static void
    error(const Pose& pose1, const Pose& pose2, const D *p1, const D *p2, const Observation &obs, D *error) {
        
        Mat3<D> Rcwi;
        Vec3<D> tcwi;
        Mat3<D> Rcwj;
        Vec3<D> tcwj;
        
        get_cam_pose(p1, pose1, Rcwi, tcwi);
        get_cam_pose(p2, pose2, Rcwj, tcwj);


        Sophus::SE3<D> delta = obs.template cast<D>();

        const auto Rij = delta.rotationMatrix();
        const auto dtij = delta.translation();
        Eigen::Map<Eigen::Matrix<D, 6, 1>> residual(error);


        Eigen::Matrix<D, 3, 3> rR = Rcwi*Rcwj.transpose()*Rij.transpose();
        residual << LogSO3(rR), Rcwi*(-Rcwj.transpose()*tcwj)+tcwi - dtij;
                 

    }



    };

    template <typename T, typename S, typename L>
    using Factor4DoFDescriptor = FactorDescriptor<T, S, Factor4DoFTraits<T, S, L>>;

}