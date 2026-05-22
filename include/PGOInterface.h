#pragma once
#include "Thirdparty/Sophus/sophus/geometry.hpp"

namespace ORB_SLAM3{

    class PoseGraphOptimizer;

    class SE3Pose {
        public:

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
    };

    class PoseGraphOptimizerInterface {
        public:

        PoseGraphOptimizerInterface(const unsigned int max_poses, const unsigned int max_edges);
        ~PoseGraphOptimizerInterface();

        void add_pose(const int id, ORB_SLAM3::KeyFrame* pKF);
        void add_pose(const int id, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, ORB_SLAM3::KeyFrame* pKF);
        SE3Pose get_pose(const int id);
        void set_fixed(const int id, const bool fixed);
        void add_factor(const int id1, const int id2, const Sophus::SE3d &relative_pose, const double* info);
        void optimize(const size_t iterations, const double lambda, const bool verbose);

        void clear();
        void reserve(const unsigned int max_poses, const unsigned int max_edges);

        private:

        PoseGraphOptimizer* pgo;

    };

}