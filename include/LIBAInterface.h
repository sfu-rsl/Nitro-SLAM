#pragma once

// Interface to the Graphite local inertial BA implementation. The optimizer
// behind this interface is persistent, so its device/managed allocations are
// reused across calls (see LIBA.cu).

#include <Eigen/Core>
#include <cstddef>

namespace ORB_SLAM3{

    class KeyFrame;

    namespace IMU { class Preintegrated; }

    class LocalInertialBAOptimizer;

    // World to camera pose of the left camera
    class LIBASE3Pose {
        public:

        Eigen::Matrix3d Rcw;
        Eigen::Vector3d tcw;
    };

    class LocalInertialBAInterface {
        public:

        // Camera model of the vertices in the graph. The optimizer is templated on
        // the camera model, so a separate instance is kept for each model.
        enum class CameraModel { Pinhole, KannalaBrandt8 };

        LocalInertialBAInterface(const unsigned int max_keyframes, const unsigned int max_map_points,
                                 const unsigned int max_visual_factors);
        ~LocalInertialBAInterface();

        // Selects the optimizer to use. The instance is created on first use and kept
        // afterwards so that its allocations can be reused.
        void set_camera_model(const CameraModel model);

        void clear();
        void reserve(const unsigned int max_keyframes, const unsigned int max_map_points,
                     const unsigned int max_visual_factors);

        // Vertices
        void add_pose(const size_t id, KeyFrame* pKF, const bool fixed);
        void add_velocity(const size_t id, KeyFrame* pKF, const bool fixed);
        void add_gyro_bias(const size_t id, KeyFrame* pKF, const bool fixed);
        void add_acc_bias(const size_t id, KeyFrame* pKF, const bool fixed);
        void add_map_point(const size_t id, const Eigen::Vector3d &position);

        bool has_pose(const size_t id);
        bool has_velocity(const size_t id);
        bool has_gyro_bias(const size_t id);
        bool has_acc_bias(const size_t id);

        // Factors
        void add_inertial_factor(const size_t pose1, const size_t velocity1, const size_t gyro_bias1,
                                 const size_t acc_bias1, const size_t pose2, const size_t velocity2,
                                 IMU::Preintegrated* pInt, const bool robust, const double info_scale,
                                 const double huber_delta);
        void add_gyro_rw_factor(const size_t gyro_bias1, const size_t gyro_bias2, const double* info);
        void add_acc_rw_factor(const size_t acc_bias1, const size_t acc_bias2, const double* info);
        size_t add_mono_factor(const size_t map_point, const size_t pose, const Eigen::Vector2d &obs,
                               const double information, const int cam_idx, const double huber_delta);
        size_t add_stereo_factor(const size_t map_point, const size_t pose, const Eigen::Vector3d &obs,
                                 const double information, const int cam_idx, const double huber_delta);

        void optimize(const size_t iterations, const double lambda, bool* stop_flag,
                      double &initial_error, double &final_error, const bool verbose);

        // Results
        double mono_chi2(const size_t factor_id);
        double stereo_chi2(const size_t factor_id);
        bool mono_depth_positive(const size_t factor_id);

        LIBASE3Pose get_pose(const size_t id);
        Eigen::Vector3d get_velocity(const size_t id);
        Eigen::Vector3d get_gyro_bias(const size_t id);
        Eigen::Vector3d get_acc_bias(const size_t id);
        Eigen::Vector3d get_map_point(const size_t id);

        private:

        LocalInertialBAOptimizer* liba;             // active optimizer
        LocalInertialBAOptimizer* liba_pinhole;
        LocalInertialBAOptimizer* liba_fisheye;

        unsigned int max_keyframes;
        unsigned int max_map_points;
        unsigned int max_visual_factors;

    };

}
