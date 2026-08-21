#include "Optimizer.h"
#include "LIBAInterface.h"
#include "GPUTypes.h"
#include "Kernels/CudaUtils.h"

#include <graphite/vector.hpp>
#include <graphite/loss.hpp>
#include <graphite/solver/eigen_schur.hpp>
#include <graphite/optimizer/levenberg_marquardt.hpp>

#include <unordered_map>

namespace ORB_SLAM3 {

// Camera model independent interface to the optimizer. The interface class holds one
// instance per camera model, so that the allocations of both can be reused.
class LocalInertialBAOptimizer {
    public:

    virtual ~LocalInertialBAOptimizer() {}

    virtual void clear() = 0;
    virtual void reserve(const unsigned int max_keyframes, const unsigned int max_map_points,
                         const unsigned int max_visual_factors) = 0;

    virtual void add_pose(const size_t id, KeyFrame* pKF, const bool fixed) = 0;
    virtual void add_velocity(const size_t id, KeyFrame* pKF, const bool fixed) = 0;
    virtual void add_gyro_bias(const size_t id, KeyFrame* pKF, const bool fixed) = 0;
    virtual void add_acc_bias(const size_t id, KeyFrame* pKF, const bool fixed) = 0;
    virtual void add_map_point(const size_t id, const Eigen::Vector3d &position) = 0;

    virtual bool has_pose(const size_t id) = 0;
    virtual bool has_velocity(const size_t id) = 0;
    virtual bool has_gyro_bias(const size_t id) = 0;
    virtual bool has_acc_bias(const size_t id) = 0;

    virtual void add_inertial_factor(const size_t pose1, const size_t velocity1, const size_t gyro_bias1,
                                     const size_t acc_bias1, const size_t pose2, const size_t velocity2,
                                     IMU::Preintegrated* pInt, const bool robust, const double info_scale,
                                     const double huber_delta) = 0;
    virtual void add_gyro_rw_factor(const size_t gyro_bias1, const size_t gyro_bias2, const double* info) = 0;
    virtual void add_acc_rw_factor(const size_t acc_bias1, const size_t acc_bias2, const double* info) = 0;
    virtual size_t add_mono_factor(const size_t map_point, const size_t pose, const Eigen::Vector2d &obs,
                                   const double information, const int cam_idx, const double huber_delta) = 0;
    virtual size_t add_stereo_factor(const size_t map_point, const size_t pose, const Eigen::Vector3d &obs,
                                     const double information, const int cam_idx, const double huber_delta) = 0;

    virtual void optimize(const size_t iterations, const double lambda, bool* stop_flag,
                          double &initial_error, double &final_error, const bool verbose) = 0;

    virtual double mono_chi2(const size_t factor_id) = 0;
    virtual double stereo_chi2(const size_t factor_id) = 0;
    virtual bool mono_depth_positive(const size_t factor_id) = 0;

    virtual LIBASE3Pose get_pose(const size_t id) = 0;
    virtual Eigen::Vector3d get_velocity(const size_t id) = 0;
    virtual Eigen::Vector3d get_gyro_bias(const size_t id) = 0;
    virtual Eigen::Vector3d get_acc_bias(const size_t id) = 0;
    virtual Eigen::Vector3d get_map_point(const size_t id) = 0;

};

template <typename Camera>
class LocalInertialBAOptimizerImpl : public LocalInertialBAOptimizer {
    public:

    using FP = double;
    using SP = double;

    LocalInertialBAOptimizerImpl(const unsigned int max_keyframes, const unsigned int max_map_points,
                                 const unsigned int max_visual_factors);
    ~LocalInertialBAOptimizerImpl() override;

    void clear() override;
    void reserve(const unsigned int max_keyframes, const unsigned int max_map_points,
                 const unsigned int max_visual_factors) override;

    void add_pose(const size_t id, KeyFrame* pKF, const bool fixed) override;
    void add_velocity(const size_t id, KeyFrame* pKF, const bool fixed) override;
    void add_gyro_bias(const size_t id, KeyFrame* pKF, const bool fixed) override;
    void add_acc_bias(const size_t id, KeyFrame* pKF, const bool fixed) override;
    void add_map_point(const size_t id, const Eigen::Vector3d &position) override;

    bool has_pose(const size_t id) override { return pose_desc.exists(id); }
    bool has_velocity(const size_t id) override { return velocity_desc.exists(id); }
    bool has_gyro_bias(const size_t id) override { return gyro_bias_desc.exists(id); }
    bool has_acc_bias(const size_t id) override { return acc_bias_desc.exists(id); }

    void add_inertial_factor(const size_t pose1, const size_t velocity1, const size_t gyro_bias1,
                             const size_t acc_bias1, const size_t pose2, const size_t velocity2,
                             IMU::Preintegrated* pInt, const bool robust, const double info_scale,
                             const double huber_delta) override;
    void add_gyro_rw_factor(const size_t gyro_bias1, const size_t gyro_bias2, const double* info) override;
    void add_acc_rw_factor(const size_t acc_bias1, const size_t acc_bias2, const double* info) override;
    size_t add_mono_factor(const size_t map_point, const size_t pose, const Eigen::Vector2d &obs,
                           const double information, const int cam_idx, const double huber_delta) override;
    size_t add_stereo_factor(const size_t map_point, const size_t pose, const Eigen::Vector3d &obs,
                             const double information, const int cam_idx, const double huber_delta) override;

    void optimize(const size_t iterations, const double lambda, bool* stop_flag,
                  double &initial_error, double &final_error, const bool verbose) override;

    double mono_chi2(const size_t factor_id) override { return mono_desc.chi2(factor_id); }
    double stereo_chi2(const size_t factor_id) override { return stereo_desc.chi2(factor_id); }
    bool mono_depth_positive(const size_t factor_id) override;

    LIBASE3Pose get_pose(const size_t id) override;
    Eigen::Vector3d get_velocity(const size_t id) override { return *velocity_desc.get_vertex(id); }
    Eigen::Vector3d get_gyro_bias(const size_t id) override { return *gyro_bias_desc.get_vertex(id); }
    Eigen::Vector3d get_acc_bias(const size_t id) override { return *acc_bias_desc.get_vertex(id); }
    Eigen::Vector3d get_map_point(const size_t id) override { return *mp_desc.get_vertex(id); }

    private:

    // Only descriptors which actually hold vertices/factors are added to the graph,
    // so this has to be done once the graph is built.
    void add_descriptors();

    // Returns the (cached) managed cameras of a keyframe
    static constexpr size_t max_cameras = 2;
    std::array<Camera*, max_cameras> get_cameras(KeyFrame* pKFi);

    static constexpr uint8_t optimization_level = 0;

    using Pose = gpu::ImuCamPose<FP, Camera>;
    using PoseDescriptorType = gpu::PoseDescriptor<FP, SP, Camera>;

    graphite::Graph<FP, SP> graph;
    graphite::EigenSchurLDLTSolver<FP, SP> solver;
    graphite::StreamPool streams;

    // Vertex storage
    graphite::managed_vector<Pose> poses;
    graphite::managed_vector<gpu::Velocity<FP>> velocities;
    graphite::managed_vector<gpu::GyroBias<FP>> gyro_biases;
    graphite::managed_vector<gpu::AccBias<FP>> acc_biases;
    graphite::managed_vector<gpu::SBAPointXYZ<FP>> map_points;

    // Cameras are shared between keyframes and stay allocated across calls
    std::unordered_map<GeometricCamera*, Camera*> cameras;

    // Vertex descriptors
    PoseDescriptorType pose_desc;
    gpu::VelocityDescriptor<FP, SP> velocity_desc;
    gpu::GyroBiasDescriptor<FP, SP> gyro_bias_desc;
    gpu::AccBiasDescriptor<FP, SP> acc_bias_desc;
    gpu::SBAPointXYZDescriptor<FP, SP> mp_desc;

    // Factor descriptors
    gpu::InertialConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 9>, PoseDescriptorType> ic_desc_rk;
    gpu::InertialConstraintDescriptor<FP, SP, graphite::DefaultLoss<FP, 9>, PoseDescriptorType> ic_desc;
    gpu::GyroRWConstraintDescriptor<FP, SP, graphite::DefaultLoss<FP, 3>> gc_desc;
    gpu::AccRWConstraintDescriptor<FP, SP, graphite::DefaultLoss<FP, 3>> ac_desc;
    gpu::StereoConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 3>, Camera> stereo_desc;
    gpu::MonoConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 2>, Camera> mono_desc;

    // Number of vertices/factors added since the last clear
    size_t pose_count;
    size_t velocity_count;
    size_t gyro_bias_count;
    size_t acc_bias_count;
    size_t map_point_count;

    size_t ic_rk_count;
    size_t ic_count;
    size_t gc_count;
    size_t ac_count;
    size_t mono_count;
    size_t stereo_count;

};

}

namespace ORB_SLAM3 {

    // Interface implementation
    LocalInertialBAInterface::LocalInertialBAInterface(const unsigned int max_keyframes,
            const unsigned int max_map_points, const unsigned int max_visual_factors):
    liba(nullptr), liba_pinhole(nullptr), liba_fisheye(nullptr),
    max_keyframes(max_keyframes), max_map_points(max_map_points), max_visual_factors(max_visual_factors) {
        // The optimizer itself is created by the first call to set_camera_model, so that
        // no memory is reserved for a camera model which is never used.
    }

    LocalInertialBAInterface::~LocalInertialBAInterface() {

        if (liba_pinhole) {
            delete liba_pinhole;
        }
        if (liba_fisheye) {
            delete liba_fisheye;
        }
        liba_pinhole = nullptr;
        liba_fisheye = nullptr;
        liba = nullptr;

    }

    void LocalInertialBAInterface::set_camera_model(const CameraModel model) {
        if (model == CameraModel::Pinhole) {
            if (!liba_pinhole) {
                liba_pinhole = new LocalInertialBAOptimizerImpl<gpu::PinholeCamera<double>>(
                        max_keyframes, max_map_points, max_visual_factors);
            }
            liba = liba_pinhole;
        }
        else {
            if (!liba_fisheye) {
                liba_fisheye = new LocalInertialBAOptimizerImpl<gpu::KannalaBrandt8Camera<double>>(
                        max_keyframes, max_map_points, max_visual_factors);
            }
            liba = liba_fisheye;
        }
    }

    void LocalInertialBAInterface::clear() {
        liba->clear();
    }

    void LocalInertialBAInterface::reserve(const unsigned int max_keyframes, const unsigned int max_map_points,
                                           const unsigned int max_visual_factors) {
        liba->reserve(max_keyframes, max_map_points, max_visual_factors);
    }

    void LocalInertialBAInterface::add_pose(const size_t id, KeyFrame* pKF, const bool fixed) {
        liba->add_pose(id, pKF, fixed);
    }

    void LocalInertialBAInterface::add_velocity(const size_t id, KeyFrame* pKF, const bool fixed) {
        liba->add_velocity(id, pKF, fixed);
    }

    void LocalInertialBAInterface::add_gyro_bias(const size_t id, KeyFrame* pKF, const bool fixed) {
        liba->add_gyro_bias(id, pKF, fixed);
    }

    void LocalInertialBAInterface::add_acc_bias(const size_t id, KeyFrame* pKF, const bool fixed) {
        liba->add_acc_bias(id, pKF, fixed);
    }

    void LocalInertialBAInterface::add_map_point(const size_t id, const Eigen::Vector3d &position) {
        liba->add_map_point(id, position);
    }

    bool LocalInertialBAInterface::has_pose(const size_t id) {
        return liba->has_pose(id);
    }

    bool LocalInertialBAInterface::has_velocity(const size_t id) {
        return liba->has_velocity(id);
    }

    bool LocalInertialBAInterface::has_gyro_bias(const size_t id) {
        return liba->has_gyro_bias(id);
    }

    bool LocalInertialBAInterface::has_acc_bias(const size_t id) {
        return liba->has_acc_bias(id);
    }

    void LocalInertialBAInterface::add_inertial_factor(const size_t pose1, const size_t velocity1,
                    const size_t gyro_bias1, const size_t acc_bias1, const size_t pose2, const size_t velocity2,
                    IMU::Preintegrated* pInt, const bool robust, const double info_scale, const double huber_delta) {
        liba->add_inertial_factor(pose1, velocity1, gyro_bias1, acc_bias1, pose2, velocity2,
                                  pInt, robust, info_scale, huber_delta);
    }

    void LocalInertialBAInterface::add_gyro_rw_factor(const size_t gyro_bias1, const size_t gyro_bias2,
                    const double* info) {
        liba->add_gyro_rw_factor(gyro_bias1, gyro_bias2, info);
    }

    void LocalInertialBAInterface::add_acc_rw_factor(const size_t acc_bias1, const size_t acc_bias2,
                    const double* info) {
        liba->add_acc_rw_factor(acc_bias1, acc_bias2, info);
    }

    size_t LocalInertialBAInterface::add_mono_factor(const size_t map_point, const size_t pose,
                    const Eigen::Vector2d &obs, const double information, const int cam_idx,
                    const double huber_delta) {
        return liba->add_mono_factor(map_point, pose, obs, information, cam_idx, huber_delta);
    }

    size_t LocalInertialBAInterface::add_stereo_factor(const size_t map_point, const size_t pose,
                    const Eigen::Vector3d &obs, const double information, const int cam_idx,
                    const double huber_delta) {
        return liba->add_stereo_factor(map_point, pose, obs, information, cam_idx, huber_delta);
    }

    void LocalInertialBAInterface::optimize(const size_t iterations, const double lambda, bool* stop_flag,
                    double &initial_error, double &final_error, const bool verbose) {
        liba->optimize(iterations, lambda, stop_flag, initial_error, final_error, verbose);
    }

    double LocalInertialBAInterface::mono_chi2(const size_t factor_id) {
        return liba->mono_chi2(factor_id);
    }

    double LocalInertialBAInterface::stereo_chi2(const size_t factor_id) {
        return liba->stereo_chi2(factor_id);
    }

    bool LocalInertialBAInterface::mono_depth_positive(const size_t factor_id) {
        return liba->mono_depth_positive(factor_id);
    }

    LIBASE3Pose LocalInertialBAInterface::get_pose(const size_t id) {
        return liba->get_pose(id);
    }

    Eigen::Vector3d LocalInertialBAInterface::get_velocity(const size_t id) {
        return liba->get_velocity(id);
    }

    Eigen::Vector3d LocalInertialBAInterface::get_gyro_bias(const size_t id) {
        return liba->get_gyro_bias(id);
    }

    Eigen::Vector3d LocalInertialBAInterface::get_acc_bias(const size_t id) {
        return liba->get_acc_bias(id);
    }

    Eigen::Vector3d LocalInertialBAInterface::get_map_point(const size_t id) {
        return liba->get_map_point(id);
    }


    // Implementation of LocalInertialBAOptimizerImpl
    template <typename Camera>
    LocalInertialBAOptimizerImpl<Camera>::LocalInertialBAOptimizerImpl(const unsigned int max_keyframes,
            const unsigned int max_map_points, const unsigned int max_visual_factors):
    streams(7),
    ic_desc_rk(&pose_desc, &velocity_desc, &gyro_bias_desc, &acc_bias_desc, &pose_desc, &velocity_desc),
    ic_desc(&pose_desc, &velocity_desc, &gyro_bias_desc, &acc_bias_desc, &pose_desc, &velocity_desc),
    gc_desc(&gyro_bias_desc, &gyro_bias_desc),
    ac_desc(&acc_bias_desc, &acc_bias_desc),
    stereo_desc(&mp_desc, &pose_desc),
    mono_desc(&mp_desc, &pose_desc),
    pose_count(0), velocity_count(0), gyro_bias_count(0), acc_bias_count(0), map_point_count(0),
    ic_rk_count(0), ic_count(0), gc_count(0), ac_count(0), mono_count(0), stereo_count(0) {
        mp_desc.set_eliminate(true);
        this->reserve(max_keyframes, max_map_points, max_visual_factors);
    }

    template <typename Camera>
    LocalInertialBAOptimizerImpl<Camera>::~LocalInertialBAOptimizerImpl() {
        for (auto& [cam_ptr, cam] : cameras) {
            if (cam) {
                freeSharedMemory(cam);
            }
        }
        cameras.clear();
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::clear() {
        // None of these free their memory, so the allocations are reused by the next call
        graph.clear();

        pose_desc.clear();
        velocity_desc.clear();
        gyro_bias_desc.clear();
        acc_bias_desc.clear();
        mp_desc.clear();

        ic_desc_rk.clear();
        ic_desc.clear();
        gc_desc.clear();
        ac_desc.clear();
        stereo_desc.clear();
        mono_desc.clear();

        pose_count = 0;
        velocity_count = 0;
        gyro_bias_count = 0;
        acc_bias_count = 0;
        map_point_count = 0;

        ic_rk_count = 0;
        ic_count = 0;
        gc_count = 0;
        ac_count = 0;
        mono_count = 0;
        stereo_count = 0;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::reserve(const unsigned int max_keyframes,
            const unsigned int max_map_points, const unsigned int max_visual_factors) {

        // The descriptors store pointers to the vertices, so these must not be resized
        // while the graph is being built (i.e. reserve everything up front).
        if (poses.size() < max_keyframes) {
            poses.resize(max_keyframes);
            velocities.resize(max_keyframes);
            gyro_biases.resize(max_keyframes);
            acc_biases.resize(max_keyframes);
        }
        if (map_points.size() < max_map_points) {
            map_points.resize(max_map_points);
        }

        pose_desc.reserve(max_keyframes);
        velocity_desc.reserve(max_keyframes);
        gyro_bias_desc.reserve(max_keyframes);
        acc_bias_desc.reserve(max_keyframes);
        mp_desc.reserve(max_map_points);

        ic_desc_rk.reserve(max_keyframes);
        ic_desc.reserve(max_keyframes);
        gc_desc.reserve(max_keyframes);
        ac_desc.reserve(max_keyframes);
        stereo_desc.reserve(max_visual_factors);
        mono_desc.reserve(max_visual_factors);
    }

    template <typename Camera>
    std::array<Camera*, LocalInertialBAOptimizerImpl<Camera>::max_cameras>
    LocalInertialBAOptimizerImpl<Camera>::get_cameras(KeyFrame* pKFi) {

        constexpr size_t num_params = Camera::parameter_size;

        std::array<Camera*, max_cameras> cams = {nullptr, nullptr};

        GeometricCamera* pCameras[max_cameras] = {pKFi->mpCamera, pKFi->mpCamera2};

        for (size_t i = 0; i < max_cameras; i++) {
            if (!pCameras[i]) {
                continue;
            }

            const auto it = cameras.find(pCameras[i]);
            if (it != cameras.end()) {
                cams[i] = it->second;
                continue;
            }

            Camera* cam;
            allocateSharedMemory((void**)&cam, sizeof(Camera));

            std::array<FP, num_params> cam_params;
            for (size_t j = 0; j < cam_params.size(); j++) {
                cam_params[j] = pCameras[i]->getParameter(j);
            }
            *cam = Camera(cam_params);

            cameras[pCameras[i]] = cam;
            cams[i] = cam;
        }

        return cams;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_pose(const size_t id, KeyFrame* pKF, const bool fixed) {
        if (pose_count >= poses.size()) {
            std::cerr << "LIBA: pose storage exhausted (" << poses.size() << "), skipping vertex " << id << std::endl;
            return;
        }
        auto cams = get_cameras(pKF);
        poses[pose_count] = Pose(pKF, cams.data());
        pose_desc.add_vertex(id, &poses[pose_count], fixed);
        pose_count++;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_velocity(const size_t id, KeyFrame* pKF, const bool fixed) {
        if (velocity_count >= velocities.size()) {
            std::cerr << "LIBA: velocity storage exhausted (" << velocities.size() << "), skipping vertex " << id << std::endl;
            return;
        }
        velocities[velocity_count] = pKF->GetVelocity().cast<FP>();
        velocity_desc.add_vertex(id, &velocities[velocity_count], fixed);
        velocity_count++;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_gyro_bias(const size_t id, KeyFrame* pKF, const bool fixed) {
        if (gyro_bias_count >= gyro_biases.size()) {
            std::cerr << "LIBA: gyro bias storage exhausted (" << gyro_biases.size() << "), skipping vertex " << id << std::endl;
            return;
        }
        gyro_biases[gyro_bias_count] = pKF->GetGyroBias().cast<FP>();
        gyro_bias_desc.add_vertex(id, &gyro_biases[gyro_bias_count], fixed);
        gyro_bias_count++;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_acc_bias(const size_t id, KeyFrame* pKF, const bool fixed) {
        if (acc_bias_count >= acc_biases.size()) {
            std::cerr << "LIBA: acc bias storage exhausted (" << acc_biases.size() << "), skipping vertex " << id << std::endl;
            return;
        }
        acc_biases[acc_bias_count] = pKF->GetAccBias().cast<FP>();
        acc_bias_desc.add_vertex(id, &acc_biases[acc_bias_count], fixed);
        acc_bias_count++;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_map_point(const size_t id, const Eigen::Vector3d &position) {
        if (map_point_count >= map_points.size()) {
            std::cerr << "LIBA: map point storage exhausted (" << map_points.size() << "), skipping vertex " << id << std::endl;
            return;
        }
        map_points[map_point_count] = position.cast<FP>();
        mp_desc.add_vertex(id, &map_points[map_point_count], false);
        map_point_count++;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_inertial_factor(const size_t pose1, const size_t velocity1,
            const size_t gyro_bias1, const size_t acc_bias1, const size_t pose2, const size_t velocity2,
            IMU::Preintegrated* pInt, const bool robust, const double info_scale, const double huber_delta) {

        gpu::InertialConstraintData<FP> ic_data(pInt);
        auto info = ic_data.template get_information_matrix<SP>(pInt);
        info *= info_scale;

        if (robust) {
            ic_desc_rk.add_factor({pose1, velocity1, gyro_bias1, acc_bias1, pose2, velocity2},
                    graphite::Empty(), info.data(), ic_data, graphite::HuberLoss<FP, 9>(huber_delta));
            ic_rk_count++;
        }
        else {
            ic_desc.add_factor({pose1, velocity1, gyro_bias1, acc_bias1, pose2, velocity2},
                    graphite::Empty(), info.data(), ic_data, graphite::DefaultLoss<FP, 9>());
            ic_count++;
        }
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_gyro_rw_factor(const size_t gyro_bias1,
            const size_t gyro_bias2, const double* info) {
        gpu::Mat3<SP> InfoG = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(info).cast<SP>();
        gc_desc.add_factor({gyro_bias1, gyro_bias2}, graphite::Empty(), InfoG.data(), graphite::Empty(),
                graphite::DefaultLoss<FP, 3>());
        gc_count++;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_acc_rw_factor(const size_t acc_bias1,
            const size_t acc_bias2, const double* info) {
        gpu::Mat3<SP> InfoA = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(info).cast<SP>();
        ac_desc.add_factor({acc_bias1, acc_bias2}, graphite::Empty(), InfoA.data(), graphite::Empty(),
                graphite::DefaultLoss<FP, 3>());
        ac_count++;
    }

    template <typename Camera>
    size_t LocalInertialBAOptimizerImpl<Camera>::add_mono_factor(const size_t map_point, const size_t pose,
            const Eigen::Vector2d &obs, const double information, const int cam_idx, const double huber_delta) {
        gpu::Mat2<SP> info = gpu::Mat2<SP>::Identity() * information;
        const auto f_id = mono_desc.add_factor({map_point, pose}, obs.cast<FP>(), info.data(), cam_idx,
                graphite::HuberLoss<FP, 2>(huber_delta));
        mono_count++;
        return f_id;
    }

    template <typename Camera>
    size_t LocalInertialBAOptimizerImpl<Camera>::add_stereo_factor(const size_t map_point, const size_t pose,
            const Eigen::Vector3d &obs, const double information, const int cam_idx, const double huber_delta) {
        gpu::Mat3<SP> info = gpu::Mat3<SP>::Identity() * information;
        const auto f_id = stereo_desc.add_factor({map_point, pose}, obs.cast<FP>(), info.data(), cam_idx,
                graphite::HuberLoss<FP, 3>(huber_delta));
        stereo_count++;
        return f_id;
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::add_descriptors() {

        // Remove the descriptors of the previous call (this keeps the memory of the graph)
        graph.clear();

        if (pose_count > 0) {
            graph.add_vertex_descriptor(&pose_desc);
        }
        if (velocity_count > 0) {
            graph.add_vertex_descriptor(&velocity_desc);
        }
        if (gyro_bias_count > 0) {
            graph.add_vertex_descriptor(&gyro_bias_desc);
        }
        if (acc_bias_count > 0) {
            graph.add_vertex_descriptor(&acc_bias_desc);
        }
        if (map_point_count > 0) {
            graph.add_vertex_descriptor(&mp_desc);
        }

        if (gc_count > 0) {
            graph.add_factor_descriptor(&gc_desc);
        }
        if (ac_count > 0) {
            graph.add_factor_descriptor(&ac_desc);
        }
        if (ic_rk_count > 0) {
            graph.add_factor_descriptor(&ic_desc_rk);
        }
        if (ic_count > 0) {
            graph.add_factor_descriptor(&ic_desc);
        }
        if (stereo_count > 0) {
            graph.add_factor_descriptor(&stereo_desc);
        }
        if (mono_count > 0) {
            graph.add_factor_descriptor(&mono_desc);
        }
    }

    template <typename Camera>
    void LocalInertialBAOptimizerImpl<Camera>::optimize(const size_t iterations, const double lambda,
            bool* stop_flag, double &initial_error, double &final_error, const bool verbose) {

        add_descriptors();

        graphite::optimizer::LevenbergMarquardtOptions<FP, SP> options;
        options.solver = &solver;
        options.iterations = iterations;
        options.initial_damping = lambda;
        options.optimization_level = optimization_level;
        options.streams = &streams;
        options.stop_flag = stop_flag;
        options.verbose = verbose;
        options.use_identity = true;

        graph.scale_system(false);

        // Compute initial error
        graph.initialize_optimization(optimization_level);
        graph.build_structure();
        graph.compute_error();
        initial_error = graph.chi2();

        graphite::optimizer::levenberg_marquardt2<FP, SP>(&graph, &options);

        final_error = graph.chi2();
    }

    template <typename Camera>
    bool LocalInertialBAOptimizerImpl<Camera>::mono_depth_positive(const size_t factor_id) {
        const auto cam_idx = *mono_desc.get_constraint_data(factor_id);
        const auto ids = mono_desc.get_vertex_ids(factor_id);

        const auto mp_vec = *mp_desc.get_vertex(ids[0]);
        const auto pose = *pose_desc.get_vertex(ids[1]);

        return pose.isDepthPositive(mp_vec, cam_idx);
    }

    template <typename Camera>
    LIBASE3Pose LocalInertialBAOptimizerImpl<Camera>::get_pose(const size_t id) {
        const auto & p = *pose_desc.get_vertex(id);
        return LIBASE3Pose{p.Rcw[0], p.tcw[0]};
    }

} // namespace ORB_SLAM3
