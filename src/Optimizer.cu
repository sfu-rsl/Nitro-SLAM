#include "Optimizer.h"
#include "PGOTypes.h"
#include <graphite/solver/eigen.hpp>
#include "PGOInterface.h"

namespace ORB_SLAM3 {
class PoseGraphOptimizer {
    public:


    PoseGraphOptimizer(const unsigned int max_poses, const unsigned int max_edges);


    // void add_pose(const int id, const Pose4DoF<double> &pose);

    void add_pose(const int id, ORB_SLAM3::KeyFrame* pKF);
    void add_pose(const int id, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, ORB_SLAM3::KeyFrame* pKF);
    
    SE3Pose get_pose(const int id);
    void set_fixed(const int id, const bool fixed);
    void add_factor(const int id1, const int id2, const Sophus::SE3d &relative_pose, const double* info);
    void optimize(const size_t iterations, const double lambda, const bool verbose);
    void clear();
    void reserve(const unsigned int max_poses, const unsigned int max_edges);

    private:

    graphite::Graph<double, double> graph;
    // graphite::BlockJacobiPreconditioner<double, double> preconditioner;
    // graphite::PCGSolver<double, double> solver;
    graphite::EigenLDLTSolver<double, double> solver;
    // graphite::cudssSolver<double, double> solver;
    graphite::StreamPool streams;

    using Pose = gpu::ImuCamPose<double, gpu::PinholeCamera<double>>;
    graphite::managed_vector<Pose> poses;
    gpu::Pose4DoFDescriptor<double, double> pose_desc;
    gpu::Factor4DoFDescriptor<double, double, graphite::DefaultLoss<double, 6>> edge_desc;

};

}

namespace ORB_SLAM3 {

    // Interface implementation
    PoseGraphOptimizerInterface::PoseGraphOptimizerInterface(const unsigned int max_poses, const unsigned int max_edges) {
        pgo = new PoseGraphOptimizer(max_poses, max_edges);
    }

    PoseGraphOptimizerInterface::~PoseGraphOptimizerInterface() {
    
        if (pgo) {
            delete pgo;
        }
        pgo = nullptr;
    
    }

    void PoseGraphOptimizerInterface::clear() {
        pgo->clear();
    }

    void PoseGraphOptimizerInterface::reserve(const unsigned int max_poses, const unsigned int max_edges) {
        pgo->reserve(max_poses, max_edges);
    }

    void PoseGraphOptimizerInterface::add_pose(const int id, ORB_SLAM3::KeyFrame* pKF) {
        pgo->add_pose(id, pKF);
    }

    void PoseGraphOptimizerInterface::add_pose(const int id, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, ORB_SLAM3::KeyFrame* pKF) {
        pgo->add_pose(id, Rwc, twc, pKF);
    }

    // void PoseGraphOptimizerInterface::add_pose(const int id, const Pose4DoF<double> & pose) {
    //     pgo->add_pose(id, pose);
    // }
    SE3Pose PoseGraphOptimizerInterface::get_pose(const int id) {
        return pgo->get_pose(id);
    }

    void PoseGraphOptimizerInterface::set_fixed(const int id, const bool fixed) {
        pgo->set_fixed(id, fixed);
    }

    void PoseGraphOptimizerInterface::add_factor(const int id1, const int id2,
                    const Sophus::SE3d& relative_pose,
                    const double* info) {
        pgo->add_factor(id1, id2, relative_pose, info);
    }

    void PoseGraphOptimizerInterface::optimize(const size_t iterations, const double lambda, const bool verbose) {
        pgo->optimize(iterations, lambda, verbose);
    }


    // Implementation of PoseGraphOptimizer
    PoseGraphOptimizer::PoseGraphOptimizer(const unsigned int max_poses, const unsigned int max_edges):
    // solver(true), // cuDSS solver
    // solver(10, 1e-6, std::numeric_limits<double>::infinity(), &preconditioner),
    streams(4), poses(max_poses), edge_desc(&pose_desc, &pose_desc) {
        this->reserve(max_poses, max_edges);
        // set up a simple PGO problem and optimize it to deal with startup overhead
        // if (max_poses > 1) {
        //     poses[0] = Pose();
        //     poses[1] = Pose();
        //     pose_desc.add_vertex(0, &poses[0]);
        //     pose_desc.add_vertex(1, &poses[1]);
        //     edge_desc.add_factor({0, 1}, Sophus::SE3d(), nullptr, graphite::Empty(), graphite::DefaultLoss<double, 6>());
        //     optimize(1, 1e-4, false);

        //     clear();
        // }
    }

    void PoseGraphOptimizer::clear() {
        graph.clear();
        pose_desc.clear();
        edge_desc.clear();
    }

    void PoseGraphOptimizer::reserve(const unsigned int max_poses, const unsigned int max_edges) {
        pose_desc.reserve(max_poses);
        poses.resize(max_poses);
        edge_desc.reserve(max_edges);
        graph.add_vertex_descriptor(&pose_desc);
        graph.add_factor_descriptor(&edge_desc);
    }

    void PoseGraphOptimizer::add_pose(const int id, ORB_SLAM3::KeyFrame* pKF) {
        poses[id] = Pose(pKF, nullptr);
        pose_desc.add_vertex(id, &poses[id]);
    }

    void PoseGraphOptimizer::add_pose(const int id, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, ORB_SLAM3::KeyFrame* pKF) {
        poses[id] = Pose(Rwc, twc, pKF);
        pose_desc.add_vertex(id, &poses[id]);
    }

    SE3Pose PoseGraphOptimizer::get_pose(const int id) {
        const auto & p = poses[id];
        // return Sophus::SE3d (p.Rcw[0], p.tcw[0]);
        return SE3Pose{p.Rcw[0], p.tcw[0]};
    }

    void PoseGraphOptimizer::set_fixed(const int id, const bool fixed) {
        pose_desc.set_fixed(id, fixed);
    }

    void PoseGraphOptimizer::add_factor(const int id1, const int id2,
                    const Sophus::SE3d& relative_pose,
                    const double* info) {
        edge_desc.add_factor({(size_t)id1, (size_t)id2}, relative_pose, info, graphite::Empty(), graphite::DefaultLoss<double, 6>());
    }

    void PoseGraphOptimizer::optimize(const size_t iterations, const double lambda, const bool verbose) {
        graphite::optimizer::LevenbergMarquardtOptions<double, double> options;
        options.solver = &solver;
        options.iterations = iterations;
        options.initial_damping = lambda; // note: original code uses g2o to autocompute initial lambda
        options.streams = &streams;
        options.verbose = verbose;

        graph.scale_system(false);

        graphite::optimizer::levenberg_marquardt2(&graph, &options);

    }
} // namespace ORB_SLAM3