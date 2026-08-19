#pragma once
#include "Optimizer.h"

#include <graphite/vector.hpp>
#include <graphite/loss.hpp>
// #include "GPUPose.h"
#include "GPUTypes.h"
// #include "PGOTypes.h"
#include <graphite/solver/eigen_schur.hpp>
#include <graphite/solver/cudss_schur.hpp>
#include <graphite/solver/pcg.hpp>
#include <graphite/preconditioner/block_jacobi.hpp>
#include <graphite/optimizer/levenberg_marquardt.hpp>

#include <memory>



namespace ORB_SLAM3 {

namespace OptimizerGPU {


template <typename FP, typename Camera, size_t max_cameras>
static std::array<Camera*, max_cameras> get_cameras(KeyFrame* pKFi, std::unordered_map<GeometricCamera*, Camera*>& cameras){
    std::array<Camera*, max_cameras> cams = {nullptr, nullptr};
    constexpr size_t num_params = Camera::parameter_size;
    if (cameras.find(pKFi->mpCamera) == cameras.end()) {
        Camera* cam;
        cudaMallocManaged(&cam, sizeof(Camera));
        // Initialize the camera with placement new
        std::array<FP, num_params> cam_params;
        for (size_t i = 0; i < cam_params.size(); i++) {
            cam_params[i] = pKFi->mpCamera->getParameter(i);
        }

        // new (cam) Camera(cam_params);
        *cam = Camera(cam_params);
        cameras[pKFi->mpCamera] = cam;
        cams[0] = cam;
    }
    else {
        cams[0] = cameras[pKFi->mpCamera];
    }
    if (pKFi->mpCamera2) {
        if (cameras.find(pKFi->mpCamera2) == cameras.end()) {
            Camera* cam;
            cudaMallocManaged(&cam, sizeof(Camera));
            std::array<FP, num_params> cam_params;
            for (size_t i = 0; i < cam_params.size(); i++) {
                cam_params[i] = pKFi->mpCamera2->getParameter(i);
            }
            // new (cam) Camera(cam_params);
            *cam = Camera(cam_params);

            cameras[pKFi->mpCamera2] = cam;
            cams[1] = cam;
        }
        else {
            cams[1] = cameras[pKFi->mpCamera2];
        }
    }
    return cams;
};

template <typename FP, typename SP, typename Camera, bool skip_recovery>
void FullInertialBAInternal(bool use_pcg, Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
{
    using namespace graphite;
    using namespace gpu;

    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    // Setup optimizer
    // g2o::SparseOptimizer optimizer;
    // g2o::BlockSolverX::LinearSolverType * linearSolver;

    // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    // g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    // solver->setUserLambdaInit(1e-5);
    // optimizer.setAlgorithm(solver);
    // optimizer.setVerbose(false);

    Graph<FP, SP> graph;
    StreamPool streams(7); // 1 works - is there a problem with 7?
    BlockJacobiPreconditioner<FP, SP> preconditioner;
    std::unique_ptr<Solver<FP, SP>> solver;
    if (use_pcg) {
        solver = std::make_unique<PCGSolver<FP, SP>>(100, 1.0e-12, 10.0, &preconditioner);
    } else {
        solver = std::make_unique<cudssSchurSolver<FP, SP>>();
    }
    graph.scale_system(false);
    constexpr uint8_t optimization_level = 0;
    const double lambda = 1e-5;


    // if(pbStopFlag)
    //     optimizer.setForceStopFlag(pbStopFlag);

    int nNonFixed = 0;

    // Create storage
    constexpr size_t max_cameras = 2;
    using Pose = ImuCamPose<FP, Camera>;

    const auto num_keyframes_total = vpKFs.size() + 1;
    graphite::managed_vector<Pose> poses(num_keyframes_total);
    graphite::managed_vector<Velocity<FP>> velocities(num_keyframes_total);
    graphite::managed_vector<GyroBias<FP>> gyro_biases(num_keyframes_total);
    graphite::managed_vector<AccBias<FP>> acc_biases(num_keyframes_total);
    std::unordered_map<GeometricCamera*, Camera*>  cameras;    

    auto cleanup_cameras = [&cameras]() {
        for (auto& kv : cameras) {
            auto* cam = kv.second;
            if (cam) {
                cudaFree(cam);
                kv.second = nullptr;
            }
        }
        cameras.clear();
    };

    // Create vertex descriptors
    auto pose_desc = PoseDescriptor<FP, SP, Camera>();
    pose_desc.reserve(num_keyframes_total);
    graph.add_vertex_descriptor(&pose_desc);

    auto velocity_desc = VelocityDescriptor<FP, SP>();
    velocity_desc.reserve(num_keyframes_total);
    graph.add_vertex_descriptor(&velocity_desc);

    auto gyro_bias_desc = GyroBiasDescriptor<FP, SP>();
    gyro_bias_desc.reserve(num_keyframes_total);
    graph.add_vertex_descriptor(&gyro_bias_desc);

    auto acc_bias_desc = AccBiasDescriptor<FP, SP>();
    acc_bias_desc.reserve(num_keyframes_total);
    graph.add_vertex_descriptor(&acc_bias_desc);
    size_t alloc_index = 0;
    std::cout << "Creating keyframe vertices for FullInertialBA" << std::endl;
    // Set KeyFrame vertices
    KeyFrame* pIncKF;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        // VertexPose * VP = new VertexPose(pKFi);
        // VP->setId(pKFi->mnId);
        auto cams = get_cameras<FP, Camera, max_cameras>(pKFi, cameras);
        poses[alloc_index] = Pose(pKFi, cams.data());

        pIncKF=pKFi;
        bool bFixed = false;
        if(bFixLocal)
        {
            bFixed = (pKFi->mnBALocalForKF>=(maxKFid-1)) || (pKFi->mnBAFixedForKF>=(maxKFid-1));
            if(!bFixed)
                nNonFixed++;
            // VP->setFixed(bFixed);
        }
        pose_desc.add_vertex(pKFi->mnId, &poses[alloc_index], bFixed);
        // optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            // VertexVelocity* VV = new VertexVelocity(pKFi);
            // VV->setId(maxKFid+3*(pKFi->mnId)+1);
            // VV->setFixed(bFixed);
            // optimizer.addVertex(VV);

            velocities[alloc_index] = pKFi->GetVelocity().cast<FP>();
            velocity_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 1, &velocities[alloc_index], bFixed);

            if (!bInit)
            {
                // VertexGyroBias* VG = new VertexGyroBias(pKFi);
                // VG->setId(maxKFid+3*(pKFi->mnId)+2);
                // VG->setFixed(bFixed);
                // optimizer.addVertex(VG);

                gyro_biases[alloc_index] = pKFi->GetGyroBias().cast<FP>();
                gyro_bias_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 2, &gyro_biases[alloc_index], bFixed);

                // VertexAccBias* VA = new VertexAccBias(pKFi);
                // VA->setId(maxKFid+3*(pKFi->mnId)+3);
                // VA->setFixed(bFixed);
                // optimizer.addVertex(VA);

                acc_biases[alloc_index] = pKFi->GetAccBias().cast<FP>();
                acc_bias_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 3, &acc_biases[alloc_index], bFixed);
            }
        }
        alloc_index++;
    }

    if (bInit)
    {
        std::cout << "Creating biases for bInit=true" << std::endl;
        // VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        // VG->setId(4*maxKFid+2);
        // VG->setFixed(false);
        // optimizer.addVertex(VG);
        gyro_biases[alloc_index] = pIncKF->GetGyroBias().cast<FP>();
        gyro_bias_desc.add_vertex(4 * maxKFid + 2, &gyro_biases[alloc_index], false);

        // VertexAccBias* VA = new VertexAccBias(pIncKF);
        // VA->setId(4*maxKFid+3);
        // VA->setFixed(false);
        // optimizer.addVertex(VA);

        acc_biases[alloc_index] = pIncKF->GetAccBias().cast<FP>();
        acc_bias_desc.add_vertex(4 * maxKFid + 3, &acc_biases[alloc_index], false);

        alloc_index++;
    }

    if(bFixLocal)
    {
        if(nNonFixed<3) {
            cleanup_cameras();
            return;
        }
    }

    const size_t N = vpKFs.size();
    using PoseDescriptorType = decltype(pose_desc);
    auto ic_desc_rk = InertialConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 9>, PoseDescriptorType>
    (&pose_desc, &velocity_desc, &gyro_bias_desc, &acc_bias_desc, &pose_desc, &velocity_desc); // For robust kernel
    ic_desc_rk.reserve(N);
    graph.add_factor_descriptor(&ic_desc_rk);

    // auto ic_desc = InertialConstraintDescriptor<FP, SP, 
    //     graphite::DefaultLoss<FP, 9>, decltype(pose_desc)>(&pose_desc, &velocity_desc, 
    //         &gyro_bias_desc, &acc_bias_desc, &pose_desc, &velocity_desc);
    // ic_desc.reserve(N);

    auto gc_desc = GyroRWConstraintDescriptor<FP, SP, graphite::DefaultLoss<FP, 3>>(&gyro_bias_desc, &gyro_bias_desc);
    gc_desc.reserve(N);

    auto ac_desc = AccRWConstraintDescriptor<FP, SP, graphite::DefaultLoss<FP, 3>>(&acc_bias_desc, &acc_bias_desc);
    ac_desc.reserve(N);

    // These constraints below are only added when !bInit (see the IMU link
    // loop), so registering them during initialization leaves two descriptors with
    // zero factors.
    if (!bInit)
    {
        graph.add_factor_descriptor(&gc_desc);
        graph.add_factor_descriptor(&ac_desc);
    }

    std::cout << "Creating IMU links for FullInertialBA" << std::endl;
    // IMU links
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
            if(pKFi->bImu && pKFi->mPrevKF->bImu)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                // g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                // g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
                const size_t VP1 = pKFi->mPrevKF->mnId;
                const size_t VV1 = maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1;

                // g2o::HyperGraph::Vertex* VG1;
                // g2o::HyperGraph::Vertex* VA1;
                // g2o::HyperGraph::Vertex* VG2;
                // g2o::HyperGraph::Vertex* VA2;
                size_t VG1 = 0;
                size_t VA1 = 0;
                size_t VG2 = 0;
                size_t VA2 = 0;
                if (!bInit)
                {
                    // VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
                    // VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
                    // VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                    // VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

                    VG1 = maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2;
                    VA1 = maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3;
                    VG2 = maxKFid + 3 * (pKFi->mnId) + 2;
                    VA2 = maxKFid + 3 * (pKFi->mnId) + 3;
                }
                else
                {
                    // VG1 = optimizer.vertex(4*maxKFid+2);
                    // VA1 = optimizer.vertex(4*maxKFid+3);
                    VG1 = 4 * maxKFid + 2;
                    VA1 = 4 * maxKFid + 3;
                }

                // g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                // g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
                const size_t VP2 = pKFi->mnId;
                const size_t VV2 = maxKFid + 3 * (pKFi->mnId) + 1;

                if (!bInit)
                {
                    // if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                    if (!pose_desc.exists(VP1) || !velocity_desc.exists(VV1) || 
                        !gyro_bias_desc.exists(VG1) || !acc_bias_desc.exists(VA1) || 
                        !pose_desc.exists(VP2) || !velocity_desc.exists(VV2) || 
                        !gyro_bias_desc.exists(VG2) || !acc_bias_desc.exists(VA2))
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                        continue;
                    }
                }
                else
                {
                    // if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                    if (!pose_desc.exists(VP1) || !velocity_desc.exists(VV1) || 
                        !gyro_bias_desc.exists(VG1) || !acc_bias_desc.exists(VA1) || 
                        !pose_desc.exists(VP2) || !velocity_desc.exists(VV2))
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;
                        continue;
                    }
                }

                // EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                // ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                // ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                // ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                // ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                // ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                // ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                // g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                // ei->setRobustKernel(rki);
                // rki->setDelta(sqrt(16.92));

                const InertialConstraintData<FP> ic_data(pKFi->mpImuPreintegrated);
                const auto info = ic_data.template get_information_matrix<SP>(pKFi->mpImuPreintegrated);

                ic_desc_rk.add_factor({VP1, VV1, VG1, VA1, VP2, VV2}, Empty(),
                                        info.data(), ic_data, graphite::HuberLoss<FP, 9>(sqrt(16.92)));

                // optimizer.addEdge(ei);

                if (!bInit)
                {
                    // EdgeGyroRW* egr= new EdgeGyroRW();
                    // egr->setVertex(0,VG1);
                    // egr->setVertex(1,VG2);
                    // Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
                    // egr->setInformation(InfoG);
                    // egr->computeError();
                    // optimizer.addEdge(egr);
                    Mat3<SP> InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse().cast<SP>();
                    gc_desc.add_factor({VG1, VG2}, Empty(), InfoG.data());

                    // EdgeAccRW* ear = new EdgeAccRW();
                    // ear->setVertex(0,VA1);
                    // ear->setVertex(1,VA2);
                    // Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
                    // ear->setInformation(InfoA);
                    // ear->computeError();
                    // optimizer.addEdge(ear);

                    Mat3<SP> InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse().cast<SP>();
                    ac_desc.add_factor({VA1, VA2}, Empty(), InfoA.data());
                }
            }
            else
                cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
        }
    }

    auto gyro_prior_desc = GyroRWPriorDescriptor<FP, SP, DefaultLoss<FP, 3>>(&gyro_bias_desc);
    auto acc_prior_desc = AccRWPriorDescriptor<FP, SP, DefaultLoss<FP, 3>>(&acc_bias_desc);
    if (bInit)
    {
        std::cout << "Creating priors for FullInertialBA" << std::endl;
        // g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);
        // g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);

        const size_t VG = 4 * maxKFid + 2;
        const size_t VA = 4 * maxKFid + 3;

        // Add prior to comon biases
        // Eigen::Vector3f bprior;
        Vec3<FP> bprior;
        bprior.setZero();

        // EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        // epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        // double infoPriorA = priorA; //
        // epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        // optimizer.addEdge(epa);

        Mat3<SP> infoPriorA = (priorA * Eigen::Matrix3d::Identity()).cast<SP>();
        acc_prior_desc.add_factor({VA}, bprior, infoPriorA.data());

        // EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        // epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        // double infoPriorG = priorG; //
        // epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        // optimizer.addEdge(epg);
        Mat3<SP> infoPriorG = (priorG * Eigen::Matrix3d::Identity()).cast<SP>();
        gyro_prior_desc.add_factor({VG}, bprior, infoPriorG.data());

        // Register prior descriptors
        graph.add_factor_descriptor(&gyro_prior_desc);
        graph.add_factor_descriptor(&acc_prior_desc);
    }

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid*5;

    vector<bool> vbNotIncludedMP(vpMPs.size(),false);

    // Create storage for reprojection errors
    graphite::managed_vector<SBAPointXYZ<FP>> map_points(vpMPs.size());

    auto mp_desc = SBAPointXYZDescriptor<FP, SP>();
    mp_desc.set_eliminate(true);
    graph.add_vertex_descriptor(&mp_desc);

    // const size_t nExpectedSize = vpMPs.size() * vpKFs.size(); // very bad
    const size_t nExpectedSize = 200 * vpKFs.size(); // guess 200 MPs per KF

    auto stereo_desc = StereoConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 3>, Camera>(&mp_desc, &pose_desc);
    stereo_desc.reserve(nExpectedSize);

    auto mono_desc = MonoConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 2>, Camera>(&mp_desc, &pose_desc);
    mono_desc.reserve(nExpectedSize);

    size_t stereo_count = 0;
    size_t mono_count = 0;

    std::cout << "Creating reprojection constraints for FullInertialBA" << std::endl;
    // Add reprojection constraints
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        // g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        // vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId+iniMPid+1;
        // vPoint->setId(id);
        // vPoint->setMarginalized(true);
        // optimizer.addVertex(vPoint);

        map_points[i] = pMP->GetWorldPos().cast<FP>();
        mp_desc.add_vertex(id, &map_points[i], false);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();


        bool bAllFixed = true;

        //Set edges
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnId>maxKFid)
                continue;

            if(!pKFi->isBad())
            {
                const int leftIndex = get<0>(mit->second);
                cv::KeyPoint kpUn;

                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    // Eigen::Matrix<double,2,1> obs;
                    Vec2<FP> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // EdgeMono* e = new EdgeMono(0);
                    // g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        // if(!VP->fixed())
                        if(!pose_desc.is_fixed(pKFi->mnId))
                            bAllFixed=false;

                    // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    // e->setVertex(1, VP);
                    // e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    Mat2<SP> info = Mat2<SP>::Identity() * invSigma2;
                    const auto f_id = mono_desc.add_factor({id, pKFi->mnId}, obs, info.data(), 0, graphite::HuberLoss<FP, 2>(thHuberMono));

                    // e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    // e->setRobustKernel(rk);
                    // rk->setDelta(thHuberMono);

                    // optimizer.addEdge(e);
                    mono_count++;
                }
                else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    // Eigen::Matrix<double,3,1> obs;
                    Vec3<FP> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // EdgeStereo* e = new EdgeStereo(0);

                    // g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        // if(!VP->fixed())
                        if(!pose_desc.is_fixed(pKFi->mnId))
                            bAllFixed=false;

                    // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    // e->setVertex(1, VP);
                    // e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Mat3<SP> info = Mat3<SP>::Identity() * invSigma2;
                    const auto f_id = stereo_desc.add_factor({id, pKFi->mnId}, obs, info.data(), 0, graphite::HuberLoss<FP, 3>(thHuberStereo));

                    // e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    // e->setRobustKernel(rk);
                    // rk->setDelta(thHuberStereo);

                    // optimizer.addEdge(e);
                    stereo_count++;
                }

                if(pKFi->mpCamera2){ // Monocular right observation
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size()){
                        rightIndex -= pKFi->NLeft;

                        // Eigen::Matrix<double,2,1> obs;
                        Vec2<FP> obs;
                        kpUn = pKFi->mvKeysRight[rightIndex];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        // EdgeMono *e = new EdgeMono(1);

                        // g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            // if(!VP->fixed())
                            if (!pose_desc.is_fixed(pKFi->mnId))
                                bAllFixed=false;

                        // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        // e->setVertex(1, VP);
                        // e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        // e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                        Mat2<SP> info = Mat2<SP>::Identity() * invSigma2;
                        const auto f_id = mono_desc.add_factor({id, pKFi->mnId}, obs, info.data(), 1, graphite::HuberLoss<FP, 2>(thHuberMono));

                        // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        // e->setRobustKernel(rk);
                        // rk->setDelta(thHuberMono);

                        // optimizer.addEdge(e);
                        mono_count++;
                    }
                }
            }
        }

        if(bAllFixed)
        {
            // std::cout << "All fixed, removing vertex " << id << std::endl;
            // optimizer.removeVertex(vPoint);
            mp_desc.remove_vertex(id);
            // std::cout << "Removed vertex " << id << std::endl;
            vbNotIncludedMP[i]=true;
        }
    }

    if (stereo_count > 0) {
        graph.add_factor_descriptor(&stereo_desc);
    }
    if (mono_count > 0) {
        graph.add_factor_descriptor(&mono_desc);
    }

    if(pbStopFlag)
        if(*pbStopFlag) {
            cleanup_cameras();
            return;
        }


    // optimizer.initializeOptimization();
    // optimizer.optimize(its);

    optimizer::LevenbergMarquardtOptions<FP, SP> options;
    options.iterations = its;
    options.initial_damping = lambda;
    options.optimization_level = optimization_level;
    options.streams = &streams;
    options.stop_flag = pbStopFlag;
    options.solver = solver.get();
    options.verbose = true;
    options.use_identity = true;

    optimizer::levenberg_marquardt2<FP, SP>(&graph, &options);
    
    // print graph info
    // print counts of vds
    std::cout << "Pose vertices: " << pose_desc.count() << std::endl;
    std::cout << "Velocity vertices: " << velocity_desc.count() << std::endl;
    std::cout << "Gyro bias vertices: " << gyro_bias_desc.count() << std::endl;
    std::cout << "Acc bias vertices: " << acc_bias_desc.count() << std::endl;
    std::cout << "Point vertices: " << mp_desc.count() << std::endl;

    // print total number of active constraints
    size_t total_constraints = 0;
    if (stereo_count > 0) {
        total_constraints += stereo_desc.active_count();
        std::cout << "Stereo constraints: " << stereo_desc.active_count() << std::endl;
    }
    if (mono_count > 0) {
        total_constraints += mono_desc.active_count();
        std::cout << "Mono constraints: " << mono_desc.active_count() << std::endl;
    }
    total_constraints += ic_desc_rk.active_count();
    std::cout << "Inertial constraints: " << ic_desc_rk.active_count() << std::endl;

    total_constraints += gc_desc.active_count();
    std::cout << "Gyro RW constraints: " << gc_desc.active_count() << std::endl;
    total_constraints += ac_desc.active_count();
    std::cout << "Acc RW constraints: " << ac_desc.active_count() << std::endl;

    total_constraints += gyro_prior_desc.active_count();
    std::cout << "Gyro prior constraints: " << gyro_prior_desc.active_count() << std::endl;
    total_constraints += acc_prior_desc.active_count();
    std::cout << "Acc prior constraints: " << acc_prior_desc.active_count() << std::endl;

    std::cout << "Total constraints: " << total_constraints << std::endl;


    if constexpr (skip_recovery) {
        std::cout << "Exiting early before recovering optimized data!" << std::endl;
        cleanup_cameras();
        return;
    }

    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        // VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        const auto pose = pose_desc.get_vertex(pKFi->mnId);
        if(nLoopId==0)
        {
            // Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
            Sophus::SE3f Tcw(pose->Rcw[0].template cast<float>(), pose->tcw[0].template cast<float>());
            pKFi->SetPose(Tcw);
        }
        else
        {
            // pKFi->mTcwGBA = Sophus::SE3f(VP->estimate().Rcw[0].cast<float>(),VP->estimate().tcw[0].cast<float>());
            pKFi->mTcwGBA = Sophus::SE3f(pose->Rcw[0].template cast<float>(), pose->tcw[0].template cast<float>());
            pKFi->mnBAGlobalForKF = nLoopId;

        }
        if(pKFi->bImu)
        {
            // VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            const auto VV = velocity_desc.get_vertex(maxKFid + 3 * (pKFi->mnId) + 1);
            if(nLoopId==0)
            {
                // pKFi->SetVelocity(VV->estimate().cast<float>());
                pKFi->SetVelocity(VV->template cast<float>());
            }
            else
            {
                // pKFi->mVwbGBA = VV->estimate().cast<float>();
                pKFi->mVwbGBA = VV->template cast<float>();

            }

            // VertexGyroBias* VG;
            // VertexAccBias* VA;
            const GyroBias<FP>* VG;
            const AccBias<FP>* VA;
            if (!bInit)
            {
                // VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                // VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));

                VG = gyro_bias_desc.get_vertex(maxKFid + 3 * (pKFi->mnId) + 2);
                VA = acc_bias_desc.get_vertex(maxKFid + 3 * (pKFi->mnId) + 3);
            }
            else
            {
                // VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                // VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));

                VG = gyro_bias_desc.get_vertex(4 * maxKFid + 2);
                VA = acc_bias_desc.get_vertex(4 * maxKFid + 3);
            }

            // Vector6d vb;
            // vb << VG->estimate(), VA->estimate();
            Eigen::Matrix<double, 6, 1> vb;
            vb << VG->template cast<double>(), VA->template cast<double>();
            IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
            if(nLoopId==0)
            {
                pKFi->SetNewBias(b);
            }
            else
            {
                pKFi->mBiasGBA = b;
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMPs[i];
        // g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        const auto vPoint = mp_desc.get_vertex(pMP->mnId + iniMPid + 1);
        if(nLoopId==0)
        {
            // pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->SetWorldPos(vPoint->template cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            // pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mPosGBA = vPoint->template cast<float>();
            pMP->mnBAGlobalForKF = nLoopId;
        }

    }

    pMap->IncreaseChangeIndex();
    cleanup_cameras();
}

void FullInertialBA(bool use_pcg, Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess) {
    using namespace gpu;
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    KeyFrame* pKF = vpKFs.front();
    if (pKF->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        OptimizerGPU::FullInertialBAInternal<double, double, PinholeCamera<double>, false>(use_pcg, pMap, its, bFixLocal, nLoopId, pbStopFlag, bInit, priorG, priorA, vSingVal, bHess);
    }
    else {
        OptimizerGPU::FullInertialBAInternal<double, double, KannalaBrandt8Camera<double>, false>(use_pcg, pMap, its, bFixLocal, nLoopId, pbStopFlag, bInit, priorG, priorA, vSingVal, bHess);
    }
}

}
}