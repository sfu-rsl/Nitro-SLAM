#pragma once
#include "Optimizer.h"

#include <graphite/vector.hpp>
#include <graphite/loss.hpp>
// #include "GPUPose.h"
#include "GPUTypes.h"
// #include "PGOTypes.h"
#include <graphite/solver/eigen_schur.hpp>
#include <graphite/optimizer/levenberg_marquardt.hpp>



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



template <typename Camera>
void LocalInertialBAInternal(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, bool bLarge, bool bRecInit)
{
    using namespace graphite;
    using namespace gpu;
    Map* pCurrentMap = pKF->GetMap();

    int maxOpt=10;
    int opt_it=10;
    if(bLarge)
    {
        maxOpt=25;
        opt_it=4;
    }
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2,maxOpt);
    const unsigned long maxKFid = pKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    list<KeyFrame*> lpOptVisKFs;

    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    const int maxCovKF = 0;
    for(int i=0, iend=vpNeighsKFs.size(); i<iend; i++)
    {
        if(lpOptVisKFs.size() >= maxCovKF)
            break;

        KeyFrame* pKFi = vpNeighsKFs[i];
        if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);

            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    const int maxFixKF = 200;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if(lFixedKeyFrames.size()>=maxFixKF)
            break;
    }

    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup optimizer
    /*
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    */

    using FP = double;
    using SP = double;
    // using FP = float;
    // using SP = float;
    // using SP = __nv_bfloat16;

    Graph<FP, SP> graph;

    // BlockJacobiPreconditioner<FP, SP> preconditioner;
    // IdentityPreconditioner<FP, SP> preconditioner;
    // PCGSolver<FP, SP> solver(10, 1e-1, 5.0, &preconditioner);
    EigenSchurLDLTSolver<FP, SP> solver;
    StreamPool streams(7);
    constexpr uint8_t optimization_level = 0;
    double lambda = 1e0;

    

    if(bLarge)
    {
        // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        // solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
        // optimizer.setAlgorithm(solver);
        lambda = 1e-2;
    }
    else
    {
        // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        // solver->setUserLambdaInit(1e0);
        // optimizer.setAlgorithm(solver);
        lambda = 1e0;
    }


    // Set Local temporal KeyFrame vertices

    // TODO: Find a better way to allocate these vertices
    // Simplify this and assume stereo cameras only with only pinhole model
    // using Camera = PinholeCamera<FP>;
    // using Camera = KannalaBrandt8Camera<FP>;
    constexpr size_t max_cameras = 2;
    using Pose = ImuCamPose<FP, Camera>;

    const auto num_keyframes_total = (vpOptimizableKFs.size() + lpOptVisKFs.size()
                                                                + lFixedKeyFrames.size());
    graphite::managed_vector<Pose> poses(num_keyframes_total);
    graphite::managed_vector<Velocity<FP>> velocities(num_keyframes_total);
    graphite::managed_vector<GyroBias<FP>> gyro_biases(num_keyframes_total);
    graphite::managed_vector<AccBias<FP>> acc_biases(num_keyframes_total);
    // graphite::managed_vector<Camera> cameras(num_keyframes_total*2);

    std::unordered_map<GeometricCamera*, Camera*>  cameras;
    // std::unordered_map<long unsigned int, Pose> pose_map;

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

    
    /*
    auto get_cameras = [&cameras](KeyFrame* pKFi) -> std::array<Camera*, max_cameras> {
        std::array<Camera*, max_cameras> cams = {nullptr, nullptr};
        if (cameras.find(pKFi->mpCamera) == cameras.end()) {
            Camera* cam;
            cudaMallocManaged(&cam, sizeof(Camera));
            // Initialize the camera with placement new
            std::array<FP, 4> cam_params;
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

                std::array<FP, 4> cam_params;
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
    */


    size_t alloc_index = 0;

    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        // VertexPose * VP = new VertexPose(pKFi);
        // VP->setId(pKFi->mnId);
        // VP->setFixed(false);
        // optimizer.addVertex(VP);

        auto cams = get_cameras<FP, Camera, max_cameras>(pKFi, cameras);
        poses[alloc_index] = Pose(pKFi, cams.data());
        pose_desc.add_vertex(pKFi->mnId, &poses[alloc_index], false);

        if(pKFi->bImu)
        {
            // VertexVelocity* VV = new VertexVelocity(pKFi);
            // VV->setId(maxKFid+3*(pKFi->mnId)+1);
            // VV->setFixed(false);
            // optimizer.addVertex(VV);

            velocities[alloc_index] = pKFi->GetVelocity().cast<FP>();
            velocity_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 1, &velocities[alloc_index], false);

            // VertexGyroBias* VG = new VertexGyroBias(pKFi);
            // VG->setId(maxKFid+3*(pKFi->mnId)+2);
            // VG->setFixed(false);
            // optimizer.addVertex(VG);

            gyro_biases[alloc_index] = pKFi->GetGyroBias().cast<FP>();
            gyro_bias_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 2, &gyro_biases[alloc_index], false);


            // VertexAccBias* VA = new VertexAccBias(pKFi);
            // VA->setId(maxKFid+3*(pKFi->mnId)+3);
            // VA->setFixed(false);
            // optimizer.addVertex(VA);

            acc_biases[alloc_index] = pKFi->GetAccBias().cast<FP>();
            acc_bias_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 3, &acc_biases[alloc_index], false);
        }
        alloc_index++;
    }

    // Set Local visual KeyFrame vertices
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        // KeyFrame* pKFi = *it;
        // VertexPose * VP = new VertexPose(pKFi);
        // VP->setId(pKFi->mnId);
        // VP->setFixed(false);
        // optimizer.addVertex(VP);

        KeyFrame* pKFi = *it;
        auto cams = get_cameras<FP, Camera, max_cameras>(pKFi, cameras);
        poses[alloc_index] = Pose(pKFi, cams.data());
        pose_desc.add_vertex(pKFi->mnId, &poses[alloc_index], false);
        alloc_index++;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        // KeyFrame* pKFi = *lit;
        // VertexPose * VP = new VertexPose(pKFi);
        // VP->setId(pKFi->mnId);
        // VP->setFixed(true);
        // optimizer.addVertex(VP);


        KeyFrame* pKFi = *lit;
        auto cams = get_cameras<FP, Camera, max_cameras>(pKFi, cameras);
        poses[alloc_index] = Pose(pKFi, cams.data());
        pose_desc.add_vertex(pKFi->mnId, &poses[alloc_index], true);

        if(pKFi->bImu) // This should be done only for keyframe just before temporal window
        {
            // VertexVelocity* VV = new VertexVelocity(pKFi);
            // VV->setId(maxKFid+3*(pKFi->mnId)+1);
            // VV->setFixed(true);
            // optimizer.addVertex(VV);

            velocities[alloc_index] = pKFi->GetVelocity().cast<FP>();
            velocity_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 1, &velocities[alloc_index], true);

            // VertexGyroBias* VG = new VertexGyroBias(pKFi);
            // VG->setId(maxKFid+3*(pKFi->mnId)+2);
            // VG->setFixed(true);
            // optimizer.addVertex(VG);

            gyro_biases[alloc_index] = pKFi->GetGyroBias().cast<FP>();
            gyro_bias_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 2, &gyro_biases[alloc_index], true);

            // VertexAccBias* VA = new VertexAccBias(pKFi);
            // VA->setId(maxKFid+3*(pKFi->mnId)+3);
            // VA->setFixed(true);
            // optimizer.addVertex(VA);

            acc_biases[alloc_index] = pKFi->GetAccBias().cast<FP>();
            acc_bias_desc.add_vertex(maxKFid + 3 * (pKFi->mnId) + 3, &acc_biases[alloc_index], true);

        }

        alloc_index++;
    }

    // Create intertial constraints
    // vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    // vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    // vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);

    using PoseDescriptorType = decltype(pose_desc);

    auto ic_desc_rk = InertialConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 9>, PoseDescriptorType>
    (&pose_desc, &velocity_desc, &gyro_bias_desc, &acc_bias_desc, &pose_desc, &velocity_desc); // For robust kernel
    // ic_desc.reserve(N);
    // graph.add_factor_descriptor(&ic_desc_rk);

    auto ic_desc = InertialConstraintDescriptor<FP, SP, 
        graphite::DefaultLoss<FP, 9>, decltype(pose_desc)>(&pose_desc, &velocity_desc, 
            &gyro_bias_desc, &acc_bias_desc, &pose_desc, &velocity_desc);
    ic_desc.reserve(N);
    // graph.add_factor_descriptor(&ic_desc);

    auto gc_desc = GyroRWConstraintDescriptor<FP, SP, graphite::DefaultLoss<FP, 3>>(&gyro_bias_desc, &gyro_bias_desc);
    gc_desc.reserve(N);
    graph.add_factor_descriptor(&gc_desc);

    auto ac_desc = AccRWConstraintDescriptor<FP, SP, graphite::DefaultLoss<FP, 3>>(&acc_bias_desc, &acc_bias_desc);
    ac_desc.reserve(N);
    graph.add_factor_descriptor(&ac_desc);

    size_t icr_count = 0;
    size_t ic_count = 0;
    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            // g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            // g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            // g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            // g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            // g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            // g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            // g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            // g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            const size_t VP1 = pKFi->mPrevKF->mnId;
            const size_t VV1 = maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1;
            const size_t VG1 = maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2;
            const size_t VA1 = maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3;
            const size_t VP2 = pKFi->mnId;
            const size_t VV2 = maxKFid + 3 * (pKFi->mnId) + 1;
            const size_t VG2 = maxKFid + 3 * (pKFi->mnId) + 2;
            const size_t VA2 = maxKFid + 3 * (pKFi->mnId) + 3;

            // if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            // {
            //     cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
            //     continue;
            // }

            if (!pose_desc.exists(VP1) || !pose_desc.exists(VP2) ||
                !velocity_desc.exists(VV1) || !velocity_desc.exists(VV2) ||
                !gyro_bias_desc.exists(VG1) || !gyro_bias_desc.exists(VG2) ||
                !acc_bias_desc.exists(VA1) || !acc_bias_desc.exists(VA2)) {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                cerr << "Error: Vertex not found for inertial edge." << endl;
                continue;
            }

            // vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            // vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            // vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            // vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            // vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            // vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            // vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            InertialConstraintData<FP> ic_data(pKFi->mpImuPreintegrated);
            auto info = ic_data.get_information_matrix<SP>(pKFi->mpImuPreintegrated);

            if(i==N-1 || bRecInit)
            {
                // All inertial residuals are included without robust cost function, but not that one linking the
                // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                // error due to fixing variables.
                /*
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if(i==N-1)
                    vei[i]->setInformation(vei[i]->information()*1e-2);
                rki->setDelta(sqrt(16.92));
                */

                if (i == N-1) {
                    // Downweight the last inertial edge
                    info *= 1e-2;
                }
                ic_desc_rk.add_factor({VP1, VV1, VG1, VA1, VP2, VV2}, Empty(), info.data(), ic_data, graphite::HuberLoss<FP, 9>(sqrt(16.92)));
                icr_count++;
            }
            else {
                ic_desc.add_factor({VP1, VV1, VG1, VA1, VP2, VV2}, Empty(), info.data(), ic_data, graphite::DefaultLoss<FP, 9>());
                ic_count++;
            }
            // optimizer.addEdge(vei[i]);

            // vegr[i] = new EdgeGyroRW();
            // vegr[i]->setVertex(0,VG1);
            // vegr[i]->setVertex(1,VG2);
            // Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            // vegr[i]->setInformation(InfoG);
            // optimizer.addEdge(vegr[i]);

            Mat3<SP> InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse().cast<SP>();
            gc_desc.add_factor({VG1, VG2}, Empty(), InfoG.data(), Empty(), graphite::DefaultLoss<FP, 3>());


            // vear[i] = new EdgeAccRW();
            // vear[i]->setVertex(0,VA1);
            // vear[i]->setVertex(1,VA2);
            // Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            // vear[i]->setInformation(InfoA);           
            // optimizer.addEdge(vear[i]);
            Mat3<SP> InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse().cast<SP>();
            ac_desc.add_factor({VA1, VA2}, Empty(), InfoA.data(), Empty(), graphite::DefaultLoss<FP, 3>());

        }
        else
            cout << "ERROR building inertial edge" << endl;
    }
    if (icr_count > 0) {
        graph.add_factor_descriptor(&ic_desc_rk);
    }
    if (ic_count > 0) {
        graph.add_factor_descriptor(&ic_desc);
    }

    // Set MapPoint vertices
    const int nExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    // vector<EdgeMono*> vpEdgesMono;
    // vpEdgesMono.reserve(nExpectedSize);
    vector<size_t> mono_ids;
    mono_ids.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    // vector<EdgeStereo*> vpEdgesStereo;
    // vpEdgesStereo.reserve(nExpectedSize);
    vector<size_t> stereo_ids;
    stereo_ids.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    graphite::managed_vector<SBAPointXYZ<FP>> map_points(lLocalMapPoints.size());
    
    auto mp_desc = SBAPointXYZDescriptor<FP, SP>();
    mp_desc.set_eliminate(true);
    mp_desc.reserve(lLocalMapPoints.size());
    graph.add_vertex_descriptor(&mp_desc);

    auto stereo_desc = StereoConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 3>, Camera>(&mp_desc, &pose_desc);
    stereo_desc.reserve(nExpectedSize);

    auto mono_desc = MonoConstraintDescriptor<FP, SP, graphite::HuberLoss<FP, 2>, Camera>(&mp_desc, &pose_desc);
    mono_desc.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    map<int,int> mVisEdges;
    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        mVisEdges[pKFi->mnId] = 0;
    }
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        mVisEdges[(*lit)->mnId] = 0;
    }

    size_t mp_index = 0;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        // g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        // vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        map_points[mp_index] = pMP->GetWorldPos().cast<FP>();


        unsigned long id = pMP->mnId+iniMPid+1;
        // vPoint->setId(id);
        // vPoint->setMarginalized(true);
        // optimizer.addVertex(vPoint);

        mp_desc.add_vertex(id, &map_points[mp_index], false);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                continue;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                cv::KeyPoint kpUn;

                // Monocular left observation
                if(leftIndex != -1 && pKFi->mvuRight[leftIndex]<0)
                {
                    
                    mVisEdges[pKFi->mnId]++;

                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<FP,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // EdgeMono* e = new EdgeMono(0);

                    // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    // e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    // e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.cast<double>());

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    // e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    Mat2<SP> info = Mat2<SP>::Identity() * invSigma2;
                    const auto f_id = mono_desc.add_factor({id, pKFi->mnId}, obs, info.data(), 0, graphite::HuberLoss<FP, 2>(thHuberMono));


                    // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    // e->setRobustKernel(rk);
                    // rk->setDelta(thHuberMono);

                    // optimizer.addEdge(e);
                    // vpEdgesMono.push_back(e);
                    mono_ids.push_back(f_id);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                    
                }
                // Stereo-observation
                else if(leftIndex != -1)// Stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    mVisEdges[pKFi->mnId]++;

                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Vec3<FP> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // EdgeStereo* e = new EdgeStereo(0);

                    // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    // e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    // e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2).cast<double>());

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    // e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);
                    Mat3<SP> info = Mat3<SP>::Identity() * invSigma2;

                    // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    // e->setRobustKernel(rk);
                    // rk->setDelta(thHuberStereo);


                    auto f_id = stereo_desc.add_factor({id, pKFi->mnId}, obs, info.data(), 0, graphite::HuberLoss<FP, 3>(thHuberStereo));


                    // optimizer.addEdge(e);
                    // vpEdgesStereo.push_back(e);
                    stereo_ids.push_back(f_id);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                // Monocular right observation
                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;
                        mVisEdges[pKFi->mnId]++;

                        Eigen::Matrix<FP,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        // EdgeMono* e = new EdgeMono(1);

                        // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        // e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        // e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs.cast<double>());

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                        // e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        Mat2<SP> info = Mat2<SP>::Identity() * invSigma2;
                        const auto f_id = mono_desc.add_factor({id, pKFi->mnId}, obs, info.data(), 1, graphite::HuberLoss<FP, 2>(thHuberMono));
                        // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        // e->setRobustKernel(rk);
                        // rk->setDelta(thHuberMono);

                        // optimizer.addEdge(e);
                        // vpEdgesMono.push_back(e);
                        mono_ids.push_back(f_id);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                }
            }
        }
        mp_index++;
    }

    if (stereo_ids.size() > 0) {
        graph.add_factor_descriptor(&stereo_desc);
    }
    if (mono_ids.size() > 0) {
        graph.add_factor_descriptor(&mono_desc);
    }

    //cout << "Total map points: " << lLocalMapPoints.size() << endl;
    for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
    {
        assert(mit->second>=3);
    }

    // optimizer.initializeOptimization();
    // optimizer.computeActiveErrors();
    // float err = optimizer.activeRobustChi2();
    // optimizer.optimize(opt_it); // Originally to 2
    optimizer::LevenbergMarquardtOptions<FP, SP> options;
    options.solver = &solver;
    options.iterations = opt_it;
    options.initial_damping =  lambda;
    options.optimization_level = optimization_level;
    options.streams = &streams;
    options.stop_flag = pbStopFlag;
    options.verbose = false;

    // std::cout << "LIBA OPTIMIZING!" << std::endl;
    optimizer::levenberg_marquardt2<FP, SP>(&graph, &options);
    // std::cout << "LIBA OPTIMIZATION DONE!" << std::endl;
    // float err_end = optimizer.activeRobustChi2();
    // if(pbStopFlag)
    //     optimizer.setForceStopFlag(pbStopFlag);

    // return early for testing
    // return;

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    // vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    // vToErase.reserve(vpEdgesStereo.size());
    vToErase.reserve(stereo_ids.size()+mono_ids.size());

    // Check inlier observations
    // Mono

    // for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    for (size_t i=0, iend=mono_ids.size(); i<iend;i++)
    {
        // EdgeMono* e = vpEdgesMono[i];
        const auto f_id = mono_ids[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        bool bClose = pMP->mTrackDepth<10.f;

        if(pMP->isBad())
            continue;

        // TODO: Reimplement isdepthpositive
        const auto cam_idx = *mono_desc.get_constraint_data(f_id);
        const auto ids = mono_desc.get_vertex_ids(f_id);

        const auto mp_vec = *mp_desc.get_vertex(ids[0]);
        const auto pose = *pose_desc.get_vertex(ids[1]);

        const auto mchi2 = mono_desc.chi2(f_id);
        if((mchi2>chi2Mono2 && !bClose) || (mchi2>1.5f*chi2Mono2 && bClose) 
        || !pose.isDepthPositive(mp_vec, cam_idx))
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }



    // Stereo
    // for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    for(size_t i=0, iend=stereo_ids.size(); i<iend;i++)
    {
        // EdgeStereo* e = vpEdgesStereo[i];
        const auto f_id = stereo_ids[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        // if(e->chi2()>chi2Stereo2)
        if (stereo_desc.chi2(f_id) > chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);


    // TODO: Some convergence problems have been detected here
    // if((2*err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
    // {
    //     cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
    //     return;
    // }
    // TODO: Need to reimplement this - might be related to a bug that was already fixed



    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // Recover optimized data
    // Local temporal Keyframes
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        // VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        const Pose* pose = pose_desc.get_vertex(pKFi->mnId);
        // Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        Sophus::SE3f Tcw(pose->Rcw[0].template cast<float>(), pose->tcw[0].template cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;

        if(pKFi->bImu)
        {
            // VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            const auto VV = velocity_desc.get_vertex(maxKFid + 3 * (pKFi->mnId) + 1);
            // pKFi->SetVelocity(VV->estimate().cast<float>());
            pKFi->SetVelocity(VV->cast<float>());
            // VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            // VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            const auto VG = gyro_bias_desc.get_vertex(maxKFid + 3 * (pKFi->mnId) + 2);
            const auto VA = acc_bias_desc.get_vertex(maxKFid + 3 * (pKFi->mnId) + 3);
            // Vector6d b;
            Eigen::Matrix<double, 6, 1> b;
            // b << VG->estimate(), VA->estimate();
            b << VG->cast<double>(), VA->cast<double>();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));

        }
    }

    // Local visual KeyFrame
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        // VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        // Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        const auto VP = pose_desc.get_vertex(pKFi->mnId);
        Sophus::SE3f Tcw(VP->Rcw[0].template cast<float>(), VP->tcw[0].template cast<float>());

        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        // g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        const auto vPoint = mp_desc.get_vertex(pMP->mnId + iniMPid + 1);

        // pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->SetWorldPos(vPoint->cast<float>());

        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();

    // Deallocate cameras
    for (auto& [cam_ptr, cam] : cameras) {
        if (cam) {
            cudaFree(cam);
        }
    }

}



void LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, bool bLarge, bool bRecInit) {
    using namespace gpu;
    if (pKF->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
        OptimizerGPU::LocalInertialBAInternal<PinholeCamera<double>>(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges, bLarge, bRecInit);
    }
    else {
        OptimizerGPU::LocalInertialBAInternal<KannalaBrandt8Camera<double>>(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges, bLarge, bRecInit);
    }
}

}

}