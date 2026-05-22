#include <iostream>
#include "Kernels/FuseKernel.h"
#include "Kernels/MappingKernelController.h"

void FuseKernel::initialize() {
    if (memory_is_initialized)
        return;

    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize, neighborCount;

    if (CudaUtils::cameraIsFisheye) {
        mapPointVecSize = maxFeatures*2;
        neighborCount = MAX_NEIGHBOR_KF_COUNT*2;
    }
    else {
        mapPointVecSize = maxFeatures;
        neighborCount = MAX_NEIGHBOR_KF_COUNT;
    }

    checkCudaError(cudaMalloc((void**)&d_currKFMapPoints, MAX_NEIGHBOR_KF_COUNT * mapPointVecSize * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "Failed to allocate memory for d_currKFMapPoints");
    checkCudaError(cudaMalloc((void**)&d_neighKFs, neighborCount * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*)), "Failed to allocate memory for d_neighKFs");
    checkCudaError(cudaMalloc((void**)&d_Tcw, neighborCount * sizeof(Sophus::SE3f)), "Failed to allocate memory for d_Tcw");
    checkCudaError(cudaMalloc((void**)&d_TcwRight, neighborCount * sizeof(Sophus::SE3f)), "Failed to allocate memory for d_TcwRight");
    checkCudaError(cudaMalloc((void**)&d_Ow, neighborCount * sizeof(Eigen::Vector3f)), "Failed to allocate memory for d_Ow");
    checkCudaError(cudaMalloc((void**)&d_OwRight, neighborCount * sizeof(Eigen::Vector3f)), "Failed to allocate memory for d_OwRight");
    checkCudaError(cudaMalloc((void**)&d_bestDists, neighborCount * mapPointVecSize * sizeof(int)), "Failed to allocate memory for d_bestDists");
    checkCudaError(cudaMalloc((void**)&d_bestIdxs, neighborCount * mapPointVecSize * sizeof(int)), "Failed to allocate memory for d_bestIdxs");

    memory_is_initialized = true;
}

void FuseKernel::shutdown() {
    if (!memory_is_initialized) 
        return;

    checkCudaError(cudaFree(d_currKFMapPoints),"Failed to free fuse kernel memory: d_currKFMapPoints");
    checkCudaError(cudaFree(d_neighKFs),"Failed to free fuse kernel memory: d_neighKFs");
    checkCudaError(cudaFree(d_Tcw),"Failed to free fuse kernel memory: d_Tcw");
    checkCudaError(cudaFree(d_TcwRight),"Failed to free fuse kernel memory: d_TcwRight");
    checkCudaError(cudaFree(d_Ow),"Failed to free fuse kernel memory: d_Ow");
    checkCudaError(cudaFree(d_OwRight),"Failed to free fuse kernel memory: d_OwRight");
    checkCudaError(cudaFree(d_bestDists),"Failed to free fuse kernel memory: d_bestDists");
    checkCudaError(cudaFree(d_bestIdxs),"Failed to free fuse kernel memory: d_bestIdxs");
}

__device__ int predictScale(float currentDist, float maxDistance, MAPPING_DATA_WRAPPER::CudaKeyFrame* pKF) {
    float ratio = maxDistance/currentDist;
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

__device__ inline bool isInImage(MAPPING_DATA_WRAPPER::CudaKeyFrame* keyframe, const float &x, const float &y) {
    return (x>=keyframe->mnMinX && x<keyframe->mnMaxX && y>=keyframe->mnMinY && y<keyframe->mnMaxY);
}

__device__ Eigen::Vector2f fisheyeProject(const Eigen::Vector3f &v3D, float* mvParameters) {
    const float x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const float psi = atan2f(v3D[1], v3D[0]);

    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5
                          + mvParameters[6] * theta7 + mvParameters[7] * theta9;

    Eigen::Vector2f res;
    res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
    res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];

    return res;
}

__device__ Eigen::Vector2f pinholeProject(const Eigen::Vector3f &v3D, float* mvParameters) {
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];

    return res;
}

__global__ void fuseKernel(MAPPING_DATA_WRAPPER::CudaMapPoint* currKFMapPoints, MAPPING_DATA_WRAPPER::CudaKeyFrame* neighKF, 
                           int numPoints, bool cameraIsFisheye, Eigen::Vector3f Ow, Sophus::SE3f Tcw, float th, bool bRight,
                           int* bestDists, int* bestIdxs) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numPoints)
        return;

    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    MAPPING_DATA_WRAPPER::CudaMapPoint pMP = currKFMapPoints[idx];
    
    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = Tcw * p3Dw;
    const float invz = 1/p3Dc(2);

    Eigen::Vector2f uv;
    if (cameraIsFisheye)
        uv = fisheyeProject(p3Dc, neighKF->camera1.mvParameters);
    else
        uv = pinholeProject(p3Dc, neighKF->camera1.mvParameters);
            
    if ((p3Dc(2) < 0.0f) || (!isInImage(neighKF, uv(0), uv(1))))
        return;

    const float ur = uv(0) - neighKF->mbf*invz;
    const float maxDistance = 1.2 * pMP.mfMaxDistance;
    const float minDistance = 0.8 * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw - Ow;
    const float dist3D = PO.norm();

    if (dist3D < minDistance || dist3D > maxDistance)
        return;

    Eigen::Vector3f Pn = pMP.mNormalVector;
    if (PO.dot(Pn) < 0.5*dist3D)
        return;

    int nPredictedLevel = predictScale(dist3D, pMP.mfMaxDistance, neighKF);
    const float radius = th * neighKF->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    // ## GetFeaturesInArea Function ##
    float factorX = radius;
    float factorY = radius;
    float x = uv.x();
    float y = uv.y();

    const int nMinCellX = max(0,(int)floor((x - neighKF->mnMinX - factorX) * neighKF->mfGridElementWidthInv));
    if (nMinCellX >= neighKF->mnGridCols) 
        return;
    
    const int nMaxCellX = min((int)neighKF->mnGridCols-1,(int)ceil((x - neighKF->mnMinX + factorX) * neighKF->mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y - neighKF->mnMinY - factorY) * neighKF->mfGridElementHeightInv));
    if (nMinCellY >= neighKF->mnGridRows)
        return;
    
    const int nMaxCellY = min((int)neighKF->mnGridRows-1,(int)ceil((y - neighKF->mnMinY + factorY) * neighKF->mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {   
            std::size_t* vCell;
            int vCell_size;
            if (!bRight) {
                vCell = &neighKF->flatMGrid[ix * neighKF->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                vCell_size = neighKF->flatMGrid_size[ix * neighKF->mnGridRows + iy];
            } else {
                vCell = &neighKF->flatMGridRight[ix * neighKF->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                vCell_size = neighKF->flatMGridRight_size[ix * neighKF->mnGridRows + iy];
            }

            for (size_t j=0, jend=vCell_size; j<jend; j++) {
                size_t temp_idx = vCell[j];

                const MAPPING_DATA_WRAPPER::CudaKeyPoint &kpUn = (neighKF->Nleft == -1) ? neighKF->mvKeysUn[temp_idx]
                                                                                        : (!bRight) ? neighKF->mvKeys[temp_idx]
                                                                                                    : neighKF->mvKeysRight[temp_idx];
                
                const float distx = kpUn.ptx-x;
                const float disty = kpUn.pty-y;
                
                if (fabs(distx) < radius && fabs(disty) < radius) {

                    const int &kpLevel= kpUn.octave;

                    if (kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
                        continue;

                    if (neighKF->mvuRight[temp_idx] >= 0) {
                        // Check reprojection error in stereo
                        const float &kpx = kpUn.ptx;
                        const float &kpy = kpUn.pty;
                        const float &kpr = neighKF->mvuRight[temp_idx];
                        const float ex = x-kpx;
                        const float ey = y-kpy;
                        const float er = ur-kpr;
                        const float e2 = ex*ex+ey*ey+er*er;

                        if (e2 * neighKF->mvInvLevelSigma2[kpLevel] > 7.8)
                            continue;
                    }
                    else {
                        const float &kpx = kpUn.ptx;
                        const float &kpy = kpUn.pty;
                        const float ex = x-kpx;
                        const float ey = y-kpy;
                        const float e2 = ex*ex+ey*ey;
                        if(e2 * neighKF->mvInvLevelSigma2[kpLevel] > 5.99)
                            continue;
                    }

                    if (bRight) 
                        temp_idx += neighKF->Nleft;

                    const uint8_t* dKF = &neighKF->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

                    int dist = DescriptorDistance(MPdescriptor,dKF);

                    if (dist<bestDist) {
                        bestDist = dist;
                        bestIdx = temp_idx;
                    }
                }
            }
        }
    }
    
    bestDists[idx] = bestDist;
    bestIdxs[idx] = bestIdx;
}

void FuseKernel::launch(ORB_SLAM3::KeyFrame *neighKF, const vector<ORB_SLAM3::MapPoint*> &currKFMapPoints, const float th, 
                        const bool bRight, ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow, 
                        vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs) {

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized)
        initialize();

    MAPPING_DATA_WRAPPER::CudaKeyFrame* d_neighKF = CudaKeyFrameStorage::getCudaKeyFrame(neighKF->mnId);
    if (d_neighKF == nullptr) {
        cerr << "[ERROR] FuseKernel::launch: ] CudaKeyFrameStorage doesn't have the keyframe: " << neighKF->mnId << "\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point startCopyObjectCreation = std::chrono::steady_clock::now();
#endif

    // TODO: avoid copying the map points for all of the neighbors and do it once only
    int numValidPoints = 0;
    MAPPING_DATA_WRAPPER::CudaMapPoint wrappedCurrKFMapPoints[currKFMapPoints.size()];
    for (int i = 0; i < currKFMapPoints.size(); i++) {
        ORB_SLAM3::MapPoint* pMP = currKFMapPoints[i];
        if (!pMP || pMP->isBad() || pMP->IsInKeyFrame(neighKF))
            continue;
        validMapPoints.push_back(pMP);
        wrappedCurrKFMapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
        numValidPoints++;
    }

    if (numValidPoints == 0)
        return;

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endCopyObjectCreation = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpy = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(d_currKFMapPoints, wrappedCurrKFMapPoints, sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)*numValidPoints, cudaMemcpyHostToDevice), "Failed to copy vector wrappedCurrKFMapPoints from host to device");

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endMemcpy = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startKernel = std::chrono::steady_clock::now();
#endif

    int blockSize = 256;
    int numBlocks = (numValidPoints + blockSize -1) / blockSize;
    fuseKernel<<<numBlocks, blockSize>>>(d_currKFMapPoints, d_neighKF, numValidPoints, CudaUtils::cameraIsFisheye, Ow, Tcw, th, bRight, d_bestDists, d_bestIdxs);

    checkCudaError(cudaGetLastError(), "[fuseKernel:] Failed to launch kernel");
    checkCudaError(cudaDeviceSynchronize(), "[fuseKernel:] cudaDeviceSynchronize failed after kernel launch");  

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpyToCPU = std::chrono::steady_clock::now();
#endif
    
    checkCudaError(cudaMemcpy(bestDists, d_bestDists, numValidPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host");
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, numValidPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host");

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endMemcpyToCPU = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    
    double copyObjectCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endCopyObjectCreation - startCopyObjectCreation).count();
    double memcpyToGPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpy - startMemcpy).count();
    double kernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endKernel - startKernel).count();
    double memcpyToCPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpyToCPU - startMemcpyToCPU).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    input_data_wrap_time.back().second += copyObjectCreation;
    input_data_transfer_time.back().second += memcpyToGPU;
    kernel_exec_time.back().second += kernel;
    output_data_transfer_time.back().second += memcpyToCPU;
    total_exec_time.back().second += total;

    frameCounter++;
#endif
    
    // cout << "================================= Fuse called for KF: " << neighKF->mnId << " =================================\n";
    // for (int i = 0; i < numValidPoints; i++) {
    //     if (h_bestDist[i] != 256)
    //         printf("(i: %d, bestDist: %d, bestIdx: %d), ", i, h_bestDist[i], h_bestIdx[i]);
    // }
    // printf("\n");
    
    // cout << "************ CPU Side ************\n";
    // origFuse(neighKF, currKF->GetMapPointMatches(), th, bRight);
}

__global__ void fuseKernelV2(
    MAPPING_DATA_WRAPPER::CudaMapPoint* currKFMapPoints, MAPPING_DATA_WRAPPER::CudaKeyFrame** neighKFs, int numPoints, int numKFs, 
    Eigen::Vector3f *Ow, Eigen::Vector3f *OwRight, Sophus::SE3f *Tcw, Sophus::SE3f *TcwRight, bool cameraIsFisheye, float th,
    int* bestDists, int* bestIdxs
) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int maxIdx, neighKFIdx, mapPointIdx;
    bool bRight = false;
    if (cameraIsFisheye) {
        maxIdx = numPoints * numKFs * 2;
        neighKFIdx = idx / numPoints / 2;
        mapPointIdx = idx % numPoints;
        if (idx % (2*numPoints) >= numPoints)
            bRight = true;
    }
    else {
        maxIdx = numPoints * numKFs;
        neighKFIdx = idx / numPoints;
        mapPointIdx = idx % numPoints;
    }

    if (idx >= maxIdx || neighKFIdx >= numKFs || mapPointIdx >= numPoints)
        return;

    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    MAPPING_DATA_WRAPPER::CudaMapPoint pMP = currKFMapPoints[mapPointIdx];
    MAPPING_DATA_WRAPPER::CudaKeyFrame *neighKF = neighKFs[neighKFIdx];
    
    Sophus::SE3f currTcw = bRight ? TcwRight[neighKFIdx] : Tcw[neighKFIdx];
    Eigen::Vector3f currOw = bRight ? OwRight[neighKFIdx] : Ow[neighKFIdx];

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = currTcw * p3Dw;
    const float invz = 1/p3Dc(2);

    Eigen::Vector2f uv;
    if (cameraIsFisheye) {
        if (!bRight)
            uv = fisheyeProject(p3Dc, neighKF->camera1.mvParameters);
        else
            uv = fisheyeProject(p3Dc, neighKF->camera2.mvParameters);
    }
    else
        uv = pinholeProject(p3Dc, neighKF->camera1.mvParameters);
            
    if ((p3Dc(2) < 0.0f) || (!isInImage(neighKF, uv(0), uv(1))))
        return;

    const float ur = uv(0) - neighKF->mbf*invz;
    const float maxDistance = 1.2 * pMP.mfMaxDistance;
    const float minDistance = 0.8 * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw - currOw;
    const float dist3D = PO.norm();

    if (dist3D < minDistance || dist3D > maxDistance)
        return;

    Eigen::Vector3f Pn = pMP.mNormalVector;
    if (PO.dot(Pn) < 0.5*dist3D)
        return;

    int nPredictedLevel = predictScale(dist3D, pMP.mfMaxDistance, neighKF);
    const float radius = th * neighKF->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    // ## GetFeaturesInArea Function ##
    float factorX = radius;
    float factorY = radius;
    float x = uv.x();
    float y = uv.y();

    const int nMinCellX = max(0,(int)floor((x - neighKF->mnMinX - factorX) * neighKF->mfGridElementWidthInv));
    if (nMinCellX >= neighKF->mnGridCols) 
        return;
    
    const int nMaxCellX = min((int)neighKF->mnGridCols-1,(int)ceil((x - neighKF->mnMinX + factorX) * neighKF->mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y - neighKF->mnMinY - factorY) * neighKF->mfGridElementHeightInv));
    if (nMinCellY >= neighKF->mnGridRows)
        return;
    
    const int nMaxCellY = min((int)neighKF->mnGridRows-1,(int)ceil((y - neighKF->mnMinY + factorY) * neighKF->mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {   
            std::size_t* vCell;
            int vCell_size;
            if (!bRight) {
                vCell = &neighKF->flatMGrid[ix * neighKF->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                vCell_size = neighKF->flatMGrid_size[ix * neighKF->mnGridRows + iy];
            } else {
                vCell = &neighKF->flatMGridRight[ix * neighKF->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                vCell_size = neighKF->flatMGridRight_size[ix * neighKF->mnGridRows + iy];
            }

            for (size_t j=0, jend=vCell_size; j<jend; j++) {
                size_t temp_idx = vCell[j];

                const MAPPING_DATA_WRAPPER::CudaKeyPoint &kpUn = (neighKF->Nleft == -1) ? neighKF->mvKeysUn[temp_idx]
                                                                                        : (!bRight) ? neighKF->mvKeys[temp_idx]
                                                                                                    : neighKF->mvKeysRight[temp_idx];
                
                const float distx = kpUn.ptx-x;
                const float disty = kpUn.pty-y;
                
                if (fabs(distx) < radius && fabs(disty) < radius) {

                    const int &kpLevel= kpUn.octave;

                    if (kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
                        continue;

                    if (neighKF->mvuRight[temp_idx] >= 0) {
                        // Check reprojection error in stereo
                        const float &kpx = kpUn.ptx;
                        const float &kpy = kpUn.pty;
                        const float &kpr = neighKF->mvuRight[temp_idx];
                        const float ex = x-kpx;
                        const float ey = y-kpy;
                        const float er = ur-kpr;
                        const float e2 = ex*ex+ey*ey+er*er;

                        if (e2 * neighKF->mvInvLevelSigma2[kpLevel] > 7.8)
                            continue;
                    }
                    else {
                        const float &kpx = kpUn.ptx;
                        const float &kpy = kpUn.pty;
                        const float ex = x-kpx;
                        const float ey = y-kpy;
                        const float e2 = ex*ex+ey*ey;
                        if(e2 * neighKF->mvInvLevelSigma2[kpLevel] > 5.99)
                            continue;
                    }

                    if (bRight) 
                        temp_idx += neighKF->Nleft;

                    const uint8_t* dKF = &neighKF->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

                    int dist = DescriptorDistance(MPdescriptor,dKF);

                    if (dist<bestDist) {
                        bestDist = dist;
                        bestIdx = temp_idx;
                    }
                }
            }
        }
    }
    
    bestDists[idx] = bestDist;
    bestIdxs[idx] = bestIdx;
}

void FuseKernel::launchV2(std::vector<ORB_SLAM3::KeyFrame*> neighKFs, ORB_SLAM3::KeyFrame *currKF, float th, 
                          vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs) {

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized)
        initialize();
    
    std::vector<ORB_SLAM3::MapPoint*> currKFMapPoints = currKF->GetMapPointMatches();
    int neighKFSize = neighKFs.size();
    if (neighKFSize == 0 || currKFMapPoints.size() == 0)
        return;

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point startCopyObjectCreation = std::chrono::steady_clock::now();
#endif

    MAPPING_DATA_WRAPPER::CudaKeyFrame* neighKFsGPUAddress[neighKFSize];

    for (int i = 0; i < neighKFSize; i++) {
        neighKFsGPUAddress[i] = CudaKeyFrameStorage::getCudaKeyFrame(neighKFs[i]->mnId);
        if (neighKFsGPUAddress[i] == nullptr) {
            cerr << "[ERROR] FuseKernel::launch: ] CudaKeyFrameStorage doesn't have the keyframe: " << neighKFs[i]->mnId << "\n";
            MappingKernelController::shutdownKernels(true, true);
            exit(EXIT_FAILURE);
        }
    }

    int numValidPoints = 0;
    MAPPING_DATA_WRAPPER::CudaMapPoint wrappedCurrKFMapPoints[currKFMapPoints.size()];
    for (int i = 0; i < currKFMapPoints.size(); i++) {
        ORB_SLAM3::MapPoint* pMP = currKFMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;
        else {
            wrappedCurrKFMapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
            validMapPoints.push_back(pMP);
            numValidPoints++;
        }
    }

    if (numValidPoints == 0)
        return;

    Sophus::SE3f Tcw[neighKFSize], TcwRight[neighKFSize];
    Eigen::Vector3f Ow[neighKFSize], OwRight[neighKFSize];
    for (int i = 0; i < neighKFSize; i++) {
        Tcw[i] = neighKFs[i]->GetPose();
        Ow[i] = neighKFs[i]->GetCameraCenter();

        if (CudaUtils::cameraIsFisheye) {
            TcwRight[i] = neighKFs[i]->GetRightPose();
            OwRight[i] = neighKFs[i]->GetRightCameraCenter();
        }
    }

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endCopyObjectCreation = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpy = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(d_neighKFs, neighKFsGPUAddress, neighKFSize * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*), cudaMemcpyHostToDevice), "Failed to copy vector neighKFsGPUAddress from host to device");
    checkCudaError(cudaMemcpy(d_currKFMapPoints, wrappedCurrKFMapPoints, numValidPoints * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "Failed to copy vector wrappedCurrKFMapPoints from host to device");
    checkCudaError(cudaMemcpy(d_Tcw, Tcw, neighKFSize * sizeof(Sophus::SE3f), cudaMemcpyHostToDevice), "Failed to copy vector Tcw from host to device");
    checkCudaError(cudaMemcpy(d_Ow, Ow, neighKFSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), "Failed to copy vector Ow from host to device");
    if (CudaUtils::cameraIsFisheye) {
        checkCudaError(cudaMemcpy(d_TcwRight, TcwRight, neighKFSize * sizeof(Sophus::SE3f), cudaMemcpyHostToDevice), "Failed to copy vector TcwRight from host to device");
        checkCudaError(cudaMemcpy(d_OwRight, OwRight, neighKFSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), "Failed to copy vector OwRight from host to device");
    }

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endMemcpy = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startKernel = std::chrono::steady_clock::now();
#endif

    int keyFramesToProcessCount = CudaUtils::cameraIsFisheye ? neighKFSize*2 : neighKFSize;
    int blockSize = 256;
    int numBlocks = (numValidPoints*keyFramesToProcessCount + blockSize - 1) / blockSize;

    fuseKernelV2<<<numBlocks, blockSize>>>(
        d_currKFMapPoints, d_neighKFs, numValidPoints, neighKFSize, d_Ow, d_OwRight, d_Tcw, d_TcwRight,
        CudaUtils::cameraIsFisheye, th, d_bestDists, d_bestIdxs
    );

    checkCudaError(cudaGetLastError(), "[fuseKernelV2:] Failed to launch kernel");
    checkCudaError(cudaDeviceSynchronize(), "[fuseKernelV2:] cudaDeviceSynchronize failed after kernel launch");  

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpyToCPU = std::chrono::steady_clock::now();
#endif
    
    checkCudaError(cudaMemcpy(bestDists, d_bestDists, numValidPoints * keyFramesToProcessCount * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host");
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, numValidPoints * keyFramesToProcessCount * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host");

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endMemcpyToCPU = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    
    double copyObjectCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endCopyObjectCreation - startCopyObjectCreation).count();
    double memcpyToGPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpy - startMemcpy).count();
    double kernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endKernel - startKernel).count();
    double memcpyToCPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpyToCPU - startMemcpyToCPU).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    input_data_wrap_time.emplace_back(frameCounter, copyObjectCreation);
    input_data_transfer_time.emplace_back(frameCounter, memcpyToGPU);
    kernel_exec_time.emplace_back(frameCounter, kernel);
    output_data_transfer_time.emplace_back(frameCounter, memcpyToCPU);
    total_exec_time.emplace_back(frameCounter, total);
#endif

    // cout << "\n\n////////////////////////////////////////// Current KF: " << currKF->mnId << " //////////////////////////////////////////\n";
    // for (int iKF = 0; iKF < neighKFSize; iKF++) {
    //     cout << "================================= Neighbor KF: " << neighKFs[iKF]->mnId << " =================================\n";
    //     for (int iMP = 0; iMP < numValidPoints; iMP++) {
    //         int idx = (currKF->NLeft == -1) ? iKF*numValidPoints + iMP : iKF*numValidPoints*2 + iMP;
    //         if (bestDists[idx] != 256 && !validMapPoints[iMP]->IsInKeyFrame(neighKFs[iKF]))
    //             printf("(i: %d, bestDist: %d, bestIdx: %d), ", iMP, bestDists[idx], bestIdxs[idx]);
    //     }
    //     printf("\n");
        
    //     cout << "************ CPU Side ************\n";
    //     origFuse(neighKFs[iKF], currKF->GetMapPointMatches(), th, 0);

    //     if (currKF->NLeft != -1) {
    //         cout << ".................... Right Image ....................\n";
    //         for (int iMP = 0; iMP < numValidPoints; iMP++) {
    //             int idx = iKF*numValidPoints*2 + numValidPoints + iMP;
    //             if (bestDists[idx] != 256 && !validMapPoints[iMP]->IsInKeyFrame(neighKFs[iKF]))
    //                 printf("(i: %d, bestDist: %d, bestIdx: %d), ", iMP, bestDists[idx], bestIdxs[idx]);
    //         }
    //         printf("\n");
            
    //         cout << "************ CPU Side ************\n";
    //         origFuse(neighKFs[iKF], currKF->GetMapPointMatches(), th, 1);
    //     }
    // }
}

void FuseKernel::origFuse(ORB_SLAM3::KeyFrame *pKF, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints, const float th, const bool bRight) {
    ORB_SLAM3::GeometricCamera* pCamera;
    Sophus::SE3f Tcw;
    Eigen::Vector3f Ow;

    if(bRight) {
        Tcw = pKF->GetRightPose();
        Ow = pKF->GetRightCameraCenter();
        pCamera = pKF->mpCamera2;
    }
    else {
        Tcw = pKF->GetPose();
        Ow = pKF->GetCameraCenter();
        pCamera = pKF->mpCamera;
    }

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    // For debbuging
    int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;

    int validMapPointCounter = -1;

    for(int i=0; i<nMPs; i++) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];

        if(!pMP) {
            count_notMP++;
            continue;
        }
        
        if(pMP->isBad()) {
            count_bad++;
            continue;
        }

        validMapPointCounter++;
        
        if(pMP->IsInKeyFrame(pKF)) {
            count_isinKF++;
            continue;
        }
        
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0f) {
            count_negdepth++;
            continue;
        }

        const float invz = 1/p3Dc(2);

        const Eigen::Vector2f uv = pCamera->project(p3Dc);

        // Point must be inside the image
        if(!pKF->IsInImage(uv(0),uv(1))) {
            count_notinim++;
            continue;
        }

        const float ur = uv(0)-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance) {
            count_dist++;
            continue;
        }

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D) {
            count_normal++;
            continue;
        }

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius,bRight);

        if(vIndices.empty()) {
            count_notidx++;
            continue;
        }

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;

        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++) {
            size_t idx = *vit;

            const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                        : (!bRight) ? pKF -> mvKeys[idx]
                                                                    : pKF -> mvKeysRight[idx];

            const int &kpLevel= kp.octave;


            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) {
                continue;
            }

            if(pKF->mvuRight[idx]>=0) {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = uv(0)-kpx;
                const float ey = uv(1)-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8){
                    continue;
                }
            }
            else {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv(0)-kpx;
                const float ey = uv(1)-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99){
                    continue;
                }
            }

            if(bRight) idx += pKF->NLeft;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist != 256)
            printf("(i: %d, bestDist: %d, bestIdx: %d), ", validMapPointCounter, bestDist, bestIdx);
    }

    printf("\n");
}

int FuseKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void FuseKernel::saveStats(const std::string &file_path) {
    std::string data_path = file_path + "/FuseKernel/";
    std::cout << "[FuseKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[FuseKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }
    std::ofstream myfile;
    
    myfile.open(data_path + "/kernel_exec_time.txt");
    for (const auto& p : kernel_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_wrap_time.txt");
    for (const auto& p : input_data_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_transfer_time.txt");
    for (const auto& p : input_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
    
    myfile.open(data_path + "/output_data_transfer_time.txt");
    for (const auto& p : output_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
}