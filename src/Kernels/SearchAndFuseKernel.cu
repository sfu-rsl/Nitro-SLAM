#include <algorithm>
#include <iostream>
#include "Kernels/SearchAndFuseKernel.h"
#include "Kernels/LoopClosingKernelController.h"

namespace {
// Starting capacities. A loop closure occasionally fuses more than this, so
// ensureCapacity() grows the buffers rather than overrunning them.
constexpr size_t kInitialMapPoints = 3000;
constexpr size_t kInitialConnectedKFs = 100;
}

void SearchAndFuseKernel::freeBuffers()
{
    if (mapPointCapacity == 0 && connectedKFCapacity == 0)
        return;

    cudaFreeHost(h_MapPoints);
    cudaFreeHost(h_KeyFrames);
    cudaFreeHost(h_Ow);
    cudaFreeHost(h_Tcw);
    cudaFreeHost(bestDists);
    cudaFreeHost(bestIdxs);
    cudaFree(d_MapPoints);
    cudaFree(d_KeyFrames);
    cudaFree(d_Ow);
    cudaFree(d_Tcw);
    cudaFree(d_bestDists);
    cudaFree(d_bestIdxs);

    mapPointCapacity = 0;
    connectedKFCapacity = 0;
}

// The buffers were previously fixed at 3000 points / 100 keyframes while launch()
// filled and copied using the *runtime* sizes, unchecked. A larger loop closure
// therefore wrote past the pinned host arrays and asked cudaMemcpy to write past
// the device allocations, which fails with cudaErrorInvalidValue. Grow instead of
// clamping: dropping points would silently corrupt the fuse result.
void SearchAndFuseKernel::ensureCapacity(size_t numMapPoints, size_t numKFs)
{
    if (numMapPoints <= mapPointCapacity && numKFs <= connectedKFCapacity)
        return;

    const size_t newMapPoints = std::max(numMapPoints, std::max(mapPointCapacity * 2, kInitialMapPoints));
    const size_t newKFs       = std::max(numKFs,       std::max(connectedKFCapacity * 2, kInitialConnectedKFs));

    // Only interesting when we are actually growing past the starting capacity --
    // that is the case that used to overrun the buffers and fail the memcpy.
    if (mapPointCapacity != 0 || connectedKFCapacity != 0)
        std::cout << "[SearchAndFuseKernel:] growing buffers: points "
                  << mapPointCapacity << " -> " << newMapPoints << ", KFs "
                  << connectedKFCapacity << " -> " << newKFs
                  << " (needed " << numMapPoints << "/" << numKFs << ")" << std::endl;

    freeBuffers();

    checkCudaError(cudaMallocHost((void**)&h_MapPoints, newMapPoints * sizeof(*h_MapPoints)), "SearchAndFuseKernel: failed to allocate h_MapPoints");
    checkCudaError(cudaMallocHost((void**)&h_KeyFrames, newKFs * sizeof(*h_KeyFrames)), "SearchAndFuseKernel: failed to allocate h_KeyFrames");
    checkCudaError(cudaMallocHost((void**)&h_Ow, newKFs * sizeof(*h_Ow)), "SearchAndFuseKernel: failed to allocate h_Ow");
    checkCudaError(cudaMallocHost((void**)&h_Tcw, newKFs * sizeof(*h_Tcw)), "SearchAndFuseKernel: failed to allocate h_Tcw");
    checkCudaError(cudaMallocHost((void**)&bestDists, newKFs * newMapPoints * sizeof(int)), "SearchAndFuseKernel: failed to allocate bestDists");
    checkCudaError(cudaMallocHost((void**)&bestIdxs, newKFs * newMapPoints * sizeof(int)), "SearchAndFuseKernel: failed to allocate bestIdxs");

    checkCudaError(cudaMalloc(&d_MapPoints, newMapPoints * sizeof(*d_MapPoints)), "SearchAndFuseKernel: failed to allocate d_MapPoints");
    checkCudaError(cudaMalloc(&d_KeyFrames, newKFs * sizeof(*d_KeyFrames)), "SearchAndFuseKernel: failed to allocate d_KeyFrames");
    checkCudaError(cudaMalloc(&d_Ow, newKFs * sizeof(*d_Ow)), "SearchAndFuseKernel: failed to allocate d_Ow");
    checkCudaError(cudaMalloc(&d_Tcw, newKFs * sizeof(*d_Tcw)), "SearchAndFuseKernel: failed to allocate d_Tcw");
    checkCudaError(cudaMalloc(&d_bestDists, newKFs * newMapPoints * sizeof(int)), "SearchAndFuseKernel: failed to allocate d_bestDists");
    checkCudaError(cudaMalloc(&d_bestIdxs, newKFs * newMapPoints * sizeof(int)), "SearchAndFuseKernel: failed to allocate d_bestIdxs");

    mapPointCapacity = newMapPoints;
    connectedKFCapacity = newKFs;
}

void SearchAndFuseKernel::initialize()
{
    if (memory_is_initialized)
        return;

    ensureCapacity(kInitialMapPoints, kInitialConnectedKFs);
    memory_is_initialized = true;
}

void SearchAndFuseKernel::shutdown()
{
    if (!memory_is_initialized) 
        return;

    freeBuffers();
    memory_is_initialized = false;
}

__device__ inline Eigen::Vector2f KannalaBrandt8Project(const Eigen::Vector3f &v3D, float* mvParameters)
{
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

__device__ inline Eigen::Vector2f PinholeProject(const Eigen::Vector3f &v3D, float* mvParameters)
{
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];
    return res;
}

__device__ inline bool isInImage1(MAPPING_DATA_WRAPPER::CudaKeyFrame* keyframe, const float &x, const float &y)
{
    return (x>=keyframe->mnMinX && x<keyframe->mnMaxX && y>=keyframe->mnMinY && y<keyframe->mnMaxY);
}

__device__ int predictScale1(float currentDist, float maxDistance, MAPPING_DATA_WRAPPER::CudaKeyFrame* pKF)
{
    float ratio = maxDistance/currentDist;
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

__global__ void searchAndFuseKernel(Eigen::Vector3f* Ow, Sophus::SE3f *Tcw, 
                            MAPPING_DATA_WRAPPER::CudaKeyFrame** connectedKFs, MAPPING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int chunkNumKFs, int totalNumKFs, int baseKFIdx, int numPoints, float th,
                            int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIdx = numPoints * totalNumKFs;
    int connectedKFIdx = idx / numPoints;
    int mapPointIdx = idx % numPoints;
    
    if (idx >= maxIdx || connectedKFIdx >= totalNumKFs || mapPointIdx >= numPoints){
        return;
    }

    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    MAPPING_DATA_WRAPPER::CudaMapPoint& pMP = mapPoints[mapPointIdx];
    MAPPING_DATA_WRAPPER::CudaKeyFrame *connectedKF = connectedKFs[connectedKFIdx];

    // printf("idx: %llu, pMP.mnId: %llu, keyframe.mnId: %llu, connectedKF->mapPointsId_size: %d\n", idx, pMP.mnId, connectedKF->mnId, connectedKF->mapPointsId_size);

    Sophus::SE3f currTcw = Tcw[connectedKFIdx];
    Eigen::Vector3f currOw = Ow[connectedKFIdx];

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = currTcw * p3Dw;

    if (p3Dc(2) < 0.0f){
        return;
    }

    Eigen::Vector2f uv;
    if(connectedKF->Nleft != -1)
        uv = KannalaBrandt8Project(p3Dc, connectedKF->camera1.mvParameters);
    else if(connectedKF->Nleft == -1)
        uv = PinholeProject(p3Dc, connectedKF->camera1.mvParameters);
    
    if ((!isInImage1(connectedKF, uv(0), uv(1)))){
        return;
    }

    const float maxDistance = 1.2f * pMP.mfMaxDistance;
    const float minDistance = 0.8f * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw - currOw;
    const float dist3D = PO.norm();

    if (dist3D < minDistance || dist3D > maxDistance)
        return;
    
    Eigen::Vector3f Pn = pMP.mNormalVector;
    if (PO.dot(Pn) < 0.5*dist3D)
        return;

    int nPredictedLevel = predictScale1(dist3D, pMP.mfMaxDistance, connectedKF);
    const float radius = th * connectedKF->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = uv(0);
    float y = uv(1);

    const int nMinCellX = max(0,(int)floor((x - connectedKF->mnMinX - factorX) * connectedKF->mfGridElementWidthInv));
    if (nMinCellX >= connectedKF->mnGridCols) 
        return;
    
    const int nMaxCellX = min((int)connectedKF->mnGridCols-1,(int)ceil((x - connectedKF->mnMinX + factorX) * connectedKF->mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y - connectedKF->mnMinY - factorY) * connectedKF->mfGridElementHeightInv));
    if (nMinCellY >= connectedKF->mnGridRows)
        return;
    
    const int nMaxCellY = min((int)connectedKF->mnGridRows-1,(int)ceil((y - connectedKF->mnMinY + factorY) * connectedKF->mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return;
    
    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {   
            std::size_t* vCell;
            int vCell_size;
            
            vCell = &connectedKF->flatMGrid[ix * connectedKF->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
            vCell_size = connectedKF->flatMGrid_size[ix * connectedKF->mnGridRows + iy];
            
            for (size_t j=0, jend=vCell_size; j<jend; j++) {
                size_t temp_idx = vCell[j];

                const MAPPING_DATA_WRAPPER::CudaKeyPoint &kpUn = (connectedKF->Nleft == -1) ? connectedKF->mvKeysUn[temp_idx]
                                                                                        : (!false) ? connectedKF->mvKeys[temp_idx]
                                                                                                    : connectedKF->mvKeysRight[temp_idx];
                
                const float distx = kpUn.ptx-x;
                const float disty = kpUn.pty-y;

                if (fabs(distx) < radius && fabs(disty) < radius) {
                    const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;

                    if (kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
                        continue;
                    
                    const uint8_t* dKF = &connectedKF->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

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


int SearchAndFuseKernel::launch(std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, float th,
                        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints, vector<ORB_SLAM3::MapPoint*> &vpReplacePoints)
{
#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif
    if (!memory_is_initialized)
        initialize();

    const int TH_LOW = 50;
    int numValidPoints = 0;
    int connectedKFSize = connectedKFs.size();
    size_t mapPointVecSize = vpMapPoints.size();

    ensureCapacity(mapPointVecSize, (size_t)connectedKFSize);

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point startCopyObjectCreation = std::chrono::steady_clock::now();
#endif

    for (size_t i = 0; i < connectedKFSize; i++) {
        h_Tcw[i] = Sophus::SE3f(connectedScws[i].rotationMatrix(),connectedScws[i].translation()/connectedScws[i].scale());
        h_Ow[i] = h_Tcw[i].inverse().translation();
    }
    
    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;
        else {
            h_MapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
            numValidPoints++;
        }
    }

    for (int i=0; i < connectedKFSize; i++){
        ORB_SLAM3::KeyFrame* pKF = connectedKFs[i];
        h_KeyFrames[i] = CudaKeyFrameAllocator::create(pKF);
    }

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endCopyObjectCreation = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpy = std::chrono::steady_clock::now();
#endif
    
    checkCudaError(cudaMemcpy(d_MapPoints, h_MapPoints, numValidPoints * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "Failed to copy h_MapPoints to host");
    checkCudaError(cudaMemcpy(d_KeyFrames, h_KeyFrames, connectedKFSize * sizeof(*h_KeyFrames), cudaMemcpyHostToDevice), "Failed to copy h_KeyFrames to host");
    checkCudaError(cudaMemcpy(d_Ow, h_Ow, connectedKFSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), "Failed to copy h_Ow to host");
    checkCudaError(cudaMemcpy(d_Tcw, h_Tcw, connectedKFSize * sizeof(Sophus::SE3f), cudaMemcpyHostToDevice), "Failed to copy h_Tcw to host");
    
#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endMemcpy = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startKernel = std::chrono::steady_clock::now();
#endif

    int threads = 256;
    const cudaStream_t stream = cudaStreamPerThread;
    int blocks = (connectedKFSize * numValidPoints + threads - 1) / threads;
    searchAndFuseKernel<<<blocks, threads, 0, stream>>>(d_Ow, d_Tcw, d_KeyFrames, d_MapPoints, 
                                    0, connectedKFSize, 0, numValidPoints, th, 
                                    d_bestDists, d_bestIdxs);

    cudaStreamSynchronize(stream);

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpyToCPU = std::chrono::steady_clock::now();
#endif
    
    checkCudaError(cudaMemcpy(bestDists, d_bestDists, numValidPoints * connectedKFSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host1");
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, numValidPoints * connectedKFSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endMemcpyToCPU = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startPostProcessing = std::chrono::steady_clock::now();
#endif
        
    // for (int iKF = 0; iKF < connectedKFSize; iKF++) {
    //     gpuOutFile << "================================= Connected KF: " << connectedKFs[iKF]->mnId << " =================================\n";
    //     cpuOutFile << "================================= Connected KF: " << connectedKFs[iKF]->mnId << " =================================\n";
    //     cpuOutFile.flush();
    //     const set<ORB_SLAM3::MapPoint*> spAlreadyFound = connectedKFs[iKF]->GetMapPoints();
    //     for (int iMP = 0; iMP < numValidPoints; iMP++) {
    //         int idx = iKF*numValidPoints + iMP;
    //         ORB_SLAM3::MapPoint* pMP = vpMapPoints[iMP];

    //         if(spAlreadyFound.count(pMP))
    //             continue;

    //         if (bestDists[idx] != 256)
    //             gpuOutFile << "(i: " << iMP << ", bestDist: " << bestDists[idx] << ", bestIdx: " << bestIdxs[idx] << ")\n";
    //     }
    //     gpuOutFile << "\n\n";
        
    //     origFuse(connectedKFs[iKF], connectedScws[iKF], vpMapPoints, th);
    // }

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";

    int nFused = 0;
    for (int iKF = 0; iKF < connectedKFSize; iKF++)
    {
        ORB_SLAM3::KeyFrame* pKF = connectedKFs[iKF];
        const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

        for (int iMP = 0; iMP < numValidPoints; iMP++)
        {
            ORB_SLAM3::MapPoint* pMP = vpMapPoints[iMP];
            if (!pMP || pMP->isBad())
                continue;
            
            int idx = iKF*numValidPoints + iMP;
            int bestDist = bestDists[idx];
            int bestIdx = bestIdxs[idx];

            if (bestDist == 256 || bestIdx == -1)
                continue;
            
            if(spAlreadyFound.count(pMP))
                continue;

            if (bestDist <= TH_LOW) {
                ORB_SLAM3::MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if (pMPinKF) {
                    if (!pMPinKF->isBad()) {
                        vpReplacePoints[iMP] = pMPinKF;
                    }
                }
                else{
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP, bestIdx);
                }
                nFused++;
            }
        }
    }

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endPostProcessing = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    
    double copyObjectCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endCopyObjectCreation - startCopyObjectCreation).count();
    double memcpyToGPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpy - startMemcpy).count();
    double kernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endKernel - startKernel).count();
    double memcpyToCPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpyToCPU - startMemcpyToCPU).count();
    double postProcessing = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endPostProcessing - startPostProcessing).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    input_data_wrap_time.push_back(copyObjectCreation);
    input_data_transfer_time.push_back(memcpyToGPU);
    kernel_exec_time.push_back(kernel);
    output_data_transfer_time.push_back(memcpyToCPU);
    post_processing_time.push_back(postProcessing);
    total_exec_time.push_back(total);
#endif

    return nFused;
}


void SearchAndFuseKernel::origFuse(ORB_SLAM3::KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<ORB_SLAM3::MapPoint*> &vpPoints, const float th)
{
    std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    int validMapPointCounter = -1;
    
    for(int iMP=0; iMP<nPoints; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];

        if(pMP->isBad())
            continue;
        
        validMapPointCounter++;

        if(spAlreadyFound.count(pMP))
            continue;
        
        // cpuOutFile << "idx: " << validMapPointCounter << ", pMP.mnId: " << pMP->mnId << ", keyframe.mnId: " << pKF->mnId << ", spAlreadyFound.size(): " << spAlreadyFound.size() << std::endl;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0f)
            continue;

        // Project into Image
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;

        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++){
            const size_t idx = *vit;

            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) {
                continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist != INT_MAX)
            cpuOutFile << "(i: " << validMapPointCounter << ", bestDist: " << bestDist << ", bestIdx: " << bestIdx << ")\n";
    }
    cpuOutFile << "\n\n";
}


int SearchAndFuseKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
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

void SearchAndFuseKernel::saveStats(const std::string &file_path) {
    std::string data_path = file_path + "/SearchAndFuseKernel/";
    std::cout << "[SearchAndFuseKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[SearchAndFuseKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }
    std::ofstream myfile;
    
    myfile.open(data_path + "/kernel_exec_time.txt");
    for (const auto& p : kernel_exec_time) {
        myfile << p << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_wrap_time.txt");
    for (const auto& p : input_data_wrap_time) {
        myfile << p << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_transfer_time.txt");
    for (const auto& p : input_data_transfer_time) {
        myfile << p << std::endl;
    }
    myfile.close();
    
    myfile.open(data_path + "/output_data_transfer_time.txt");
    for (const auto& p : output_data_transfer_time) {
        myfile << p << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/post_processing_time.txt");
    for (const auto& p : post_processing_time) {
        myfile << p << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p << std::endl;
    }
    myfile.close();
}