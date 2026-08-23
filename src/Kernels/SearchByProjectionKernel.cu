#include <algorithm>
#include <iostream>
#include "Kernels/SearchByProjectionKernel.h"
#include <omp.h>

// Starting capacity for the map-point buffers. Every launch path fills them with
// vpPoints.size() entries; ensureMapPointCapacity() grows them rather than running
// off the end of pinned host memory and past the device allocations.
// All launch paths run on the loop closing thread, so this needs no locking.
static const size_t kMapPointCapacity = 4100;
static size_t g_maxMapPointVecSize = 0;

void SearchByProjectionKernel::initialize(){
    if (memory_is_initialized)
        return;

    ensureMapPointCapacity(kMapPointCapacity);

    // These stay fixed at 3: the caller bounds the covisible-keyframe count with
    // covKFsSize = std::min<size_t>(3, ...) (LoopClosing.cc), so 3 is a guarantee
    // rather than an estimate.
    checkCudaError(cudaMallocHost((void**)&h_KeyFrames, 3 * sizeof(*h_KeyFrames)), "SearchByProjectionKernel: failed to allocate h_KeyFrames");
    checkCudaError(cudaMallocHost((void**)&h_Ow, 3 * sizeof(*h_Ow)), "SearchByProjectionKernel: failed to allocate h_Ow");
    checkCudaError(cudaMallocHost((void**)&h_Tcw, 3 * sizeof(*h_Tcw)), "SearchByProjectionKernel: failed to allocate h_Tcw");

    checkCudaError(cudaMalloc(&d_KeyFrame, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "SearchByProjectionKernel: failed to allocate d_KeyFrame");
    checkCudaError(cudaMalloc(&d_KeyFrames, 3 * sizeof(*d_KeyFrames)), "SearchByProjectionKernel: failed to allocate d_KeyFrames");
    checkCudaError(cudaMalloc(&d_Ow, 3 * sizeof(*d_Ow)), "SearchByProjectionKernel: failed to allocate d_Ow");
    checkCudaError(cudaMalloc(&d_Tcw, 3 * sizeof(*d_Tcw)), "SearchByProjectionKernel: failed to allocate d_Tcw");

    memory_is_initialized = true;
}

void SearchByProjectionKernel::freeMapPointBuffers()
{
    if (mapPointCapacity == 0)
        return;
    cudaFreeHost(h_MapPoints);
    cudaFreeHost(bestDists);
    cudaFreeHost(bestIdxs);
    cudaFree(d_MapPoints);
    cudaFree(d_bestDists);
    cudaFree(d_bestIdxs);
    mapPointCapacity = 0;
}

// bestDists/bestIdxs are indexed as 3 * numPoints by the kernels, hence the factor.
void SearchByProjectionKernel::ensureMapPointCapacity(size_t numMapPoints)
{
    if (numMapPoints > g_maxMapPointVecSize)
        g_maxMapPointVecSize = numMapPoints;

    if (numMapPoints <= mapPointCapacity)
        return;

    const size_t newCapacity = std::max(numMapPoints, std::max(mapPointCapacity * 2, kMapPointCapacity));

    if (mapPointCapacity != 0)
        std::cout << "[SearchByProjectionKernel:] growing map-point buffers: "
                  << mapPointCapacity << " -> " << newCapacity
                  << " (needed " << numMapPoints << ")" << std::endl;

    freeMapPointBuffers();

    checkCudaError(cudaMallocHost((void**)&h_MapPoints, newCapacity * sizeof(*h_MapPoints)), "SearchByProjectionKernel: failed to allocate h_MapPoints");
    checkCudaError(cudaMallocHost((void**)&bestDists, 3 * newCapacity * sizeof(int)), "SearchByProjectionKernel: failed to allocate bestDists");
    checkCudaError(cudaMallocHost((void**)&bestIdxs, 3 * newCapacity * sizeof(int)), "SearchByProjectionKernel: failed to allocate bestIdxs");
    checkCudaError(cudaMalloc(&d_MapPoints, newCapacity * sizeof(*d_MapPoints)), "SearchByProjectionKernel: failed to allocate d_MapPoints");
    checkCudaError(cudaMalloc(&d_bestDists, 3 * newCapacity * sizeof(int)), "SearchByProjectionKernel: failed to allocate d_bestDists");
    checkCudaError(cudaMalloc(&d_bestIdxs, 3 * newCapacity * sizeof(int)), "SearchByProjectionKernel: failed to allocate d_bestIdxs");

    mapPointCapacity = newCapacity;
}

void SearchByProjectionKernel::shutdown() {
    if (!memory_is_initialized)
        return;

    std::cout << "[SearchByProjectionKernel:] map-point high-water mark: "
              << g_maxMapPointVecSize << " / capacity " << mapPointCapacity << std::endl;

    freeMapPointBuffers();

    cudaFreeHost(h_KeyFrames);
    cudaFreeHost(h_Ow);
    cudaFreeHost(h_Tcw);

    cudaFree(d_KeyFrame);
    cudaFree(d_KeyFrames);
    cudaFree(d_Ow);
    cudaFree(d_Tcw);

    memory_is_initialized = false;
}

__device__ inline bool isInImage(MAPPING_DATA_WRAPPER::CudaKeyFrame* keyframe, const float &x, const float &y) {
    return (x>=keyframe->mnMinX && x<keyframe->mnMaxX && y>=keyframe->mnMinY && y<keyframe->mnMaxY);
}

__device__ inline int predictScale(float currentDist, float maxDistance, MAPPING_DATA_WRAPPER::CudaKeyFrame* pKF) {
    float ratio = maxDistance/currentDist;
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

__device__ inline Eigen::Vector2f KannalaBrandt8Project(const Eigen::Vector3f &v3D, float* mvParameters) {
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

__device__ inline Eigen::Vector2f PinholeProject(const Eigen::Vector3f &v3D, float* mvParameters) {
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];
    return res;
}

__global__ void searchByProjectionKernel(Eigen::Vector3f Ow, Sophus::SE3f Tcw,
                            MAPPING_DATA_WRAPPER::CudaKeyFrame *connectedKF, MAPPING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int numPoints, float th,
                            int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints)
        return;
    
    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    MAPPING_DATA_WRAPPER::CudaMapPoint& pMP = mapPoints[idx];

    const float &fx = connectedKF->fx;
    const float &fy = connectedKF->fy;
    const float &cx = connectedKF->cx;
    const float &cy = connectedKF->cy;

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = Tcw * p3Dw;

    if(p3Dc(2)<0.0)
        return;

    const float invz = 1/p3Dc(2);
    const float x1 = p3Dc(0)*invz;
    const float y1 = p3Dc(1)*invz;

    const float u = fx*x1+cx;
    const float v = fy*y1+cy;

    // Point must be inside the image
    if(!isInImage(connectedKF, u, v))
        return;

    const float maxDistance = 1.2f * pMP.mfMaxDistance;
    const float minDistance = 0.8f * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw-Ow;
    const float dist = PO.norm();


    if(dist<minDistance || dist>maxDistance)
        return;

    Eigen::Vector3f Pn = pMP.mNormalVector;

    if(PO.dot(Pn)<0.5*dist)
        return;

    int nPredictedLevel = predictScale(dist, pMP.mfMaxDistance, connectedKF);
    const float radius = th*connectedKF->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = u;
    float y = v;

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
                    // const size_t idx = *vit; idx=temp_idx
                    // if(vpMatched[idx])
                    //     continue;

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

__global__ void searchByProjectionKernel2(Eigen::Vector3f Ow, Sophus::SE3f Tcw,
                            MAPPING_DATA_WRAPPER::CudaKeyFrame *connectedKF, MAPPING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int numPoints, float th,
                            int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints)
        return;
    
    bestDists[idx] = 256;
    bestIdxs[idx] = -1;
    
    MAPPING_DATA_WRAPPER::CudaMapPoint& pMP = mapPoints[idx];

    const float &fx = connectedKF->fx;
    const float &fy = connectedKF->fy;
    const float &cx = connectedKF->cx;
    const float &cy = connectedKF->cy;

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = Tcw * p3Dw;
    

    if(p3Dc(2)<0.0){
        return;
    }

    Eigen::Vector2f uv;
    if(connectedKF->Nleft != -1)
        uv = KannalaBrandt8Project(p3Dc, connectedKF->camera1.mvParameters);
    else if(connectedKF->Nleft == -1)
        uv = PinholeProject(p3Dc, connectedKF->camera1.mvParameters);

    if (!isInImage(connectedKF, uv(0), uv(1))){
        return;
    }

    const float maxDistance = 1.2f * pMP.mfMaxDistance;
    const float minDistance = 0.8f * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw-Ow;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return;
    
    Eigen::Vector3f Pn = pMP.mNormalVector;

    if(PO.dot(Pn)<0.5*dist)
        return;

    int nPredictedLevel = predictScale(dist, pMP.mfMaxDistance, connectedKF);
    const float radius = th*connectedKF->mvScaleFactors[nPredictedLevel];
    
    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];

    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = uv.x();
    float y = uv.y();

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
                    // const size_t idx = *vit; idx=temp_idx
                    // if(vpMatched[idx])
                    //     continue;

                    const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;
                    // printf("Device: idx = %llu, id = %llu, kpLevel = %d\n", idx, pMP.mnId, kpLevel);

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

__global__ void searchByProjectionKernel3(Eigen::Vector3f* Ow, Sophus::SE3f *Tcw,
                            MAPPING_DATA_WRAPPER::CudaKeyFrame** currentCovKFs, MAPPING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int numPoints, float th, int covKFsSize,
                            int* bestDists, int* bestIdxs) 
{
    int totalNumKFs = covKFsSize;
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
    MAPPING_DATA_WRAPPER::CudaKeyFrame *connectedKF = currentCovKFs[connectedKFIdx];

    const float &fx = connectedKF->fx;
    const float &fy = connectedKF->fy;
    const float &cx = connectedKF->cx;
    const float &cy = connectedKF->cy;

    Sophus::SE3f currTcw = Tcw[connectedKFIdx];
    Eigen::Vector3f currOw = Ow[connectedKFIdx];

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = currTcw * p3Dw;

    if(p3Dc(2)<0.0){
        return;
    }

    Eigen::Vector2f uv;
    if(connectedKF->Nleft != -1)
        uv = KannalaBrandt8Project(p3Dc, connectedKF->camera1.mvParameters);
    else if(connectedKF->Nleft == -1)
        uv = PinholeProject(p3Dc, connectedKF->camera1.mvParameters);

    if (!isInImage(connectedKF, uv(0), uv(1))){
        return;
    }

    const float maxDistance = 1.2f * pMP.mfMaxDistance;
    const float minDistance = 0.8f * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw-currOw;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return;
    
    Eigen::Vector3f Pn = pMP.mNormalVector;

    if(PO.dot(Pn)<0.5*dist)
        return;

    int nPredictedLevel = predictScale(dist, pMP.mfMaxDistance, connectedKF);
    const float radius = th*connectedKF->mvScaleFactors[nPredictedLevel];
    
    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];

    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = uv.x();
    float y = uv.y();

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
                    // const size_t idx = *vit; idx=temp_idx
                    // if(vpMatched[idx])
                    //     continue;

                    const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;
                    // printf("Device: idx = %llu, id = %llu, kpLevel = %d\n", idx, pMP.mnId, kpLevel);

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

__global__ void mergedSearchByProjectionKernel(Eigen::Vector3f Ow1, Sophus::SE3f Tcw1,
                            MAPPING_DATA_WRAPPER::CudaKeyFrame *connectedKF, MAPPING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int numPoints, float th1, float th, 
                            int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIdx = numPoints * 2;
    int sectionIdx = idx % 2;
    int mapPointIdx = idx / 2;

    if (idx >= maxIdx || sectionIdx >= 2 || mapPointIdx >= numPoints)
        return;
    
    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    MAPPING_DATA_WRAPPER::CudaMapPoint& pMP = mapPoints[mapPointIdx];
    float base_th;
    if(sectionIdx == 0)
        base_th = th1;
    else
        base_th = th;

    const float &fx = connectedKF->fx;
    const float &fy = connectedKF->fy;
    const float &cx = connectedKF->cx;
    const float &cy = connectedKF->cy;

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc1 = Tcw1 * p3Dw;

    int nPredictedLevel1;

    if(p3Dc1(2)<0.0){
        return;
    }
    
    Eigen::Vector2f uv;
    float u, v;

    if(sectionIdx == 0){
        if(connectedKF->Nleft != -1)
            uv = KannalaBrandt8Project(p3Dc1, connectedKF->camera1.mvParameters);
        if(connectedKF->Nleft == -1)
            uv = PinholeProject(p3Dc1, connectedKF->camera1.mvParameters);

        if (!isInImage(connectedKF, uv(0), uv(1)))
            return;
        
        u = uv.x();
        v = uv.y();
    }
    
    
    if(sectionIdx == 1){
        const float invz = 1/p3Dc1(2);
        const float x1 = p3Dc1(0)*invz;
        const float y1 = p3Dc1(1)*invz;

        u = fx*x1+cx;
        v = fy*y1+cy;

        // Point must be inside the image
        if(!isInImage(connectedKF, u, v))
            return;
    }

    const float maxDistance = 1.2f * pMP.mfMaxDistance;
    const float minDistance = 0.8f * pMP.mfMinDistance;
    Eigen::Vector3f Pn = pMP.mNormalVector;
    float dist1;

    Eigen::Vector3f PO1 = p3Dw-Ow1;
    dist1 = PO1.norm();

    if(dist1<minDistance || dist1>maxDistance)
        return;
    
    if(PO1.dot(Pn)<0.5*dist1)
        return;
    
    float radius1;
    
    nPredictedLevel1 = predictScale(dist1, pMP.mfMaxDistance, connectedKF);
    radius1 = base_th*connectedKF->mvScaleFactors[nPredictedLevel1];
    
    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];

    int bestDist1 = 256;
    int bestIdx1 = -1;

    float factorX1 = radius1;
    float factorY1 = radius1;
    float x1 = u;
    float y1 = v;

    const int nMinCellX = max(0,(int)floor((x1 - connectedKF->mnMinX - factorX1) * connectedKF->mfGridElementWidthInv));
    if (nMinCellX >= connectedKF->mnGridCols) 
        return;

    const int nMaxCellX = min((int)connectedKF->mnGridCols-1,(int)ceil((x1 - connectedKF->mnMinX + factorX1) * connectedKF->mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y1 - connectedKF->mnMinY - factorY1) * connectedKF->mfGridElementHeightInv));
    if (nMinCellY >= connectedKF->mnGridRows)
        return;
    
    const int nMaxCellY = min((int)connectedKF->mnGridRows-1,(int)ceil((y1 - connectedKF->mnMinY + factorY1) * connectedKF->mfGridElementHeightInv));
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
                const float distx = kpUn.ptx-x1;
                const float disty = kpUn.pty-y1;

                if (fabs(distx) < radius1 && fabs(disty) < radius1) {
                    // const size_t idx = *vit; idx=temp_idx
                    // if(vpMatched[idx])
                    //     continue;

                    const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;

                    if (kpLevel < nPredictedLevel1-1 || kpLevel > nPredictedLevel1)
                        continue;

                    const uint8_t* dKF = &connectedKF->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

                    int dist1 = DescriptorDistance(MPdescriptor,dKF);

                    if (dist1<bestDist1) {
                        bestDist1 = dist1;
                        bestIdx1 = temp_idx;
                    }
                }
            }
        }
    }
    
    bestDists[idx] = bestDist1;
    bestIdxs[idx] = bestIdx1;
}

int SearchByProjectionKernel::launch(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming) {
    
#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized)
        initialize();

    int numValidPoints = 0;
    const int TH_LOW = 50;
    int nmatches=0;

    size_t mapPointVecSize = vpPoints.size();
    ensureMapPointCapacity(mapPointVecSize);

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();
    
    // Set of MapPoints already found in the KeyFrame
    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        if (!pMP || pMP->isBad())
            continue;
        else {
            h_MapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
            numValidPoints++;
        }
    }

    MAPPING_DATA_WRAPPER::CudaKeyFrame* tempKF = CudaKeyFrameAllocator::create(pKF);

    const cudaStream_t stream = cudaStreamPerThread;
    cudaMemcpyAsync(d_KeyFrame, tempKF, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyDeviceToDevice, stream);
    
    cudaMemcpyAsync(d_MapPoints, h_MapPoints, mapPointVecSize * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (mapPointVecSize + threads - 1) / threads;
    searchByProjectionKernel<<<blocks, threads, 0, stream>>>(Ow, Tcw,
                                        d_KeyFrame, d_MapPoints, 
                                        mapPointVecSize, th, 
                                        d_bestDists, d_bestIdxs);

    checkCudaError(cudaMemcpyAsync(bestDists, d_bestDists, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestDists back to host2");
    checkCudaError(cudaMemcpyAsync(bestIdxs, d_bestIdxs, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestIdxs back to host");
    cudaStreamSynchronize(stream); // ensure kernel errors propagate
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);
    // gpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    // cpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
        
    // for (int i = 0; i < numValidPoints; i++) {
    //     ORB_SLAM3::MapPoint* pMP = vpPoints[i];
    //     if(spAlreadyFound.count(pMP))
    //             continue;

    //     if(bestDists[i] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist: " << bestDists[i] << ", bestIdx: " << bestIdxs[i] << ")\n";
    // }
    // gpuOutFile << "\n\n";
    
    // origSearchByProjection(pKF, Scw, vpPoints, vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming);

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";


    for(size_t iMP = 0; iMP < mapPointVecSize; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        ORB_SLAM3::KeyFrame* pKFi = vpPointsKFs[iMP];

        if (!pMP || pMP->isBad())
            continue;

        int bestDist = bestDists[iMP];
        int bestIdx = bestIdxs[iMP];

        if (bestDist == 256 || bestIdx == -1)
            continue;

        if (spAlreadyFound.count(pMP))
            continue;

        if (bestDist <= TH_LOW*ratioHamming) {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }
    }

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    launch_1_time.push_back(total);
#endif

    return nmatches;
}

int SearchByProjectionKernel::launch(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming)
{

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized)
        initialize();


    int numValidPoints = 0;
    const int TH_LOW = 50;
    int nmatches=0;

    size_t mapPointVecSize = vpPoints.size();
    ensureMapPointCapacity(mapPointVecSize);


    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        if (!pMP || pMP->isBad()) 
            continue;
        else {
            h_MapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
            numValidPoints++;
        }
    }


    MAPPING_DATA_WRAPPER::CudaKeyFrame* tempKF = CudaKeyFrameAllocator::create(pKF);

    const cudaStream_t stream = cudaStreamPerThread;

    cudaMemcpyAsync(d_KeyFrame, tempKF, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_MapPoints, h_MapPoints, mapPointVecSize * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice, stream);
    
    int threads = 256;
    int blocks = (mapPointVecSize + threads - 1) / threads;
    searchByProjectionKernel2<<<blocks, threads, 0, stream>>>(Ow, Tcw,
                                        d_KeyFrame, d_MapPoints, 
                                        mapPointVecSize, th, 
                                        d_bestDists, d_bestIdxs);
    

    checkCudaError(cudaMemcpyAsync(bestDists, d_bestDists, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestDists back to host3");
    checkCudaError(cudaMemcpyAsync(bestIdxs, d_bestIdxs, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestIdxs back to host");

    cudaStreamSynchronize(stream);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);
    // gpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    // cpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;

    // for (int i = 0; i < numValidPoints; i++) {
    //     ORB_SLAM3::MapPoint* pMP = vpPoints[i];
    //     if(spAlreadyFound.count(pMP))
    //             continue;

    //     if(bestDists[i] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist1: " << bestDists[i] << ", bestIdx1: " << bestIdxs[i] << ")\n";
    // }
    // gpuOutFile << "\n\n";
    
    // origSearchByProjection2(pKF, Scw, vpPoints, vpMatched, th, ratioHamming);

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";

    int a = TH_LOW*ratioHamming;

    for(size_t iMP = 0; iMP < mapPointVecSize; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];

        if (!pMP || pMP->isBad())
            continue;
        
        int bestDist = bestDists[iMP];
        int bestIdx = bestIdxs[iMP];

        if (bestDist == 256 || bestIdx == -1)
            continue;
        
        if (spAlreadyFound.count(pMP))
            continue;

        if (bestDist <= a) {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    launch_2_time.push_back(total);
#endif

    return nmatches;
}

void SearchByProjectionKernel::mergedlaunch(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, Sophus::Sim3<float> &Scw1,
                        const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                        std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                        int &numProjMatches, int &numProjOptMatches)
{

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized)
        initialize();

    int numValidPoints = 0;
    const int TH_LOW = 50;
    numProjOptMatches = 0;
    numProjMatches = 0;

    size_t mapPointVecSize = vpPoints.size();
    ensureMapPointCapacity(mapPointVecSize);
    
    Sophus::SE3f Tcw1 = Sophus::SE3f(Scw1.rotationMatrix(),Scw1.translation()/Scw1.scale());
    Eigen::Vector3f Ow1 = Tcw1.inverse().translation();
    
    // omp_set_num_threads(4);
    // #pragma omp parallel for
    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        if (!pMP || pMP->isBad())
            continue;
        else {
            h_MapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
            numValidPoints++;
        }
    }
    
    MAPPING_DATA_WRAPPER::CudaKeyFrame* d_KeyFrame = CudaKeyFrameAllocator::create(pKF);
    // cudaMemcpy(d_KeyFrame, tempKF, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyDeviceToDevice);

    const cudaStream_t stream = cudaStreamPerThread;
    cudaMemcpyAsync(d_MapPoints, h_MapPoints, numValidPoints * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice, stream); //todo2

    int threads = 256;
    int blocks = (2 * numValidPoints + threads - 1) / threads;
    mergedSearchByProjectionKernel<<<blocks, threads, 0, stream>>>(Ow1, Tcw1,
                                        d_KeyFrame, d_MapPoints, 
                                        numValidPoints, th1, th, 
                                        d_bestDists, d_bestIdxs);
    

    checkCudaError(cudaMemcpyAsync(bestDists, d_bestDists, 2 * numValidPoints * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestDists back to host4"); //todo3
    checkCudaError(cudaMemcpyAsync(bestIdxs, d_bestIdxs, 2 * numValidPoints * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestIdxs back to host"); //todo4
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);
    // gpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    // cpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;

    // for (int i = 0; i < numValidPoints; i++) {
    //     ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        
    //     if(spAlreadyFound.count(pMP))
    //             continue;

    //     if(bestDists[2*i+1] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist: " << bestDists[2*i+1] << ", bestIdx: " << bestIdxs[2*i+1] << ")\n";
    // }
    // gpuOutFile << "\n\n";
    
    // for (int i = 0; i < numValidPoints; i++) {
    //     ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        
    //     if(spAlreadyFound.count(pMP))
    //             continue;
    
    //     if(bestDists[2*i] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist1: " << bestDists[2*i] << ", bestIdx1: " << bestIdxs[2*i] << ")\n";
    // }
    // gpuOutFile << "\n\n";

    // origSearchByProjection(pKF, Scw1, vpPoints, vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming);
    // origSearchByProjection2(pKF, Scw1, vpPoints, vpMatched1, th1, ratioHamming1);

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";

    int a = TH_LOW * ratioHamming;
    int b = TH_LOW*ratioHamming1;

    for(size_t iMP = 0; iMP < mapPointVecSize; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        ORB_SLAM3::KeyFrame* pKFi = vpPointsKFs[iMP];

        if (!pMP || pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
        
        int bestDist = bestDists[2*iMP+1];
        int bestIdx = bestIdxs[2*iMP+1];

        if (bestDist != 256 && bestIdx != -1 && bestDist <= a)
        {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            numProjMatches++;
        }

        int bestDist1 = bestDists[2*iMP];
        int bestIdx1 = bestIdxs[2*iMP];

        if (bestDist1 != 256 && bestIdx1 != -1 && bestDist1 <= b)
        {
            vpMatched1[bestIdx1] = pMP;
            numProjOptMatches++;
        }
    }

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    merged_launch_1_time.push_back(total);
#endif
}

void SearchByProjectionKernel::mergedlaunch(vector<ORB_SLAM3::KeyFrame*> currentCovKFs, vector<Sophus::Sim3f> currentCovmScws, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                        int th, float ratioHamming, int* num_matches, int covKFsSize)
{

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized)
        initialize();

    int numValidPoints = 0;
    const int TH_LOW = 50;

    size_t mapPointVecSize = vpPoints.size();
    ensureMapPointCapacity(mapPointVecSize);

    for (size_t i = 0; i<covKFsSize; i++) {
        h_Tcw[i] = Sophus::SE3f(currentCovmScws[i].rotationMatrix(),currentCovmScws[i].translation()/currentCovmScws[i].scale());
        h_Ow[i] = h_Tcw[i].inverse().translation();
    }

    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        if (!pMP || pMP->isBad()) 
            continue;
        else {
            h_MapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
            numValidPoints++;
        }
    }
    
    for (int i=0; i<covKFsSize; i++){
        ORB_SLAM3::KeyFrame* pKF = currentCovKFs[i];
        h_KeyFrames[i] = CudaKeyFrameAllocator::create(pKF);
    }
    
    const cudaStream_t stream = cudaStreamPerThread;
    cudaMemcpyAsync(d_MapPoints, h_MapPoints, numValidPoints * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice, stream); //todo2
    cudaMemcpyAsync(d_KeyFrames, h_KeyFrames, covKFsSize * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Ow, h_Ow, covKFsSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Tcw, h_Tcw, covKFsSize * sizeof(Sophus::SE3f), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (covKFsSize * numValidPoints + threads - 1) / threads;

    searchByProjectionKernel3<<<blocks, threads, 0, stream>>>(d_Ow, d_Tcw,
                                        d_KeyFrames, d_MapPoints,
                                        numValidPoints, th, covKFsSize, 
                                        d_bestDists, d_bestIdxs);
    
    checkCudaError(cudaMemcpyAsync(bestDists, d_bestDists, numValidPoints * covKFsSize * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestDists back to host6");
    checkCudaError(cudaMemcpyAsync(bestIdxs, d_bestIdxs, numValidPoints * covKFsSize * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to copy d_bestIdxs back to host");

    cudaStreamSynchronize(stream);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }


    
    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
        
    // for (int iKF = 0; iKF < covKFsSize; iKF++) {
    //     gpuOutFile << "================================= Connected KF: " << currentCovKFs[iKF]->mnId << " =================================\n";
    //     cpuOutFile << "================================= Connected KF: " << currentCovKFs[iKF]->mnId << " =================================\n";
    //     cpuOutFile.flush();

    //     const set<ORB_SLAM3::MapPoint*> spAlreadyFound = currentCovKFs[iKF]->GetMapPoints();

    //     for (int i = 0; i < numValidPoints; i++) {
    //         ORB_SLAM3::MapPoint* pMP = vpPoints[i];
    //         int idx = iKF*numValidPoints + i;
    //         if(spAlreadyFound.count(pMP))
    //                 continue;

    //         if(bestDists[idx] != 256)
    //             gpuOutFile << "(i: " << i << ", bestDist1: " << bestDists[idx] << ", bestIdx1: " << bestIdxs[idx] << ")\n";
    //     }
    //     gpuOutFile << "\n\n";
    //     origSearchByProjection2(currentCovKFs[iKF], currentCovmScws[iKF], vpPoints, vpMatched0, th, ratioHamming);
    // }

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";


    for (int iKF = 0; iKF < covKFsSize; iKF++)
    {
        int nmatches=0;
        const set<ORB_SLAM3::MapPoint*> spAlreadyFound = currentCovKFs[iKF]->GetMapPoints();

        for (int iMP = 0; iMP < numValidPoints; iMP++)
        {
            ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];

            int idx = iKF*numValidPoints + iMP;

            if (!pMP || pMP->isBad())
                continue;
            
            int bestDist = bestDists[idx];
            int bestIdx = bestIdxs[idx];

            if (bestDist == 256 || bestIdx == -1)
                continue;

            if(spAlreadyFound.count(pMP))
                continue;

            if (bestDist <= TH_LOW*ratioHamming) {
                // vpMatched[bestIdx] = pMP; //todo
                nmatches++;
            }
        }
        num_matches[iKF] = nmatches;
    }

#ifdef REGISTER_LOOP_CLOSING_STATS
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    merged_launch_2_time.push_back(total);
#endif
}

void SearchByProjectionKernel::origSearchByProjection(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                                       std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
    
    std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    int validMapPointCounter = -1;

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    set<ORB_SLAM3::MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<ORB_SLAM3::MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        ORB_SLAM3::KeyFrame* pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
        validMapPointCounter++;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0){
            continue;
        }

        // Project into Image
        const float invz = 1/p3Dc(2);
        const float x = p3Dc(0)*invz;
        const float y = p3Dc(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        // cpuOutFile << "Host: idx = " << validMapPointCounter << ", id = " << pMP->mnId << ", radius = " << radius << endl;
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();
        
        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // cpuOutFile << "Host: idx = " << 2 * validMapPointCounter + 1 << ", id = " << pMP->mnId << " sectionIdx = 1, " << "bestDist = " << bestDist << endl;


        if (bestDist != 256)
            cpuOutFile << "(i: " << validMapPointCounter << ", bestDist: " << bestDist << ", bestIdx: " << bestIdx << ")\n";
    }
    cpuOutFile << "\n\n";         
}

void SearchByProjectionKernel::origSearchByProjection2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming)
{
    std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    int validMapPointCounter = -1;

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    set<ORB_SLAM3::MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<ORB_SLAM3::MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
        validMapPointCounter++;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;
        
        // Depth must be positive
        if(p3Dc(2)<0.0){
            continue;
        }

        // Project into Image
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();
        
        // cpuOutFile << "Host: idx = " << validMapPointCounter << ", id = " << pMP->mnId << endl;

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        
        // cpuOutFile << "Host: idx = " << 2 * validMapPointCounter << ", id = " << pMP->mnId << " sectionIdx = 0, " << "bestDist = " << bestDist << endl;


        if (bestDist != 256)
            cpuOutFile << "(i: " << validMapPointCounter << ", bestDist1: " << bestDist << ", bestIdx1: " << bestIdx << ")\n";
    }
    cpuOutFile << "\n\n";         
}
    
int SearchByProjectionKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
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

void SearchByProjectionKernel::saveStats(const std::string &file_path) {
    std::string data_path = file_path + "/SearchByProjectionKernel/";
    std::cout << "[SearchByProjectionKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[SearchByProjectionKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }
    std::ofstream myfile;
    
    myfile.open(data_path + "/launch_1_time.txt");
    for (const auto& p : launch_1_time) {
        myfile << p << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/launch_2_time.txt");
    for (const auto& p : launch_2_time) {
        myfile << p << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/merged_launch_1_time.txt");
    for (const auto& p : merged_launch_1_time) {
        myfile << p << std::endl;
    }
    myfile.close();
    
    myfile.open(data_path + "/merged_launch_2_time.txt");
    for (const auto& p : merged_launch_2_time) {
        myfile << p << std::endl;
    }
    myfile.close();
}