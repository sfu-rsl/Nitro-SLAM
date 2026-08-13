#include <iostream>
#include <vector>

#include "Kernels/TriangulationMatchKernel.h"
#include "Kernels/CudaKeyFrameAllocator.h"
#include "Kernels/CudaUtils.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "CameraModels/GeometricCamera.h"
#include "sophus/se3.hpp"
#include <Eigen/Dense>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

// Mirrors ORBmatcher::TH_LOW. The search accepts a pair only if the descriptor
// distance is below this and is the best seen for that keypoint.
#define TRI_TH_LOW 50

namespace {

// ---------------------------------------------------------------------------
// Device helpers. Self-contained so this translation unit does not depend on the
// original kernel's device code, which is kept intact and unused by this path.
// ---------------------------------------------------------------------------

__device__ inline int descriptorDistance(const uint8_t* a, const uint8_t* b)
{
    int dist = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int va, vb;
        memcpy(&va, a + i*4, sizeof(unsigned int));
        memcpy(&vb, b + i*4, sizeof(unsigned int));
        dist += __popc(va ^ vb);
    }
    return dist;
}

// Kannala-Brandt projection, matching KannalaBrandt8::project.
__device__ inline Eigen::Vector2f kbProject(const Eigen::Vector3f &v3D, const float* p)
{
    const float x2_plus_y2 = v3D[0]*v3D[0] + v3D[1]*v3D[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const float psi   = atan2f(v3D[1], v3D[0]);

    const float t2 = theta*theta;
    const float t3 = theta*t2;
    const float t5 = t3*t2;
    const float t7 = t5*t2;
    const float t9 = t7*t2;
    const float r = theta + p[4]*t3 + p[5]*t5 + p[6]*t7 + p[7]*t9;

    Eigen::Vector2f res;
    res[0] = p[0]*r*cosf(psi) + p[2];
    res[1] = p[1]*r*sinf(psi) + p[3];
    return res;
}

// Kannala-Brandt unprojection, matching KannalaBrandt8::unproject exactly. Note the
// result is a point on the z=1 normalised plane, NOT a unit bearing vector: the linear
// triangulation below consumes r[0]/r[1] as normalised image coordinates, which is only
// valid when z == 1.
__device__ inline Eigen::Vector3f kbUnproject(float u, float v, const float* p, float precision)
{
    const float wx = (u - p[2]) / p[0];
    const float wy = (v - p[3]) / p[1];

    const float kHalfPi = 1.57079632679489661923f;
    float theta_d = sqrtf(wx*wx + wy*wy);
    theta_d = fminf(fmaxf(-kHalfPi, theta_d), kHalfPi);

    float scale = 1.0f;
    if (theta_d > 1e-8f) {
        float theta = theta_d;
        for (int j = 0; j < 10; j++) {
            const float t2 = theta*theta;
            const float t4 = t2*t2;
            const float t6 = t4*t2;
            const float t8 = t4*t4;
            const float k0 = p[4]*t2;
            const float k1 = p[5]*t4;
            const float k2 = p[6]*t6;
            const float k3 = p[7]*t8;
            const float theta_fix = (theta*(1 + k0 + k1 + k2 + k3) - theta_d)
                                  / (1 + 3*k0 + 5*k1 + 7*k2 + 9*k3);
            theta = theta - theta_fix;
            if (fabsf(theta_fix) < precision)
                break;
        }
        scale = tanf(theta) / theta_d;
    }

    return Eigen::Vector3f(wx*scale, wy*scale, 1.0f);
}

// Smallest-eigenvalue eigenvector of a 4x4 symmetric PSD matrix, by cyclic Jacobi.
// Uses the numerically stable rotation of Golub & Van Loan rather than an atan of a
// ratio, which loses the quadrant and can rotate the wrong way when the diagonal
// entries are close.
__device__ inline void smallestEigenVector4(Eigen::Matrix4f M, Eigen::Vector4f &out)
{
    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();

    for (int sweep = 0; sweep < 12; sweep++) {
        float off = 0.0f;
        for (int i = 0; i < 4; i++)
            for (int j = i+1; j < 4; j++)
                off += M(i,j)*M(i,j);
        if (off < 1e-20f)
            break;

        for (int pIdx = 0; pIdx < 3; pIdx++) {
            for (int q = pIdx+1; q < 4; q++) {
                const float apq = M(pIdx,q);
                if (fabsf(apq) < 1e-20f)
                    continue;

                const float tau = (M(q,q) - M(pIdx,pIdx)) / (2.0f*apq);
                const float t   = (tau >= 0.0f)
                                ? ( 1.0f / (tau + sqrtf(1.0f + tau*tau)))
                                : (-1.0f / (-tau + sqrtf(1.0f + tau*tau)));
                const float c = 1.0f / sqrtf(1.0f + t*t);
                const float s = t*c;

                for (int k = 0; k < 4; k++) {
                    const float mkp = M(k,pIdx);
                    const float mkq = M(k,q);
                    M(k,pIdx) = c*mkp - s*mkq;
                    M(k,q)    = s*mkp + c*mkq;
                }
                for (int k = 0; k < 4; k++) {
                    const float mpk = M(pIdx,k);
                    const float mqk = M(q,k);
                    M(pIdx,k) = c*mpk - s*mqk;
                    M(q,k)    = s*mpk + c*mqk;
                }
                for (int k = 0; k < 4; k++) {
                    const float vkp = V(k,pIdx);
                    const float vkq = V(k,q);
                    V(k,pIdx) = c*vkp - s*vkq;
                    V(k,q)    = s*vkp + c*vkq;
                }
            }
        }
    }

    int minIdx = 0;
    float minEig = M(0,0);
    for (int i = 1; i < 4; i++) {
        if (M(i,i) < minEig) { minEig = M(i,i); minIdx = i; }
    }
    out = V.col(minIdx);
}

// Linear triangulation of two bearing vectors, matching GeometricTools::Triangulate.
__device__ inline void triangulateLinear(float p1x, float p1y, float p2x, float p2y,
                                         const Eigen::Matrix<float,3,4> &Tcw1,
                                         const Eigen::Matrix<float,3,4> &Tcw2,
                                         Eigen::Vector3f &x3D)
{
    Eigen::Matrix4f A;
    A.row(0) = p1x*Tcw1.row(2) - Tcw1.row(0);
    A.row(1) = p1y*Tcw1.row(2) - Tcw1.row(1);
    A.row(2) = p2x*Tcw2.row(2) - Tcw2.row(0);
    A.row(3) = p2y*Tcw2.row(2) - Tcw2.row(1);

    Eigen::Vector4f v;
    smallestEigenVector4(A.transpose()*A, v);

    if (fabsf(v(3)) < 1e-12f)
        x3D = v.head<3>();
    else
        x3D = v.head<3>() / v(3);
}

// KannalaBrandt8::TriangulateMatches. Returns depth in camera 1, or a negative code.
__device__ inline float kbTriangulateMatches(float u1, float v1, float u2, float v2,
                                             const float* p1, const float* p2,
                                             const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12,
                                             float sigmaLevel, float unc, float precision)
{
    Eigen::Vector3f r1 = kbUnproject(u1, v1, p1, precision);
    Eigen::Vector3f r2 = kbUnproject(u2, v2, p2, precision);

    const Eigen::Vector3f r21 = R12 * r2;
    const float cosParallaxRays = r1.dot(r21) / (r1.norm() * r21.norm());
    if (cosParallaxRays > 0.9998f)
        return -1.0f;

    Eigen::Matrix<float,3,4> Tcw1;
    Tcw1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

    const Eigen::Matrix3f R21 = R12.transpose();
    Eigen::Matrix<float,3,4> Tcw2;
    Tcw2 << R21, -R21*t12;

    Eigen::Vector3f x3D;
    triangulateLinear(r1[0], r1[1], r2[0], r2[1], Tcw1, Tcw2, x3D);

    const float z1 = x3D(2);
    if (z1 <= 0.0f) return -2.0f;

    const float z2 = R21.row(2).dot(x3D) + Tcw2(2,3);
    if (z2 <= 0.0f) return -3.0f;

    const Eigen::Vector2f uv1 = kbProject(x3D, p1);
    const float ex1 = uv1(0) - u1, ey1 = uv1(1) - v1;
    if ((ex1*ex1 + ey1*ey1) > 5.991f*sigmaLevel) return -4.0f;

    const Eigen::Vector3f x3D2 = R21*x3D + Tcw2.col(3);
    const Eigen::Vector2f uv2 = kbProject(x3D2, p2);
    const float ex2 = uv2(0) - u2, ey2 = uv2(1) - v2;
    if ((ex2*ex2 + ey2*ey2) > 5.991f*unc) return -5.0f;

    return z1;
}

__device__ inline Eigen::Matrix3f invert3(const Eigen::Matrix3f &M)
{
    const float a = M(0,0), b = M(0,1), c = M(0,2);
    const float d = M(1,0), e = M(1,1), f = M(1,2);
    const float g = M(2,0), h = M(2,1), i = M(2,2);
    const float det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    // A camera K is never singular; guard only against a NaN blow-up.
    const float invDet = (fabsf(det) > 0.0f) ? (1.0f/det) : 0.0f;

    Eigen::Matrix3f inv;
    inv(0,0) =  (e*i - f*h)*invDet;  inv(0,1) = -(b*i - c*h)*invDet;  inv(0,2) =  (b*f - c*e)*invDet;
    inv(1,0) = -(d*i - f*g)*invDet;  inv(1,1) =  (a*i - c*g)*invDet;  inv(1,2) = -(a*f - c*d)*invDet;
    inv(2,0) =  (d*h - e*g)*invDet;  inv(2,1) = -(a*h - b*g)*invDet;  inv(2,2) =  (a*e - b*d)*invDet;
    return inv;
}

// Pinhole::epipolarConstrain - point-to-epipolar-line distance against the fundamental
// matrix.
__device__ inline bool pinholeEpipolar(const Eigen::Matrix3f &K1, const Eigen::Matrix3f &K2,
                                       float u1, float v1, float u2, float v2,
                                       const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12,
                                       float unc)
{
    const Eigen::Matrix3f t12x = Sophus::SO3f::hat(t12);
    const Eigen::Matrix3f F12 = invert3(K1.transpose()) * t12x * R12 * invert3(K2);

    const float a = u1*F12(0,0) + v1*F12(1,0) + F12(2,0);
    const float b = u1*F12(0,1) + v1*F12(1,1) + F12(2,1);
    const float c = u1*F12(0,2) + v1*F12(1,2) + F12(2,2);

    const float num = a*u2 + b*v2 + c;
    const float den = a*a + b*b;
    if (den == 0.0f)
        return false;

    return (num*num / den) < 3.84f*unc;
}

// ---------------------------------------------------------------------------
// Search kernel: one thread per (neighbour, idx1) work item.
// ---------------------------------------------------------------------------

__global__ void triangulationMatchKernel(
    MAPPING_DATA_WRAPPER::CudaKeyFrame*  kf1,
    MAPPING_DATA_WRAPPER::CudaKeyFrame** kf2s,
    const int* workNeighbour, const int* workIdx1, const int* workNode2, int nWork,
    const Eigen::Matrix3f* Rll, const Eigen::Matrix3f* Rlr,
    const Eigen::Matrix3f* Rrl, const Eigen::Matrix3f* Rrr,
    const Eigen::Vector3f* tll, const Eigen::Vector3f* tlr,
    const Eigen::Vector3f* trl, const Eigen::Vector3f* trr,
    const Eigen::Vector2f* eps,
    const bool* mpExists1, const bool* mpExists2,
    int matchStride, int mp2Stride, bool bFisheyeModel, bool bCoarse, float camPrecision,
    int* matches)
{
    for (int w = blockIdx.x*blockDim.x + threadIdx.x; w < nWork; w += gridDim.x*blockDim.x) {

        const int s    = workNeighbour[w];
        const int idx1 = workIdx1[w];
        const int k2   = workNode2[w];

        if (mpExists1[idx1])
            continue;

        MAPPING_DATA_WRAPPER::CudaKeyFrame* kf2 = kf2s[s];

        const int n2Begin = (k2 == 0) ? 0 : kf2->mFeatVecStartIndexes[k2-1];
        const int n2End   = kf2->mFeatVecStartIndexes[k2];

        const bool twoCam1 = kf1->camera2.isAvailable;
        const bool twoCam2 = kf2->camera2.isAvailable;

        const bool bStereo1 = (!twoCam1 && kf1->mvuRight[idx1] >= 0);
        const bool bRight1  = (kf1->Nleft == -1 || idx1 < kf1->Nleft) ? false : true;
        const MAPPING_DATA_WRAPPER::CudaKeyPoint kp1 =
            (kf1->Nleft == -1) ? kf1->mvKeysUn[idx1]
                               : (idx1 < kf1->Nleft) ? kf1->mvKeys[idx1]
                                                     : kf1->mvKeysRight[idx1 - kf1->Nleft];
        const uint8_t* d1 = kf1->mDescriptors + idx1*32;

        const float sigma1 = kf1->mvScaleFactors[kp1.octave]*kf1->mvScaleFactors[kp1.octave];

        int bestDist = TRI_TH_LOW;
        int bestIdx2 = -1;

        for (int j = n2Begin; j < n2End; j++) {
            const int idx2 = (int) kf2->mFeatVec[j];

            if (mpExists2[s*mp2Stride + idx2])
                continue;

            const bool bStereo2 = (!twoCam2 && kf2->mvuRight[idx2] >= 0);
            const uint8_t* d2 = kf2->mDescriptors + idx2*32;

            const int dist = descriptorDistance(d1, d2);
            if (dist > TRI_TH_LOW || dist > bestDist)
                continue;

            const bool bRight2 = (kf2->Nleft == -1 || idx2 < kf2->Nleft) ? false : true;
            const MAPPING_DATA_WRAPPER::CudaKeyPoint kp2 =
                (kf2->Nleft == -1) ? kf2->mvKeysUn[idx2]
                                   : (idx2 < kf2->Nleft) ? kf2->mvKeys[idx2]
                                                         : kf2->mvKeysRight[idx2 - kf2->Nleft];

            // Reject candidates lying on the epipole; only meaningful for a single
            // camera rig, matching the CPU guard.
            if (!bStereo1 && !bStereo2 && !twoCam1) {
                const float distex = eps[s](0) - kp2.ptx;
                const float distey = eps[s](1) - kp2.pty;
                if (distex*distex + distey*distey < 100.0f*kf2->mvScaleFactors[kp2.octave])
                    continue;
            }

            Eigen::Matrix3f R12;
            Eigen::Vector3f t12;
            Eigen::Matrix3f K1 = kf1->camera1.toK, K2 = kf2->camera1.toK;
            const float *P1 = kf1->camera1.mvParameters, *P2 = kf2->camera1.mvParameters;

            if (twoCam1 && twoCam2) {
                if (bRight1 && bRight2) {
                    R12 = Rrr[s]; t12 = trr[s];
                    K1 = kf1->camera2.toK; K2 = kf2->camera2.toK;
                    P1 = kf1->camera2.mvParameters; P2 = kf2->camera2.mvParameters;
                } else if (bRight1 && !bRight2) {
                    R12 = Rrl[s]; t12 = trl[s];
                    K1 = kf1->camera2.toK; K2 = kf2->camera1.toK;
                    P1 = kf1->camera2.mvParameters; P2 = kf2->camera1.mvParameters;
                } else if (!bRight1 && bRight2) {
                    R12 = Rlr[s]; t12 = tlr[s];
                    K1 = kf1->camera1.toK; K2 = kf2->camera2.toK;
                    P1 = kf1->camera1.mvParameters; P2 = kf2->camera2.mvParameters;
                } else {
                    R12 = Rll[s]; t12 = tll[s];
                }
            } else {
                R12 = Rll[s]; t12 = tll[s];
            }

            const float sigma2 = kf2->mvScaleFactors[kp2.octave]*kf2->mvScaleFactors[kp2.octave];

            bool ok;
            if (bFisheyeModel) {
                ok = kbTriangulateMatches(kp1.ptx, kp1.pty, kp2.ptx, kp2.pty, P1, P2,
                                          R12, t12, sigma1, sigma2, camPrecision) > 0.0001f;
            } else {
                ok = pinholeEpipolar(K1, K2, kp1.ptx, kp1.pty, kp2.ptx, kp2.pty,
                                     R12, t12, sigma2);
            }

            if (bCoarse || ok) {
                bestIdx2 = idx2;
                bestDist = dist;
            }
        }

        if (bestIdx2 >= 0)
            matches[s*matchStride + idx1] = bestIdx2;
    }
}

} // namespace

// ---------------------------------------------------------------------------

void TriangulationMatchKernel::initialize()
{
    if (memory_is_initialized)
        return;
    CudaKeyFrameAllocator::initialize();
    memory_is_initialized = true;
}

void TriangulationMatchKernel::ensureCapacity(size_t nWork, size_t nNeigh, size_t nMatch)
{
    if (nWork > workCapacity) {
        if (d_workIdx1) { cudaFree(d_workIdx1); cudaFree(d_workNeighbour); cudaFree(d_workNode2); }
        workCapacity = nWork*2 + 1024;
        checkCudaError(cudaMalloc(&d_workIdx1,      workCapacity*sizeof(int)), "TMK: d_workIdx1");
        checkCudaError(cudaMalloc(&d_workNeighbour, workCapacity*sizeof(int)), "TMK: d_workNeighbour");
        checkCudaError(cudaMalloc(&d_workNode2,     workCapacity*sizeof(int)), "TMK: d_workNode2");
    }
    if (nNeigh > neighbourCapacity) {
        if (d_neighKFs) {
            cudaFree(d_neighKFs);
            cudaFree(d_Rll); cudaFree(d_Rlr); cudaFree(d_Rrl); cudaFree(d_Rrr);
            cudaFree(d_tll); cudaFree(d_tlr); cudaFree(d_trl); cudaFree(d_trr);
            cudaFree(d_ep);
        }
        neighbourCapacity = nNeigh*2 + 8;
        checkCudaError(cudaMalloc(&d_neighKFs, neighbourCapacity*sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*)), "TMK: d_neighKFs");
        checkCudaError(cudaMalloc(&d_Rll, neighbourCapacity*sizeof(Eigen::Matrix3f)), "TMK: d_Rll");
        checkCudaError(cudaMalloc(&d_Rlr, neighbourCapacity*sizeof(Eigen::Matrix3f)), "TMK: d_Rlr");
        checkCudaError(cudaMalloc(&d_Rrl, neighbourCapacity*sizeof(Eigen::Matrix3f)), "TMK: d_Rrl");
        checkCudaError(cudaMalloc(&d_Rrr, neighbourCapacity*sizeof(Eigen::Matrix3f)), "TMK: d_Rrr");
        checkCudaError(cudaMalloc(&d_tll, neighbourCapacity*sizeof(Eigen::Vector3f)), "TMK: d_tll");
        checkCudaError(cudaMalloc(&d_tlr, neighbourCapacity*sizeof(Eigen::Vector3f)), "TMK: d_tlr");
        checkCudaError(cudaMalloc(&d_trl, neighbourCapacity*sizeof(Eigen::Vector3f)), "TMK: d_trl");
        checkCudaError(cudaMalloc(&d_trr, neighbourCapacity*sizeof(Eigen::Vector3f)), "TMK: d_trr");
        checkCudaError(cudaMalloc(&d_ep,  neighbourCapacity*sizeof(Eigen::Vector2f)), "TMK: d_ep");
    }
    if (nMatch > matchCapacity) {
        if (d_matches) { cudaFree(d_matches); cudaFree(d_mpExists2); }
        if (d_mpExists1) cudaFree(d_mpExists1);
        matchCapacity = nMatch*2 + 1024;
        checkCudaError(cudaMalloc(&d_matches,   matchCapacity*sizeof(int)),  "TMK: d_matches");
        checkCudaError(cudaMalloc(&d_mpExists2, matchCapacity*sizeof(bool)), "TMK: d_mpExists2");
        checkCudaError(cudaMalloc(&d_mpExists1, matchCapacity*sizeof(bool)), "TMK: d_mpExists1");
    }
}

void TriangulationMatchKernel::launch(
    ORB_SLAM3::KeyFrame* pKF1,
    const std::vector<ORB_SLAM3::KeyFrame*> &vpNeighKFs,
    bool bMonocular, bool bCoarse,
    std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices,
    std::vector<size_t> &vpNeighKFsIndexes)
{
    allvMatchedIndices.clear();
    vpNeighKFsIndexes.clear();

    // ---- select neighbours, exactly as CreateNewMapPoints does before searching ----
    const Eigen::Vector3f Ow1 = pKF1->GetCameraCenter();
    std::vector<ORB_SLAM3::KeyFrame*> kept;
    for (size_t i = 0; i < vpNeighKFs.size(); i++) {
        ORB_SLAM3::KeyFrame* pKF2 = vpNeighKFs[i];
        const float baseline = (pKF2->GetCameraCenter() - Ow1).norm();
        if (!bMonocular) {
            if (baseline < pKF2->mb) continue;
        } else {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            if (baseline/medianDepthKF2 < 0.01f) continue;
        }
        vpNeighKFsIndexes.push_back(i);
        kept.push_back(pKF2);
    }
    if (kept.empty())
        return;

    const size_t nNeigh = kept.size();
    // Index space is the keypoint count, not the ORB feature budget: a fisheye stereo
    // keyframe holds Nleft+Nright keypoints, roughly twice nFeatures_with_th, and any
    // right-camera index past the stride would be dropped.
    const int matchStride = pKF1->N;
    int mp2Stride = 0;
    for (size_t s = 0; s < nNeigh; s++)
        mp2Stride = std::max(mp2Stride, kept[s]->N);

    // ---- per-neighbour relative poses and epipole ----
    std::vector<MAPPING_DATA_WRAPPER::CudaKeyFrame*> h_kf2(nNeigh);
    std::vector<Eigen::Matrix3f> h_Rll(nNeigh), h_Rlr(nNeigh), h_Rrl(nNeigh), h_Rrr(nNeigh);
    std::vector<Eigen::Vector3f> h_tll(nNeigh), h_tlr(nNeigh), h_trl(nNeigh), h_trr(nNeigh);
    std::vector<Eigen::Vector2f> h_ep(nNeigh);

    const Sophus::SE3f T1w = pKF1->GetPose();

    for (size_t s = 0; s < nNeigh; s++) {
        ORB_SLAM3::KeyFrame* pKF2 = kept[s];

        h_kf2[s] = CudaKeyFrameAllocator::create(pKF2);

        const Sophus::SE3f T2w = pKF2->GetPose();
        const Sophus::SE3f Tw2 = pKF2->GetPoseInverse();
        h_ep[s] = pKF2->mpCamera->project(T2w * pKF1->GetCameraCenter());

        const Sophus::SE3f Tll = T1w * Tw2;
        h_Rll[s] = Tll.rotationMatrix();  h_tll[s] = Tll.translation();

        if (pKF1->mpCamera2 && pKF2->mpCamera2) {
            const Sophus::SE3f Tr1w = pKF1->GetRightPose();
            const Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            const Sophus::SE3f Tlr = T1w  * Twr2;
            const Sophus::SE3f Trl = Tr1w * Tw2;
            const Sophus::SE3f Trr = Tr1w * Twr2;
            h_Rlr[s] = Tlr.rotationMatrix();  h_tlr[s] = Tlr.translation();
            h_Rrl[s] = Trl.rotationMatrix();  h_trl[s] = Trl.translation();
            h_Rrr[s] = Trr.rotationMatrix();  h_trr[s] = Trr.translation();
        } else {
            h_Rlr[s] = h_Rrl[s] = h_Rrr[s] = h_Rll[s];
            h_tlr[s] = h_trl[s] = h_trr[s] = h_tll[s];
        }
    }

    // ---- flatten the BoW node intersections into work items ----
    // One item per (neighbour, idx1). A feature belongs to exactly one node at a given
    // level, so items are unique and the kernel needs no atomics. Unbounded: sized by
    // the actual intersection rather than a compile-time cap.
    std::vector<int> h_wNeigh, h_wIdx1, h_wNode2;
    h_wNeigh.reserve(nNeigh*(size_t)matchStride/2);
    h_wIdx1.reserve(nNeigh*(size_t)matchStride/2);
    h_wNode2.reserve(nNeigh*(size_t)matchStride/2);

    const DBoW2::FeatureVector &fv1 = pKF1->mFeatVec;
    for (size_t s = 0; s < nNeigh; s++) {
        const DBoW2::FeatureVector &fv2 = kept[s]->mFeatVec;

        DBoW2::FeatureVector::const_iterator f1it = fv1.begin(), f1end = fv1.end();
        DBoW2::FeatureVector::const_iterator f2it = fv2.begin(), f2end = fv2.end();
        size_t k2 = 0;

        while (f1it != f1end && f2it != f2end) {
            if (f1it->first == f2it->first) {
                for (size_t a = 0; a < f1it->second.size(); a++) {
                    h_wNeigh.push_back((int)s);
                    h_wIdx1.push_back((int)f1it->second[a]);
                    h_wNode2.push_back((int)k2);
                }
                ++f1it; ++f2it; ++k2;
            } else if (f1it->first < f2it->first) {
                f1it = fv1.lower_bound(f2it->first);
            } else {
                f2it = fv2.lower_bound(f1it->first);
                k2 = (size_t) std::distance(fv2.begin(), f2it);
            }
        }
    }

    const size_t nWork = h_wIdx1.size();
    const size_t nMatchSlots  = nNeigh*(size_t)matchStride;
    const size_t nMp2Slots    = nNeigh*(size_t)mp2Stride;

    allvMatchedIndices.assign(nNeigh, {});
    if (nWork == 0)
        return;

    // ---- map point snapshots ----
    std::vector<uint8_t> h_mp1(matchStride, 0), h_mp2(nMp2Slots, 0);
    for (int i = 0; i < matchStride; i++)
        h_mp1[i] = pKF1->GetMapPoint(i) ? 1 : 0;
    for (size_t s = 0; s < nNeigh; s++)
        for (int i = 0; i < kept[s]->N; i++)
            h_mp2[s*mp2Stride + i] = kept[s]->GetMapPoint(i) ? 1 : 0;

    ensureCapacity(nWork, nNeigh, std::max(nMatchSlots, nMp2Slots));

    cudaStream_t stream = cudaStreamPerThread;
    checkCudaError(cudaMemcpyAsync(d_workNeighbour, h_wNeigh.data(), nWork*sizeof(int), cudaMemcpyHostToDevice, stream), "TMK: copy wNeigh");
    checkCudaError(cudaMemcpyAsync(d_workIdx1,      h_wIdx1.data(),  nWork*sizeof(int), cudaMemcpyHostToDevice, stream), "TMK: copy wIdx1");
    checkCudaError(cudaMemcpyAsync(d_workNode2,     h_wNode2.data(), nWork*sizeof(int), cudaMemcpyHostToDevice, stream), "TMK: copy wNode2");

    std::vector<MAPPING_DATA_WRAPPER::CudaKeyFrame*> h_kf2gpu(nNeigh);
    for (size_t s = 0; s < nNeigh; s++) h_kf2gpu[s] = h_kf2[s]->gpuAddr;
    checkCudaError(cudaMemcpyAsync(d_neighKFs, h_kf2gpu.data(), nNeigh*sizeof(void*), cudaMemcpyHostToDevice, stream), "TMK: copy kf2");

    checkCudaError(cudaMemcpyAsync(d_Rll, h_Rll.data(), nNeigh*sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice, stream), "TMK: Rll");
    checkCudaError(cudaMemcpyAsync(d_Rlr, h_Rlr.data(), nNeigh*sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice, stream), "TMK: Rlr");
    checkCudaError(cudaMemcpyAsync(d_Rrl, h_Rrl.data(), nNeigh*sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice, stream), "TMK: Rrl");
    checkCudaError(cudaMemcpyAsync(d_Rrr, h_Rrr.data(), nNeigh*sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice, stream), "TMK: Rrr");
    checkCudaError(cudaMemcpyAsync(d_tll, h_tll.data(), nNeigh*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice, stream), "TMK: tll");
    checkCudaError(cudaMemcpyAsync(d_tlr, h_tlr.data(), nNeigh*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice, stream), "TMK: tlr");
    checkCudaError(cudaMemcpyAsync(d_trl, h_trl.data(), nNeigh*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice, stream), "TMK: trl");
    checkCudaError(cudaMemcpyAsync(d_trr, h_trr.data(), nNeigh*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice, stream), "TMK: trr");
    checkCudaError(cudaMemcpyAsync(d_ep,  h_ep.data(),  nNeigh*sizeof(Eigen::Vector2f), cudaMemcpyHostToDevice, stream), "TMK: ep");

    checkCudaError(cudaMemcpyAsync(d_mpExists1, h_mp1.data(), matchStride*sizeof(bool), cudaMemcpyHostToDevice, stream), "TMK: mp1");
    checkCudaError(cudaMemcpyAsync(d_mpExists2, h_mp2.data(), nMp2Slots*sizeof(bool), cudaMemcpyHostToDevice, stream), "TMK: mp2");
    checkCudaError(cudaMemsetAsync(d_matches, 0xFF, nMatchSlots*sizeof(int), stream), "TMK: init matches");

    MAPPING_DATA_WRAPPER::CudaKeyFrame* kf1gpu = CudaKeyFrameAllocator::create(pKF1);

    const bool bFisheye = (pKF1->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE);
    float camPrecision = 1e-6f;

    const int threads = 256;
    const int blocks  = (int) std::min<size_t>((nWork + threads - 1)/threads, 4096);

    triangulationMatchKernel<<<blocks, threads, 0, stream>>>(
        kf1gpu->gpuAddr, d_neighKFs,
        d_workNeighbour, d_workIdx1, d_workNode2, (int)nWork,
        d_Rll, d_Rlr, d_Rrl, d_Rrr, d_tll, d_tlr, d_trl, d_trr, d_ep,
        d_mpExists1, d_mpExists2,
        matchStride, mp2Stride, bFisheye, bCoarse, camPrecision,
        d_matches);

    std::vector<int> h_matches(nMatchSlots);
    checkCudaError(cudaMemcpyAsync(h_matches.data(), d_matches, nMatchSlots*sizeof(int), cudaMemcpyDeviceToHost, stream), "TMK: read matches");
    checkCudaError(cudaStreamSynchronize(stream), "TMK: sync");

    for (size_t s = 0; s < nNeigh; s++) {
        std::vector<std::pair<size_t,size_t>> &out = allvMatchedIndices[s];
        for (int i = 0; i < matchStride; i++) {
            const int m = h_matches[s*matchStride + i];
            if (m >= 0)
                out.push_back(std::make_pair((size_t)i, (size_t)m));
        }
        nMatchesTotal += out.size();
    }
    nCalls++;
    nWorkItemsTotal += nWork;
}

void TriangulationMatchKernel::shutdown()
{
    if (!memory_is_initialized)
        return;

    std::cout << "[TriangulationMatchKernel:] calls=" << nCalls
              << " workItems=" << nWorkItemsTotal
              << " matches=" << nMatchesTotal
              << " avgMatchesPerCall=" << (nCalls ? (double)nMatchesTotal/nCalls : 0.0)
              << std::endl;

    if (d_workIdx1)    { cudaFree(d_workIdx1); cudaFree(d_workNeighbour); cudaFree(d_workNode2); }
    if (d_neighKFs)    {
        cudaFree(d_neighKFs);
        cudaFree(d_Rll); cudaFree(d_Rlr); cudaFree(d_Rrl); cudaFree(d_Rrr);
        cudaFree(d_tll); cudaFree(d_tlr); cudaFree(d_trl); cudaFree(d_trr);
        cudaFree(d_ep);
    }
    if (d_matches)     { cudaFree(d_matches); cudaFree(d_mpExists2); }
    if (d_mpExists1)   cudaFree(d_mpExists1);

    memory_is_initialized = false;
}

void TriangulationMatchKernel::saveStats(const std::string &file_path) {}
