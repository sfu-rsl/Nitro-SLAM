#include <iostream>

#include "Kernels/MapPointCandidateKernel.h"
#include "Kernels/CudaKeyFrameStorage.h"
#include "Kernels/CudaUtils.h"
#include "CameraModels/GeometricCamera.h"
#include "sophus/se3.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <Eigen/Dense>

namespace {

// Per-candidate inputs that are cheap to precompute on the host.
struct CandIn {
    int neighbourSlot;
    int idx1;
    int idx2;
};

// Poses for one camera of one keyframe.
struct CamPose {
    Eigen::Matrix<float,3,4> Tcw;   // world -> camera
    Eigen::Matrix3f          Rwc;   // camera -> world rotation
    Eigen::Vector3f          Ow;    // camera centre in world
};

struct CandOut {
    Eigen::Vector3f x3D;
    int   valid;        // 0/1
    int   stereoPoint;  // 0/1
};

// --- camera models -------------------------------------------------------------

__device__ inline Eigen::Vector3f unprojectPinhole(float u, float v, const float* p)
{
    // p = {fx, fy, cx, cy}
    return Eigen::Vector3f((u - p[2])/p[0], (v - p[3])/p[1], 1.0f);
}

__device__ inline Eigen::Vector2f projectPinhole(const Eigen::Vector3f &x, const float* p)
{
    return Eigen::Vector2f(p[0]*x[0]/x[2] + p[2], p[1]*x[1]/x[2] + p[3]);
}

// KannalaBrandt8::unproject - returns a point on the z=1 plane, not a unit bearing.
__device__ inline Eigen::Vector3f unprojectKB(float u, float v, const float* p, float precision)
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
            const float t2 = theta*theta, t4 = t2*t2, t6 = t4*t2, t8 = t4*t4;
            const float k0 = p[4]*t2, k1 = p[5]*t4, k2 = p[6]*t6, k3 = p[7]*t8;
            const float fix = (theta*(1 + k0 + k1 + k2 + k3) - theta_d)
                            / (1 + 3*k0 + 5*k1 + 7*k2 + 9*k3);
            theta -= fix;
            if (fabsf(fix) < precision) break;
        }
        scale = tanf(theta) / theta_d;
    }
    return Eigen::Vector3f(wx*scale, wy*scale, 1.0f);
}

__device__ inline Eigen::Vector2f projectKB(const Eigen::Vector3f &v3D, const float* p)
{
    const float x2_plus_y2 = v3D[0]*v3D[0] + v3D[1]*v3D[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const float psi   = atan2f(v3D[1], v3D[0]);
    const float t2 = theta*theta, t3 = theta*t2, t5 = t3*t2, t7 = t5*t2, t9 = t7*t2;
    const float r = theta + p[4]*t3 + p[5]*t5 + p[6]*t7 + p[7]*t9;
    return Eigen::Vector2f(p[0]*r*cosf(psi) + p[2], p[1]*r*sinf(psi) + p[3]);
}

// --- linear triangulation ------------------------------------------------------
// GeometricTools::Triangulate uses Eigen's JacobiSVD and takes V.col(3). The same
// vector is the eigenvector of A'A for the smallest eigenvalue, obtained here by
// cyclic Jacobi with the stable Golub & Van Loan rotation.
__device__ inline bool triangulateLinear(const Eigen::Vector3f &xc1, const Eigen::Vector3f &xc2,
                                         const Eigen::Matrix<float,3,4> &Tc1w,
                                         const Eigen::Matrix<float,3,4> &Tc2w,
                                         Eigen::Vector3f &x3D)
{
    Eigen::Matrix4f A;
    A.row(0) = xc1(0)*Tc1w.row(2) - Tc1w.row(0);
    A.row(1) = xc1(1)*Tc1w.row(2) - Tc1w.row(1);
    A.row(2) = xc2(0)*Tc2w.row(2) - Tc2w.row(0);
    A.row(3) = xc2(1)*Tc2w.row(2) - Tc2w.row(1);

    Eigen::Matrix4f M = A.transpose()*A;
    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();

    for (int sweep = 0; sweep < 12; sweep++) {
        float off = 0.0f;
        for (int i = 0; i < 4; i++)
            for (int j = i+1; j < 4; j++) off += M(i,j)*M(i,j);
        if (off < 1e-20f) break;

        for (int p = 0; p < 3; p++) {
            for (int q = p+1; q < 4; q++) {
                const float apq = M(p,q);
                if (fabsf(apq) < 1e-20f) continue;
                const float tau = (M(q,q) - M(p,p)) / (2.0f*apq);
                const float t = (tau >= 0.0f) ?  1.0f/(tau + sqrtf(1.0f + tau*tau))
                                              : -1.0f/(-tau + sqrtf(1.0f + tau*tau));
                const float c = 1.0f/sqrtf(1.0f + t*t);
                const float s = t*c;
                for (int k = 0; k < 4; k++) {
                    const float mkp = M(k,p), mkq = M(k,q);
                    M(k,p) = c*mkp - s*mkq;  M(k,q) = s*mkp + c*mkq;
                }
                for (int k = 0; k < 4; k++) {
                    const float mpk = M(p,k), mqk = M(q,k);
                    M(p,k) = c*mpk - s*mqk;  M(q,k) = s*mpk + c*mqk;
                }
                for (int k = 0; k < 4; k++) {
                    const float vkp = V(k,p), vkq = V(k,q);
                    V(k,p) = c*vkp - s*vkq;  V(k,q) = s*vkp + c*vkq;
                }
            }
        }
    }

    int minIdx = 0; float minEig = M(0,0);
    for (int i = 1; i < 4; i++) if (M(i,i) < minEig) { minEig = M(i,i); minIdx = i; }

    const Eigen::Vector4f h = V.col(minIdx);
    if (h(3) == 0.0f)
        return false;
    x3D = h.head<3>() / h(3);
    return true;
}

// --- candidate evaluation ------------------------------------------------------

__global__ void mapPointCandidateKernel(
    MAPPING_DATA_WRAPPER::CudaKeyFrame*  kf1,
    MAPPING_DATA_WRAPPER::CudaKeyFrame** kf2s,
    const CandIn* in, int nCand,
    const CamPose* pose1,           // [0] left, [1] right
    const CamPose* pose2,           // 2 per neighbour: [2s] left, [2s+1] right
    const float* mb2, float mb1,
    bool bInertial, bool bFarPoints, float thFarPoints, float ratioFactor,
    bool bFisheye, float camPrecision,
    CandOut* out)
{
    for (int t = blockIdx.x*blockDim.x + threadIdx.x; t < nCand; t += gridDim.x*blockDim.x) {

        out[t].valid = 0;
        out[t].stereoPoint = 0;

        const int s    = in[t].neighbourSlot;
        const int idx1 = in[t].idx1;
        const int idx2 = in[t].idx2;

        MAPPING_DATA_WRAPPER::CudaKeyFrame* kf2 = kf2s[s];

        const bool twoCam1 = kf1->camera2.isAvailable;
        const bool twoCam2 = kf2->camera2.isAvailable;

        const MAPPING_DATA_WRAPPER::CudaKeyPoint kp1 =
            (kf1->Nleft == -1) ? kf1->mvKeysUn[idx1]
                               : (idx1 < kf1->Nleft) ? kf1->mvKeys[idx1]
                                                     : kf1->mvKeysRight[idx1 - kf1->Nleft];
        const float kp1_ur  = kf1->mvuRight[idx1];
        const bool  bStereo1 = (!twoCam1 && kp1_ur >= 0);
        const bool  bRight1  = (kf1->Nleft == -1 || idx1 < kf1->Nleft) ? false : true;

        const MAPPING_DATA_WRAPPER::CudaKeyPoint kp2 =
            (kf2->Nleft == -1) ? kf2->mvKeysUn[idx2]
                               : (idx2 < kf2->Nleft) ? kf2->mvKeys[idx2]
                                                     : kf2->mvKeysRight[idx2 - kf2->Nleft];
        const float kp2_ur  = kf2->mvuRight[idx2];
        const bool  bStereo2 = (!twoCam2 && kp2_ur >= 0);
        const bool  bRight2  = (kf2->Nleft == -1 || idx2 < kf2->Nleft) ? false : true;

        // Select the camera/pose pair, mirroring the CPU's four-way choice.
        const CamPose &P1 = (twoCam1 && twoCam2 && bRight1) ? pose1[1] : pose1[0];
        const CamPose &P2 = (twoCam1 && twoCam2 && bRight2) ? pose2[2*s+1] : pose2[2*s];
        const float *C1 = (twoCam1 && twoCam2 && bRight1) ? kf1->camera2.mvParameters
                                                          : kf1->camera1.mvParameters;
        const float *C2 = (twoCam1 && twoCam2 && bRight2) ? kf2->camera2.mvParameters
                                                          : kf2->camera1.mvParameters;

        // --- parallax ---
        const Eigen::Vector3f xn1 = bFisheye ? unprojectKB(kp1.ptx, kp1.pty, C1, camPrecision)
                                             : unprojectPinhole(kp1.ptx, kp1.pty, C1);
        const Eigen::Vector3f xn2 = bFisheye ? unprojectKB(kp2.ptx, kp2.pty, C2, camPrecision)
                                             : unprojectPinhole(kp2.ptx, kp2.pty, C2);

        const Eigen::Vector3f ray1 = P1.Rwc * xn1;
        const Eigen::Vector3f ray2 = P2.Rwc * xn2;
        const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm()*ray2.norm());

        float cosParallaxStereo  = cosParallaxRays + 1.0f;
        float cosParallaxStereo1 = cosParallaxStereo;
        float cosParallaxStereo2 = cosParallaxStereo;

        if (bStereo1)
            cosParallaxStereo1 = cosf(2.0f*atan2f(mb1*0.5f, kf1->mvDepth[idx1]));
        else if (bStereo2)
            cosParallaxStereo2 = cosf(2.0f*atan2f(mb2[s]*0.5f, kf2->mvDepth[idx2]));

        cosParallaxStereo = fminf(cosParallaxStereo1, cosParallaxStereo2);

        Eigen::Vector3f x3D;
        bool goodProj = false;
        bool bPointStereo = false;

        if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0.0f &&
            (bStereo1 || bStereo2 ||
             (cosParallaxRays < 0.9996f && bInertial) ||
             (cosParallaxRays < 0.9998f && !bInertial)))
        {
            goodProj = triangulateLinear(xn1, xn2, P1.Tcw, P2.Tcw, x3D);
            if (!goodProj) continue;
        }
        else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
            bPointStereo = true;
            const float z = kf1->mvDepth[idx1];
            if (z > 0.0f) {
                const float u = kf1->mvKeys[idx1].ptx, v = kf1->mvKeys[idx1].pty;
                const Eigen::Vector3f xc((u - kf1->cx)*z/kf1->fx, (v - kf1->cy)*z/kf1->fy, z);
                x3D = P1.Rwc*xc + P1.Ow;
                goodProj = true;
            }
        }
        else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
            bPointStereo = true;
            const float z = kf2->mvDepth[idx2];
            if (z > 0.0f) {
                const float u = kf2->mvKeys[idx2].ptx, v = kf2->mvKeys[idx2].pty;
                const Eigen::Vector3f xc((u - kf2->cx)*z/kf2->fx, (v - kf2->cy)*z/kf2->fy, z);
                x3D = P2.Rwc*xc + P2.Ow;
                goodProj = true;
            }
        }
        else {
            continue;   // no stereo and very low parallax
        }

        if (!goodProj) continue;

        // --- cheirality ---
        const float z1 = P1.Tcw.row(2).head<3>().dot(x3D) + P1.Tcw(2,3);
        if (z1 <= 0.0f) continue;
        const float z2 = P2.Tcw.row(2).head<3>().dot(x3D) + P2.Tcw(2,3);
        if (z2 <= 0.0f) continue;

        // --- reprojection error, first keyframe ---
        const float sigmaSq1 = kf1->mvScaleFactors[kp1.octave]*kf1->mvScaleFactors[kp1.octave];
        const float x1 = P1.Tcw.row(0).head<3>().dot(x3D) + P1.Tcw(0,3);
        const float y1 = P1.Tcw.row(1).head<3>().dot(x3D) + P1.Tcw(1,3);
        const float invz1 = 1.0f/z1;

        if (!bStereo1) {
            const Eigen::Vector3f xc1(x1, y1, z1);
            const Eigen::Vector2f uv1 = bFisheye ? projectKB(xc1, C1) : projectPinhole(xc1, C1);
            const float ex = uv1(0) - kp1.ptx, ey = uv1(1) - kp1.pty;
            if ((ex*ex + ey*ey) > 5.991f*sigmaSq1) continue;
        } else {
            const float u1   = kf1->fx*x1*invz1 + kf1->cx;
            const float u1_r = u1 - kf1->mbf*invz1;
            const float v1   = kf1->fy*y1*invz1 + kf1->cy;
            const float ex = u1 - kp1.ptx, ey = v1 - kp1.pty, er = u1_r - kp1_ur;
            if ((ex*ex + ey*ey + er*er) > 7.8f*sigmaSq1) continue;
        }

        // --- reprojection error, second keyframe ---
        const float sigmaSq2 = kf2->mvScaleFactors[kp2.octave]*kf2->mvScaleFactors[kp2.octave];
        const float x2 = P2.Tcw.row(0).head<3>().dot(x3D) + P2.Tcw(0,3);
        const float y2 = P2.Tcw.row(1).head<3>().dot(x3D) + P2.Tcw(1,3);
        const float invz2 = 1.0f/z2;

        if (!bStereo2) {
            const Eigen::Vector3f xc2(x2, y2, z2);
            const Eigen::Vector2f uv2 = bFisheye ? projectKB(xc2, C2) : projectPinhole(xc2, C2);
            const float ex = uv2(0) - kp2.ptx, ey = uv2(1) - kp2.pty;
            if ((ex*ex + ey*ey) > 5.991f*sigmaSq2) continue;
        } else {
            const float u2   = kf2->fx*x2*invz2 + kf2->cx;
            // NOTE: the CPU uses the *current* keyframe's mbf here, not pKF2's.
            const float u2_r = u2 - kf1->mbf*invz2;
            const float v2   = kf2->fy*y2*invz2 + kf2->cy;
            const float ex = u2 - kp2.ptx, ey = v2 - kp2.pty, er = u2_r - kp2_ur;
            if ((ex*ex + ey*ey + er*er) > 7.8f*sigmaSq2) continue;
        }

        // --- scale consistency ---
        const float dist1 = (x3D - P1.Ow).norm();
        const float dist2 = (x3D - P2.Ow).norm();
        if (dist1 == 0.0f || dist2 == 0.0f) continue;
        if (bFarPoints && (dist1 >= thFarPoints || dist2 >= thFarPoints)) continue;

        const float ratioDist   = dist2/dist1;
        const float ratioOctave = kf1->mvScaleFactors[kp1.octave] / kf2->mvScaleFactors[kp2.octave];
        if (ratioDist*ratioFactor < ratioOctave || ratioDist > ratioOctave*ratioFactor) continue;

        out[t].x3D         = x3D;
        out[t].valid       = 1;
        out[t].stereoPoint = bPointStereo ? 1 : 0;
    }
}

__host__ CamPose makePose(const Sophus::SE3f &Tcw, const Eigen::Vector3f &Ow)
{
    CamPose p;
    p.Tcw = Tcw.matrix3x4();
    p.Rwc = p.Tcw.block<3,3>(0,0).transpose();
    p.Ow  = Ow;
    return p;
}

} // namespace

void MapPointCandidateKernel::initialize()
{
    if (memory_is_initialized) return;
    CudaKeyFrameStorage::initializeMemory();
    memory_is_initialized = true;
}

void MapPointCandidateKernel::launch(
    ORB_SLAM3::KeyFrame* pKF1,
    const std::vector<ORB_SLAM3::KeyFrame*> &kept,
    const std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices,
    bool bInertial, bool bFarPoints, float thFarPoints, float ratioFactor,
    std::vector<Candidate> &outCandidates)
{
    outCandidates.clear();
    const size_t nNeigh = kept.size();
    if (nNeigh == 0) return;

    // Flatten the matches, preserving (neighbour, idx1) order so the host can apply the
    // one-MapPoint-per-keypoint rule in the same order the CPU would.
    thrust::host_vector<CandIn> h_in;
    for (size_t s = 0; s < nNeigh && s < allvMatchedIndices.size(); s++)
        for (size_t q = 0; q < allvMatchedIndices[s].size(); q++) {
            CandIn c;
            c.neighbourSlot = (int)s;
            c.idx1 = (int)allvMatchedIndices[s][q].first;
            c.idx2 = (int)allvMatchedIndices[s][q].second;
            h_in.push_back(c);
        }
    const size_t nCand = h_in.size();
    if (nCand == 0) return;

    thrust::host_vector<CamPose> h_pose1(2), h_pose2(2*nNeigh);
    thrust::host_vector<float>   h_mb2(nNeigh);
    thrust::host_vector<MAPPING_DATA_WRAPPER::CudaKeyFrame*> h_kf2(nNeigh);

    h_pose1[0] = makePose(pKF1->GetPose(), pKF1->GetCameraCenter());
    h_pose1[1] = pKF1->mpCamera2 ? makePose(pKF1->GetRightPose(), pKF1->GetRightCameraCenter())
                                 : h_pose1[0];

    for (size_t s = 0; s < nNeigh; s++) {
        ORB_SLAM3::KeyFrame* pKF2 = kept[s];
        h_pose2[2*s]   = makePose(pKF2->GetPose(), pKF2->GetCameraCenter());
        h_pose2[2*s+1] = pKF2->mpCamera2 ? makePose(pKF2->GetRightPose(), pKF2->GetRightCameraCenter())
                                         : h_pose2[2*s];
        h_mb2[s] = pKF2->mb;

        MAPPING_DATA_WRAPPER::CudaKeyFrame* k = CudaKeyFrameStorage::getCudaKeyFrame(pKF2->mnId);
        if (k == nullptr) k = CudaKeyFrameStorage::addCudaKeyFrame(pKF2);
        h_kf2[s] = k->gpuAddr;
    }

    MAPPING_DATA_WRAPPER::CudaKeyFrame* kf1 = CudaKeyFrameStorage::getCudaKeyFrame(pKF1->mnId);
    if (kf1 == nullptr) kf1 = CudaKeyFrameStorage::addCudaKeyFrame(pKF1);

    thrust::device_vector<CandIn>   d_in    = h_in;
    thrust::device_vector<CamPose>  d_pose1 = h_pose1;
    thrust::device_vector<CamPose>  d_pose2 = h_pose2;
    thrust::device_vector<float>    d_mb2   = h_mb2;
    thrust::device_vector<MAPPING_DATA_WRAPPER::CudaKeyFrame*> d_kf2 = h_kf2;
    thrust::device_vector<CandOut>  d_out(nCand);

    const bool bFisheye = (pKF1->mpCamera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE);

    const int threads = 256;
    const int blocks  = (int)std::min<size_t>((nCand + threads - 1)/threads, 4096);

    mapPointCandidateKernel<<<blocks, threads>>>(
        kf1->gpuAddr, thrust::raw_pointer_cast(d_kf2.data()),
        thrust::raw_pointer_cast(d_in.data()), (int)nCand,
        thrust::raw_pointer_cast(d_pose1.data()),
        thrust::raw_pointer_cast(d_pose2.data()),
        thrust::raw_pointer_cast(d_mb2.data()), pKF1->mb,
        bInertial, bFarPoints, thFarPoints, ratioFactor,
        bFisheye, 1e-6f,
        thrust::raw_pointer_cast(d_out.data()));

    checkCudaError(cudaDeviceSynchronize(), "MapPointCandidateKernel: launch");

    thrust::host_vector<CandOut> h_out = d_out;

    outCandidates.reserve(nCand/4 + 1);
    for (size_t t = 0; t < nCand; t++) {
        if (!h_out[t].valid) continue;
        Candidate c;
        c.neighbourSlot = h_in[t].neighbourSlot;
        c.idx1 = h_in[t].idx1;
        c.idx2 = h_in[t].idx2;
        c.x = h_out[t].x3D(0); c.y = h_out[t].x3D(1); c.z = h_out[t].x3D(2);
        c.bStereoPoint = (h_out[t].stereoPoint != 0);
        outCandidates.push_back(c);
    }

    nCalls++;
    nEvaluated += nCand;
    nAccepted  += outCandidates.size();
}

void MapPointCandidateKernel::shutdown()
{
    if (!memory_is_initialized) return;
    std::cout << "[MapPointCandidateKernel:] calls=" << nCalls
              << " evaluated=" << nEvaluated
              << " accepted=" << nAccepted
              << " acceptFrac=" << (nEvaluated ? (double)nAccepted/nEvaluated : 0.0) << std::endl;
    memory_is_initialized = false;
}

void MapPointCandidateKernel::saveStats(const std::string &file_path) {}
