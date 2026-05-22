#include <iostream>
#include <fstream>

#include "Kernels/SearchForTriangulationKernel.h"
#include "Kernels/MappingKernelController.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "sophus/sim3.hpp"
#include <cusolverDn.h>
#include <Eigen/Dense>
#include <math.h>
#include <csignal>

// It is advisable to define the default dense index type as int for CUDA device code.
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

// Mark the function as __host__ __device__ so that it can be used on both sides if needed.
__host__ __device__ inline Eigen::Matrix3f deviceInverse(const Eigen::Matrix3f& M) {
    // Unpack matrix elements
    float a = M(0,0), b = M(0,1), c = M(0,2);
    float d = M(1,0), e = M(1,1), f = M(1,2);
    float g = M(2,0), h = M(2,1), i = M(2,2);
    
    // Compute determinant
    float det = a * (e * i - f * h)
              - b * (d * i - f * g)
              + c * (d * h - e * g);
              
    // Handle singular matrix (here we simply return Identity)
    if (fabsf(det) < 1e-6f) {
        return Eigen::Matrix3f::Identity();
    }
    
    float invDet = 1.0f / det;
    
    Eigen::Matrix3f inv;
    inv(0,0) =  (e * i - f * h) * invDet;
    inv(0,1) = -(b * i - c * h) * invDet;
    inv(0,2) =  (b * f - c * e) * invDet;
    inv(1,0) = -(d * i - f * g) * invDet;
    inv(1,1) =  (a * i - c * g) * invDet;
    inv(1,2) = -(a * f - c * d) * invDet;
    inv(2,0) =  (d * h - e * g) * invDet;
    inv(2,1) = -(a * h - b * g) * invDet;
    inv(2,2) =  (a * e - b * d) * invDet;
    
    return inv;
}

__device__ bool pinholeEpipolarConstrain(
    MAPPING_DATA_WRAPPER::CudaCamera camera1, MAPPING_DATA_WRAPPER::CudaCamera camera2, const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp1, const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp2, 
    const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, const float unc) {
    //Compute Fundamental Matrix
    Eigen::Matrix3f t12x = Sophus::SO3f::hat(t12);
    Eigen::Matrix3f K1 = camera1.toK;
    Eigen::Matrix3f K2 = camera2.toK;
    Eigen::Matrix3f F12 = deviceInverse(K1.transpose()) * t12x * R12 * deviceInverse(K2);
    
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.ptx*F12(0,0) + kp1.pty*F12(1,0) + F12(2,0);
    const float b = kp1.ptx*F12(0,1) + kp1.pty*F12(1,1) + F12(2,1);
    const float c = kp1.ptx*F12(0,2) + kp1.pty*F12(1,2) + F12(2,2);

    const float num = a*kp2.ptx+b*kp2.pty+c;

    const float den = a*a + b*b;

    if (den == 0)
        return false;

    const float dsqr = num*num / den;

    return dsqr < 3.84*unc;
}

// __device__ function that computes (approximately) the nullspace vector of A.
// A is a 4x4 matrix (by value, in row–major order as stored in Eigen) and x3D is returned by reference.
__device__ void solveSVD(Eigen::Matrix<float,4,4> A, Eigen::Vector3f &x3D) {
    // Compute symmetric matrix M = Aᵀ * A.
    Eigen::Matrix<float,4,4> M = A.transpose() * A;
    
    // Initialize V to the 4x4 identity (accumulates Jacobi rotations)
    Eigen::Matrix<float,4,4> V = Eigen::Matrix<float,4,4>::Identity();
    
    const int maxIter = 50;
    const float eps = 1e-6f;
    
    // Perform Jacobi rotations on M.
    for (int iter = 0; iter < maxIter; iter++) {
        // Find the largest off-diagonal element in M (upper triangle).
        int p = 0, q = 1;
        float maxOff = fabsf(M(0,1));
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                float off = fabsf(M(i, j));
                if (off > maxOff) {
                    maxOff = off;
                    p = i;
                    q = j;
                }
            }
        }
        if (maxOff < eps) break;  // Converged.
        
        float app = M(p, p);
        float aqq = M(q, q);
        float apq = M(p, q);
        
        // Compute rotation angle theta.
        float theta = 0.5f * atanf((2.0f * apq) / (aqq - app));
        float c = cosf(theta);
        float s = sinf(theta);
        
        // Rotate rows/columns p and q of M.
        for (int k = 0; k < 4; k++) {
            if (k == p || k == q) continue;
            float mkp = M(k, p);
            float mkq = M(k, q);
            float new_mkp = c * mkp - s * mkq;
            float new_mkq = s * mkp + c * mkq;
            M(k, p) = new_mkp;
            M(p, k) = new_mkp;  // enforce symmetry
            M(k, q) = new_mkq;
            M(q, k) = new_mkq;
        }
        // Update the diagonal entries.
        float new_app = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        float new_aqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        M(p, p) = new_app;
        M(q, q) = new_aqq;
        M(p, q) = 0.0f;
        M(q, p) = 0.0f;
        
        // Update the eigenvector matrix V.
        for (int i = 0; i < 4; i++) {
            float vip = V(i, p);
            float viq = V(i, q);
            V(i, p) = c * vip - s * viq;
            V(i, q) = s * vip + c * viq;
        }
    }
    
    // Find the index of the smallest eigenvalue (the smallest diagonal element in M).
    int minIndex = 0;
    float minEigen = M(0, 0);
    for (int i = 1; i < 4; i++) {
        if (M(i, i) < minEigen) {
            minEigen = M(i, i);
            minIndex = i;
        }
    }
    
    // The corresponding eigenvector (4D) is the minIndex-th column of V.
    Eigen::Vector4f v = V.col(minIndex);
    
    // Convert from homogeneous (4D) to Euclidean (3D) coordinates:
    if (fabsf(v(3)) < eps) {
        x3D = v.head<3>(); // Avoid division if nearly zero.
    } else {
        x3D = v.head<3>() / v(3);
    }
}

__device__ void triangulate(float &p1x, float &p1y, float &p2x, float &p2y, const Eigen::Matrix<float,3,4> &Tcw1,
                            const Eigen::Matrix<float,3,4> &Tcw2, Eigen::Vector3f &x3D) {
    Eigen::Matrix<float,4,4> A;
    A.row(0) = p1x*Tcw1.row(2)-Tcw1.row(0);
    A.row(1) = p1y*Tcw1.row(2)-Tcw1.row(1);
    A.row(2) = p2x*Tcw2.row(2)-Tcw2.row(0);
    A.row(3) = p2y*Tcw2.row(2)-Tcw2.row(1);

    solveSVD(A, x3D);

    // Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    // Eigen::Vector4f x3Dh = svd.matrixV().col(3);
    // x3D = x3Dh.head(3)/x3Dh(3);
}

__device__ Eigen::Vector2f project(const Eigen::Vector3f &v3D, float* mvParameters) {
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

__device__ Eigen::Vector3f unprojectEig(float p2Dx, float p2Dy, float* mvParameters, const float camPrecision) {

    //Use Newthon method to solve for theta with good precision (err ~ e-6)
    float pwx = (p2Dx - mvParameters[2]) / mvParameters[0];
    float pwy = (p2Dy - mvParameters[3]) / mvParameters[1];
    float scale = 1.f;
    float theta_d = sqrtf(pwx*pwx + pwy*pwy);
    theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);

    if (theta_d > 1e-8) {
        //Compensate distortion iteratively
        float theta = theta_d;

        for (int j = 0; j < 10; j++) {
            float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 =
                    theta4 * theta4;
            float k0_theta2 = mvParameters[4] * theta2, k1_theta4 = mvParameters[5] * theta4;
            float k2_theta6 = mvParameters[6] * theta6, k3_theta8 = mvParameters[7] * theta8;
            float theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
            theta = theta - theta_fix;
            if (fabsf(theta_fix) < camPrecision)
                break;
        }
        //scale = theta - theta_d;
        scale = std::tan(theta) / theta_d;
    }

    return Eigen::Vector3f(pwx*scale, pwy*scale, 1.f);
}

__device__ float triangulateMatches(const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp1, const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp2, float *mvParameters1, float *mvParameters2, 
                                    const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, const float unc, const float camPrecision, 
                                    Eigen::Vector3f& p3D) {

    Eigen::Vector3f r1 = unprojectEig(kp1.ptx, kp1.pty, mvParameters1, camPrecision);
    Eigen::Vector3f r2 = unprojectEig(kp2.ptx, kp2.pty, mvParameters2, camPrecision);

    //Check parallax
    Eigen::Vector3f r21 = R12 * r2;

    const float cosParallaxRays = r1.dot(r21) / (r1.norm()*r21.norm());

    if(cosParallaxRays > 0.9998){
        return -1;
    }

    //Parallax is good, so we try to triangulate
    float p11x, p11y, p22x, p22y;

    p11x = r1[0];
    p11y = r1[1];

    p22x = r2[0];
    p22y = r2[1];

    Eigen::Vector3f x3D;
    Eigen::Matrix<float,3,4> Tcw1;
    Tcw1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

    Eigen::Matrix<float,3,4> Tcw2;

    Eigen::Matrix3f R21 = R12.transpose();
    Tcw2 << R21, -R21 * t12;


    triangulate(p11x, p11y, p22x, p22y, Tcw1, Tcw2, x3D);
    // cv::Mat x3Dt = x3D.t();

    float z1 = x3D(2);
    if (z1 <= 0) {
        return -2;
    }

    float z2 = R21.row(2).dot(x3D)+Tcw2(2,3);
    if (z2<=0) {
        return -3;
    }

    //Check reprojection error
    Eigen::Vector2f uv1 = project(x3D, mvParameters1);

    float errX1 = uv1(0) - kp1.ptx;
    float errY1 = uv1(1) - kp1.pty;

    if ((errX1*errX1+errY1*errY1) > 5.991*sigmaLevel){   //Reprojection error is high
        return -4;
    }

    Eigen::Vector3f x3D2 = R21 * x3D + Tcw2.col(3);
    Eigen::Vector2f uv2 = project(x3D2, mvParameters2);

    float errX2 = uv2(0) - kp2.ptx;
    float errY2 = uv2(1) - kp2.pty;

    if ((errX2*errX2+errY2*errY2) > 5.991*unc){   //Reprojection error is high
        return -5;
    }

    p3D = x3D;

    return z1;
}

__device__ bool fisheyeEpipolarConstrain(MAPPING_DATA_WRAPPER::CudaCamera camera1, MAPPING_DATA_WRAPPER::CudaCamera camera2, 
                                         const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp1, const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp2,
                                         const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, 
                                         const float unc, const float camPrecision) {
    Eigen::Vector3f p3D;
    return triangulateMatches(kp1, kp2, camera1.mvParameters, camera2.mvParameters, R12, t12, sigmaLevel, unc, camPrecision, p3D) > 0.0001f;
}

__global__ void searchForTriangulationKernel(
    MAPPING_DATA_WRAPPER::CudaKeyFrame* currKeyframe, MAPPING_DATA_WRAPPER::CudaKeyFrame** neighKeyframes,
    size_t featVecSize, size_t keyFrameMapPointCount, size_t outVecSize, const float camPrecision, const bool bCoarse,
    size_t *currFrameFeatVecIdxCorrespondences, size_t *neighFramesFeatVecIdxCorrespondences,
    Eigen::Matrix3f *Rll, Eigen::Matrix3f *Rlr, Eigen::Matrix3f *Rrl, Eigen::Matrix3f *Rrr, Eigen::Matrix3f *R12s,
    Eigen::Vector3f *tll, Eigen::Vector3f *tlr, Eigen::Vector3f *trl, Eigen::Vector3f *trr, Eigen::Vector3f *t12s, Eigen::Vector2f *ep,
    bool *currFrameMapPointExists, bool *neighFramesMapPointExists, int *matchedPairIndexes
    ) {

    int neighborIdx = blockIdx.x;
    int correspondingFeatVecIdx = threadIdx.x;
    MAPPING_DATA_WRAPPER::CudaKeyFrame* neighKeyframe = neighKeyframes[neighborIdx];
    
    Eigen::Matrix3f R12;
    Eigen::Vector3f t12;

    int currFrameFeatVecCorrespondenceIdx = currFrameFeatVecIdxCorrespondences[neighborIdx*featVecSize + correspondingFeatVecIdx];
    int neighFrameFeatVecCorrespondenceIdx = neighFramesFeatVecIdxCorrespondences[neighborIdx*featVecSize + correspondingFeatVecIdx];

    if (currFrameFeatVecCorrespondenceIdx == -1)
        return;

    if (currFrameFeatVecCorrespondenceIdx > currKeyframe->mFeatCount || neighFrameFeatVecCorrespondenceIdx > neighKeyframe->mFeatCount)
        return;

    int currFeatStartIdx = (currFrameFeatVecCorrespondenceIdx == 0) ? 0 : currKeyframe->mFeatVecStartIndexes[currFrameFeatVecCorrespondenceIdx-1];
    int currFeatVecCount = (currFrameFeatVecCorrespondenceIdx == 0) ? currKeyframe->mFeatVecStartIndexes[0] : 
        currKeyframe->mFeatVecStartIndexes[currFrameFeatVecCorrespondenceIdx] - currKeyframe->mFeatVecStartIndexes[currFrameFeatVecCorrespondenceIdx-1];
        
    int neighFeatStartIdx = (neighFrameFeatVecCorrespondenceIdx == 0) ? 0 : neighKeyframe->mFeatVecStartIndexes[neighFrameFeatVecCorrespondenceIdx-1];
    int neighFeatVecCount = (neighFrameFeatVecCorrespondenceIdx == 0) ? neighKeyframe->mFeatVecStartIndexes[0] : 
        neighKeyframe->mFeatVecStartIndexes[neighFrameFeatVecCorrespondenceIdx] - neighKeyframe->mFeatVecStartIndexes[neighFrameFeatVecCorrespondenceIdx-1];

    MAPPING_DATA_WRAPPER::CudaCamera camera1, camera2;
    camera1.toK = currKeyframe->camera1.toK;
    memcpy(camera1.mvParameters, currKeyframe->camera1.mvParameters, sizeof(float)*8);
    camera2.toK = neighKeyframe->camera1.toK;
    memcpy(camera2.mvParameters, neighKeyframe->camera1.mvParameters, sizeof(float)*8);

    if (!currKeyframe->camera2.isAvailable && !neighKeyframe->camera2.isAvailable) {
        R12 = R12s[neighborIdx];
        t12 = t12s[neighborIdx];
    }
    
    for (size_t i1 = 0; i1 < currFeatVecCount; i1++) {

        const int idx1 = (int) currKeyframe->mFeatVec[currFeatStartIdx + i1];
        
        if (currFrameMapPointExists[idx1])
            continue;

        const bool bStereo1 = (!currKeyframe->camera2.isAvailable && currKeyframe->mvuRight[idx1] >= 0);
        
        const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp1 = 
            (currKeyframe->Nleft == -1) ? currKeyframe->mvKeysUn[idx1]
                                        : (idx1 < currKeyframe->Nleft) ? currKeyframe->mvKeys[idx1]
                                                                       : currKeyframe->mvKeysRight[idx1-currKeyframe->Nleft];
                                                                       
        const bool bRight1 = (currKeyframe->Nleft == -1 || idx1 < currKeyframe->Nleft) ? false : true;

        const uint8_t *d1 = &currKeyframe->mDescriptors[idx1*DESCRIPTOR_SIZE];
        
        int bestDist = MATCH_TH_LOW;
        int bestIdx2 = -1;

        for (size_t i2 = 0; i2 < neighFeatVecCount; i2++) {

            const int idx2 = (int) neighKeyframe->mFeatVec[neighFeatStartIdx + i2];
       
            if (neighFramesMapPointExists[neighborIdx*keyFrameMapPointCount + idx2])
                continue;

            const bool bStereo2 = (!neighKeyframe->camera2.isAvailable && neighKeyframe->mvuRight[idx2] >= 0);
            const uint8_t *d2 = &neighKeyframe->mDescriptors[idx2*DESCRIPTOR_SIZE];
            
            const int dist = DescriptorDistance(d1, d2);

            if (dist > MATCH_TH_LOW || dist > bestDist)
                continue;

            const MAPPING_DATA_WRAPPER::CudaKeyPoint &kp2 = 
                (neighKeyframe->Nleft == -1) ? neighKeyframe->mvKeysUn[idx2]
                                             : (idx2 < neighKeyframe->Nleft) ? neighKeyframe->mvKeys[idx2]
                                                                             : neighKeyframe->mvKeysRight[idx2-neighKeyframe->Nleft];

            const bool bRight2 = (neighKeyframe->Nleft == -1 || idx2 < neighKeyframe->Nleft) ? false : true;

            if (!bStereo1 && !bStereo2 && !currKeyframe->camera2.isAvailable) {
                const float distex = ep[neighborIdx](0) - kp2.ptx;
                const float distey = ep[neighborIdx](1) - kp2.pty;
                if (distex*distex + distey*distey < 100 * neighKeyframe->mvScaleFactors[kp2.octave])
                    continue;
            }

            if (currKeyframe->camera2.isAvailable && neighKeyframe->camera2.isAvailable) {
                if (bRight1 && bRight2) {
                    R12 = Rrr[neighborIdx];
                    t12 = trr[neighborIdx];

                    camera1.toK = currKeyframe->camera2.toK;
                    memcpy(camera1.mvParameters, currKeyframe->camera2.mvParameters, sizeof(float)*8);
                    camera2.toK = neighKeyframe->camera2.toK;
                    memcpy(camera2.mvParameters, neighKeyframe->camera2.mvParameters, sizeof(float)*8);
                }
                else if (bRight1 && !bRight2) {
                    R12 = Rrl[neighborIdx];
                    t12 = trl[neighborIdx];

                    camera1.toK = currKeyframe->camera2.toK;
                    memcpy(camera1.mvParameters, currKeyframe->camera2.mvParameters, sizeof(float)*8);
                    camera2.toK = neighKeyframe->camera1.toK;
                    memcpy(camera2.mvParameters, neighKeyframe->camera1.mvParameters, sizeof(float)*8);
                }
                else if (!bRight1 && bRight2) {
                    R12 = Rlr[neighborIdx];
                    t12 = tlr[neighborIdx];

                    camera1.toK = currKeyframe->camera1.toK;
                    memcpy(camera1.mvParameters, currKeyframe->camera1.mvParameters, sizeof(float)*8);
                    camera2.toK = neighKeyframe->camera2.toK;
                    memcpy(camera2.mvParameters, neighKeyframe->camera2.mvParameters, sizeof(float)*8);
                }
                else {
                    R12 = Rll[neighborIdx];
                    t12 = tll[neighborIdx];

                    camera1.toK = currKeyframe->camera1.toK;
                    memcpy(camera1.mvParameters, currKeyframe->camera1.mvParameters, sizeof(float)*8);
                    camera2.toK = neighKeyframe->camera1.toK;
                    memcpy(camera2.mvParameters, neighKeyframe->camera1.mvParameters, sizeof(float)*8);
                }
            }

            bool epipolarConstrainIsMet;
            float kp1mvLevelSigma2 = currKeyframe->mvScaleFactors[kp1.octave] * currKeyframe->mvScaleFactors[kp1.octave];
            float kp2mvLevelSigma2 = neighKeyframe->mvScaleFactors[kp2.octave] * neighKeyframe->mvScaleFactors[kp2.octave];

            if (currKeyframe->Nleft == -1)
                epipolarConstrainIsMet = pinholeEpipolarConstrain(camera1, camera2, kp1, kp2, R12, t12, kp1mvLevelSigma2, kp2mvLevelSigma2);
            else
                epipolarConstrainIsMet = fisheyeEpipolarConstrain(camera1, camera2, kp1, kp2, R12, t12, kp1mvLevelSigma2, kp2mvLevelSigma2, camPrecision);
            
            if (bCoarse || epipolarConstrainIsMet) {
                bestIdx2 = idx2;
                bestDist = dist;
            }
        }

        if (bestIdx2 >= 0)
            matchedPairIndexes[neighborIdx*outVecSize + idx1] = bestIdx2;
    }
}

void SearchForTriangulationKernel::launch(ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
                                          bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2, 
                                          std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, std::vector<size_t> &vpNeighKFsIndexes) {

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    bool bCoarse = mbInertial && recentlyLost && mbIMU_BA2;
    size_t featVecSize = MAX_FEAT_VEC_SIZE;
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    if (!memory_is_initialized)
        initialize();
    
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point startCopyObjectCreation = std::chrono::steady_clock::now();
#endif

    for (size_t i = 0; i < vpNeighKFs.size(); i++) {
        
        ORB_SLAM3::KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if (!mbMonocular) {
            if (baseline < pKF2->mb) 
                continue;
        }
        else {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
                continue;
        }
        vpNeighKFsIndexes.push_back(i);
    }

    size_t nn = vpNeighKFsIndexes.size();

    if (nn == 0)
        return;

    MAPPING_DATA_WRAPPER::CudaKeyFrame* currKeyframeOnGPU = CudaKeyFrameStorage::getCudaKeyFrame(mpCurrentKeyFrame->mnId);
    if (currKeyframeOnGPU == nullptr) {
        cerr << "[ERROR] SearchForTriangulationKernel::launch: ] CudaKeyFrameStorage doesn't have the keyframe: " << mpCurrentKeyFrame->mnId << "\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    MAPPING_DATA_WRAPPER::CudaKeyFrame* neighKeyframesOnGPU[nn];
    for (size_t i = 0; i < nn; i++) {
        neighKeyframesOnGPU[i] = CudaKeyFrameStorage::getCudaKeyFrame(vpNeighKFs[vpNeighKFsIndexes[i]]->mnId);
        if (neighKeyframesOnGPU[i] == nullptr) {
            cerr << "[ERROR] SearchForTriangulationKernel::launch: ] CudaKeyFrameStorage doesn't have the keyframe: " << vpNeighKFs[vpNeighKFsIndexes[i]]->mnId << "\n";
            MappingKernelController::shutdownKernels(true, true);
            exit(EXIT_FAILURE);
        }
    }
        
    size_t currFrameFeatVecIdxCorrespondences[nn*featVecSize];
    size_t neighFramesFeatVecIdxCorrespondences[nn*featVecSize];
    std::fill_n(currFrameFeatVecIdxCorrespondences, nn*featVecSize, -1);

    Eigen::Matrix3f Rll[nn], Rlr[nn], Rrl[nn], Rrr[nn], R12s[nn];
    Eigen::Vector3f tll[nn], tlr[nn], trl[nn], trr[nn], t12s[nn];
    Eigen::Vector2f eps[nn];

    for (size_t i = 0; i < nn; i++) {

        ORB_SLAM3::KeyFrame* pKF2 = vpNeighKFs[vpNeighKFsIndexes[i]];

        const DBoW2::FeatureVector &vFeatVec1 = mpCurrentKeyFrame->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        Sophus::SE3f T1w = mpCurrentKeyFrame->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
        Eigen::Vector3f Cw = mpCurrentKeyFrame->GetCameraCenter();
        Eigen::Vector3f C2 = T2w * Cw;

        Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
        Sophus::SE3f T12;
        Sophus::SE3f Tll, Tlr, Trl, Trr;
        Eigen::Matrix3f R12; // for fastest computation
        Eigen::Vector3f t12; // for fastest computation

        eps[i] = ep;

        ORB_SLAM3::GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        if (!mpCurrentKeyFrame->mpCamera2 && !pKF2->mpCamera2){
            T12 = T1w * Tw2;
            R12 = T12.rotationMatrix();
            t12 = T12.translation();
            
            R12s[i] = R12;
            t12s[i] = t12;
        }
        else {
            Sophus::SE3f Tr1w = mpCurrentKeyFrame->GetRightPose();
            Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            Tll = T1w * Tw2;
            Tlr = T1w * Twr2;
            Trl = Tr1w * Tw2;
            Trr = Tr1w * Twr2;
        }

        Rll[i] = Tll.rotationMatrix();
        Rlr[i] = Tlr.rotationMatrix();
        Rrl[i] = Trl.rotationMatrix();
        Rrr[i] = Trr.rotationMatrix();
        tll[i] = Tll.translation();
        tlr[i] = Tlr.translation();
        trl[i] = Trl.translation();
        trr[i] = Trr.translation();

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
        size_t f1Idx = 0, f2Idx = 0;
        int counter = 0;

        while (f1it != f1end && f2it != f2end) {
            if (f1it->first == f2it->first) {
                currFrameFeatVecIdxCorrespondences[i*featVecSize + counter] = f1Idx;
                neighFramesFeatVecIdxCorrespondences[i*featVecSize + counter] = f2Idx;

                f1it++;
                f2it++;
                f1Idx++;
                f2Idx++;
                counter++;
            }
            else if (f1it->first < f2it->first) {
                f1it = vFeatVec1.lower_bound(f2it->first);
                f1Idx = std::distance(vFeatVec1.begin(), f1it);
            }
            else {
                f2it = vFeatVec2.lower_bound(f1it->first);
                f2Idx = std::distance(vFeatVec2.begin(), f2it);
            }
        }

    }

    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize, outVecSize;
    
    if (CudaUtils::cameraIsFisheye) {
        mapPointVecSize = maxFeatures*2;
        outVecSize = maxFeatures*2;
    }
    else {
        mapPointVecSize = maxFeatures;
        outVecSize = maxFeatures;
    }

    bool currFrameMapPointExists[mapPointVecSize], neighFramesMapPointExists[mapPointVecSize*nn];
    mpCurrentKeyFrame->GetMapPointAvailabality(currFrameMapPointExists);
    for (int i = 0; i < nn; i++)
        vpNeighKFs[vpNeighKFsIndexes[i]]->GetMapPointAvailabality(neighFramesMapPointExists + i*mapPointVecSize);

    float camPrecision;
    if (CudaUtils::cameraIsFisheye) {
        ORB_SLAM3::KannalaBrandt8* pKBCam = (ORB_SLAM3::KannalaBrandt8*) mpCurrentKeyFrame->mpCamera;
        camPrecision = pKBCam->GetPrecision();
    }

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endCopyObjectCreation = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpy = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(d_neighKeyframes, neighKeyframesOnGPU, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*)*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighKeyframesOnGPU from host to device");

    checkCudaError(cudaMemcpy(d_currFrameFeatVecIdxCorrespondences, currFrameFeatVecIdxCorrespondences, sizeof(size_t)*nn*featVecSize, cudaMemcpyHostToDevice), "Failed to copy vector currFrameFeatVecIdxCorrespondences from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesFeatVecIdxCorrespondences, neighFramesFeatVecIdxCorrespondences, sizeof(size_t)*nn*featVecSize, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesFeatVecIdxCorrespondences from host to device");

    checkCudaError(cudaMemcpy(d_Rll, Rll, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rll from host to device");
    checkCudaError(cudaMemcpy(d_Rlr, Rlr, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rlr from host to device");
    checkCudaError(cudaMemcpy(d_Rrl, Rrl, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rrl from host to device");
    checkCudaError(cudaMemcpy(d_Rrr, Rrr, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rrr from host to device");
    checkCudaError(cudaMemcpy(d_tll, tll, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector tll from host to device");
    checkCudaError(cudaMemcpy(d_tlr, tlr, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector tlr from host to device");
    checkCudaError(cudaMemcpy(d_trl, trl, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector trl from host to device");
    checkCudaError(cudaMemcpy(d_trr, trr, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector trr from host to device");
    checkCudaError(cudaMemcpy(d_R12, R12s, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector R12s from host to device");
    checkCudaError(cudaMemcpy(d_t12, t12s, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector t12s from host to device");
    checkCudaError(cudaMemcpy(d_ep, eps, sizeof(Eigen::Vector2f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector eps from host to device");

    checkCudaError(cudaMemcpy(d_currFrameMapPointExists, currFrameMapPointExists, sizeof(currFrameMapPointExists), cudaMemcpyHostToDevice), "Failed to copy vector currFrameMapPointExists from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesMapPointExists, neighFramesMapPointExists, sizeof(neighFramesMapPointExists), cudaMemcpyHostToDevice), "Failed to copy vector neighFramesMapPointExists from host to device");

    checkCudaError(cudaMemset(d_matchedPairIndexes, 0xFF, nn*outVecSize*sizeof(int)), "Failed to set the memory of d_matchedPairIndexes to -1");

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endMemcpy = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startsearchForTriangulationKernel = std::chrono::steady_clock::now();
#endif

    dim3 gridDim(nn);
    dim3 blockDim(featVecSize);
    searchForTriangulationKernel<<<gridDim, blockDim>>>(
        currKeyframeOnGPU, d_neighKeyframes, featVecSize, mapPointVecSize, outVecSize, camPrecision, bCoarse,
        d_currFrameFeatVecIdxCorrespondences, d_neighFramesFeatVecIdxCorrespondences,
        d_Rll, d_Rlr, d_Rrl, d_Rrr, d_R12, d_tll, d_tlr, d_trl, d_trr, d_t12, d_ep,
        d_currFrameMapPointExists, d_neighFramesMapPointExists,
        d_matchedPairIndexes
    );
    
    checkCudaError(cudaGetLastError(), "Failed to launch searchForTriangulationKernel kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching the kernel");

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endsearchForTriangulationKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpyToCPU = std::chrono::steady_clock::now();
#endif
    
    int h_matchedPairIndexes[outVecSize*nn];
    checkCudaError(cudaMemcpy(h_matchedPairIndexes, d_matchedPairIndexes, sizeof(int)*outVecSize*nn, cudaMemcpyDeviceToHost), "Failed to copy vector d_matchedPairIndexes from device to host");

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endMemcpyToCPU = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startPostProcess = std::chrono::steady_clock::now();
#endif
    
    convertToVectorOfPairs(h_matchedPairIndexes, nn, outVecSize, allvMatchedIndices);

    // For testing
    // cout << "===========================================================\n";
    // printMatchedIndices(allvMatchedIndices);
    // cout << "***********************************************************\n";
    // origCreateNewMapPoints(mpCurrentKeyFrame, vpNeighKFs, mbMonocular, mbInertial, recentlyLost, mbIMU_BA2);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endPostProcess = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();

    double copyObjectCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endCopyObjectCreation - startCopyObjectCreation).count();
    double memcpyToGPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpy - startMemcpy).count();
    double searchForTriangulationKernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endsearchForTriangulationKernel - startsearchForTriangulationKernel).count();
    double memcpyToCPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpyToCPU - startMemcpyToCPU).count();
    double postProcess = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endPostProcess - startPostProcess).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    data_wrap_time.emplace_back(frameCounter, copyObjectCreation + postProcess);
    input_data_wrap_time.emplace_back(frameCounter, copyObjectCreation);
    input_data_transfer_time.emplace_back(frameCounter, memcpyToGPU);
    kernel_exec_time.emplace_back(frameCounter, searchForTriangulationKernel);
    output_data_transfer_time.emplace_back(frameCounter, memcpyToCPU);
    data_transfer_time.emplace_back(frameCounter, memcpyToGPU + memcpyToCPU);
    output_data_wrap_time.emplace_back(frameCounter, postProcess);
    total_exec_time.emplace_back(frameCounter, total);

    frameCounter++;
#endif
}

// For testing
void SearchForTriangulationKernel::origCreateNewMapPoints(ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
                                                          bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2) {

    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();
    std::vector<std::vector<std::pair<size_t,size_t>>> allvMatchedIndices;
    std::vector<size_t> vpNeighKFsIndexes;
    
    for(size_t i=0; i<vpNeighKFs.size(); i++) {

        ORB_SLAM3::KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Search matches that fullfil epipolar constraint
        std::vector<std::pair<size_t,size_t>> vMatchedIndices;
        bool bCoarse = mbInertial && recentlyLost && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();
        origSearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);
        allvMatchedIndices.push_back(vMatchedIndices);
        vpNeighKFsIndexes.push_back(i);
    }
    printMatchedIndices(allvMatchedIndices);
}

void SearchForTriangulationKernel::origSearchForTriangulation(ORB_SLAM3::KeyFrame* pKF1, ORB_SLAM3::KeyFrame* pKF2, 
                                                              std::vector<std::pair<size_t,size_t>> &vMatchedPairs, 
                                                              const bool bOnlyStereo, const bool bCoarse) {
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();
    Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
    Eigen::Vector3f Cw = pKF1->GetCameraCenter();
    Eigen::Vector3f C2 = T2w * Cw;

    Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
    Sophus::SE3f T12;
    Sophus::SE3f Tll, Tlr, Trl, Trr;
    Eigen::Matrix3f R12; // for fastest computation
    Eigen::Vector3f t12; // for fastest computation

    ORB_SLAM3::GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

    if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
        T12 = T1w * Tw2;
        R12 = T12.rotationMatrix();
        t12 = T12.translation();
    }
    else{
        Sophus::SE3f Tr1w = pKF1->GetRightPose();
        Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
        Tll = T1w * Tw2;
        Tlr = T1w * Twr2;
        Trl = Tr1w * Tw2;
        Trr = Tr1w * Twr2;
    }

    Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr  = Tlr.rotationMatrix(), Rrl  = Trl.rotationMatrix(), Rrr  = Trr.rotationMatrix();
    Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node
    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    int counter = -1;

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            counter++;
            
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                ORB_SLAM3::MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if(pMP1)
                {
                    continue;
                }

                const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;

                const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                            : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                    : true;

                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = MATCH_TH_LOW;
                int bestIdx2 = -1;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    ORB_SLAM3::MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const int dist = origDescriptorDistance(d1,d2);

                    if(dist>MATCH_TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                    const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                        : true;

                    if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                    {
                        const float distex = ep(0)-kp2.pt.x;
                        const float distey = ep(1)-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                        {
                            continue;
                        }
                    }

                    if(pKF1->mpCamera2 && pKF2->mpCamera2){
                        if(bRight1 && bRight2){
                            R12 = Rrr;
                            t12 = trr;
                            T12 = Trr;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else if(bRight1 && !bRight2){
                            R12 = Rrl;
                            t12 = trl;
                            T12 = Trl;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera;
                        }
                        else if(!bRight1 && bRight2){
                            R12 = Rlr;
                            t12 = tlr;
                            T12 = Tlr;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else{
                            R12 = Rll;
                            t12 = tll;
                            T12 = Tll;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera;
                        }

                    }

                    if(bCoarse || pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])) // MODIFICATION_2
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                    : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }
}

int SearchForTriangulationKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
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

void SearchForTriangulationKernel::convertToVectorOfPairs(int* gpuMatchedIdxs, int nn, int maxFeatures, 
                                                          std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices) {

    for (size_t i = 0; i < nn; i++) {
        std::vector<std::pair<size_t,size_t>> tmp;
        for (size_t j = 0; j < maxFeatures; j++) {
            if (gpuMatchedIdxs[i*maxFeatures + j] != -1)
                tmp.push_back(std::make_pair(j, (size_t) gpuMatchedIdxs[i*maxFeatures + j]));
        }
        allvMatchedIndices.push_back(tmp);
    }
}

void SearchForTriangulationKernel::printMatchedIndices(const std::vector<std::vector<std::pair<size_t, size_t>>> &allvMatchedIndices) {
    for (size_t i = 0; i < allvMatchedIndices.size(); ++i) {
        std::cout << "Row " << i << ": ";
        for (const auto &pair : allvMatchedIndices[i]) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;
    }
}

void SearchForTriangulationKernel::initialize() {
    
    if (memory_is_initialized)
        return;

    size_t featVecSize = MAX_FEAT_VEC_SIZE;
    size_t maxNeighborCount = MAX_NEIGHBOUR_COUNT;
    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize, outVecSize;
    if (CudaUtils::cameraIsFisheye) {
        mapPointVecSize = maxFeatures*2;
        outVecSize = maxFeatures*2;
    }
    else {
        mapPointVecSize = maxFeatures;
        outVecSize = maxFeatures;
    }

    checkCudaError(cudaMalloc((void**)&d_neighKeyframes, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*)*maxNeighborCount), "Failed to allocate device vector d_neighKeyframes");

    checkCudaError(cudaMalloc((void**)&d_currFrameFeatVecIdxCorrespondences, sizeof(size_t)*featVecSize*maxNeighborCount), "Failed to allocate device vector d_currFrameFeatVecIdxCorrespondences");
    checkCudaError(cudaMalloc((void**)&d_neighFramesFeatVecIdxCorrespondences, sizeof(size_t)*featVecSize*maxNeighborCount), "Failed to allocate device vector d_neighFramesFeatVecIdxCorrespondences");
    
    checkCudaError(cudaMalloc((void**)&d_Rll, sizeof(Eigen::Matrix3f)*maxNeighborCount), "Failed to allocate device vector d_Rll");
    checkCudaError(cudaMalloc((void**)&d_Rlr, sizeof(Eigen::Matrix3f)*maxNeighborCount), "Failed to allocate device vector d_Rlr");
    checkCudaError(cudaMalloc((void**)&d_Rrl, sizeof(Eigen::Matrix3f)*maxNeighborCount), "Failed to allocate device vector d_Rrl");
    checkCudaError(cudaMalloc((void**)&d_Rrr, sizeof(Eigen::Matrix3f)*maxNeighborCount), "Failed to allocate device vector d_Rrr");
    checkCudaError(cudaMalloc((void**)&d_tll, sizeof(Eigen::Vector3f)*maxNeighborCount), "Failed to allocate device vector d_tll");
    checkCudaError(cudaMalloc((void**)&d_tlr, sizeof(Eigen::Vector3f)*maxNeighborCount), "Failed to allocate device vector d_tlr");
    checkCudaError(cudaMalloc((void**)&d_trl, sizeof(Eigen::Vector3f)*maxNeighborCount), "Failed to allocate device vector d_trl");
    checkCudaError(cudaMalloc((void**)&d_trr, sizeof(Eigen::Vector3f)*maxNeighborCount), "Failed to allocate device vector d_trr");    
    checkCudaError(cudaMalloc((void**)&d_R12, sizeof(Eigen::Matrix3f)*maxNeighborCount), "Failed to allocate device vector d_R12");
    checkCudaError(cudaMalloc((void**)&d_t12, sizeof(Eigen::Vector3f)*maxNeighborCount), "Failed to allocate device vector d_t12");
    checkCudaError(cudaMalloc((void**)&d_ep, sizeof(Eigen::Vector2f)*maxNeighborCount), "Failed to allocate device vector d_ep");
    checkCudaError(cudaMalloc((void**)&d_currFrameMapPointExists, sizeof(bool)*mapPointVecSize), "Failed to allocate device vector d_currFrameMapPointExists");
    checkCudaError(cudaMalloc((void**)&d_neighFramesMapPointExists, sizeof(bool)*mapPointVecSize*maxNeighborCount), "Failed to allocate device vector d_neighFramesMapPointExists");
    
    checkCudaError(cudaMalloc((void**)&d_matchedPairIndexes, sizeof(int)*outVecSize*maxNeighborCount), "Failed to allocate device vector d_matchedPairIndexes");

    memory_is_initialized = true;
}

void SearchForTriangulationKernel::shutdown() {
    if (memory_is_initialized) {
        cudaFree(d_neighKeyframes);
        cudaFree(d_currFrameFeatVecIdxCorrespondences);
        cudaFree(d_neighFramesFeatVecIdxCorrespondences);
        
        cudaFree(d_Rll);
        cudaFree(d_Rlr);
        cudaFree(d_Rrl);
        cudaFree(d_Rrr);
        cudaFree(d_tll);
        cudaFree(d_tlr);
        cudaFree(d_trl);
        cudaFree(d_trr);
        cudaFree(d_R12);
        cudaFree(d_t12);
        cudaFree(d_ep);
        cudaFree(d_currFrameMapPointExists);
        cudaFree(d_neighFramesMapPointExists);

        cudaFree(d_matchedPairIndexes);
    }

    memory_is_initialized = false;
}

void SearchForTriangulationKernel::saveStats(const std::string &file_path) {

    std::string data_path = file_path + "/SearchForTriangulationKernel/";
    std::cout << "[SearchForTriangulationKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[SearchForTriangulationKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }
    std::ofstream myfile;
    
    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/kernel_exec_time.txt");
    for (const auto& p : kernel_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/data_transfer_time.txt");
    for (const auto& p : data_transfer_time) {
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

    myfile.open(data_path + "/data_wrap_time.txt");
    for (const auto& p : data_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_wrap_time.txt");
    for (const auto& p : input_data_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/output_data_wrap_time.txt");
    for (const auto& p : output_data_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
}