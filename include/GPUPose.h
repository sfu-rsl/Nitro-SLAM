#pragma once
#include <Eigen/Dense>
#include <array>
#include "KeyFrame.h"

namespace gpu {

    template <typename T>
    using Vec2 = Eigen::Matrix<T, 2, 1>;

    template <typename T>
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    template <typename T>
    using Mat2 = Eigen::Matrix<T, 2, 2>;

    template <typename T>
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    template <typename T>
    using Mat4 = Eigen::Matrix<T, 4, 4>;

    template <typename T>
    class PinholeCamera {
        public:
        static constexpr size_t parameter_size = 4;
        std::array<T, parameter_size> mvParameters; // fx, fy, cx, cy

        hd_fn PinholeCamera(const std::array<T, parameter_size> &params) : mvParameters(params) {}

        hd_fn Vec2<T> project(const Vec3<T> &v3D) const {
            Vec2<T> res;
            res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
            res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];

            return res;
        }

        hd_fn Eigen::Matrix<T, 2, 3> projectJac(const Vec3<T> &v3D) const {
            Eigen::Matrix<T, 2, 3> Jac;
            Jac(0, 0) = mvParameters[0] / v3D[2];
            Jac(0, 1) = 0.f;
            Jac(0, 2) = -mvParameters[0] * v3D[0] / (v3D[2] * v3D[2]);
            Jac(1, 0) = 0.f;
            Jac(1, 1) = mvParameters[1] / v3D[2];
            Jac(1, 2) = -mvParameters[1] * v3D[1] / (v3D[2] * v3D[2]);

            return Jac;
        }

        hd_fn T uncertainty2(const Vec2<T> &p2D) const {
            return 1.0;
        }

    };

    template <typename T>
    class KannalaBrandt8Camera {
        public:

        static constexpr size_t parameter_size = 8;
        std::array<T, parameter_size> mvParameters;

        hd_fn KannalaBrandt8Camera(const std::array<T, parameter_size> &params) : mvParameters(params) {}

        hd_fn Vec2<T> project(const Vec3<T> &v3D) const {
            const T x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1];
            const T theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
            const T psi = atan2f(v3D[1], v3D[0]);

            const T theta2 = theta * theta;
            const T theta3 = theta * theta2;
            const T theta5 = theta3 * theta2;
            const T theta7 = theta5 * theta2;
            const T theta9 = theta7 * theta2;
            const T r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5
                            + mvParameters[6] * theta7 + mvParameters[7] * theta9;

            Vec2<T> res;
            res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
            res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];

            return res;
        }

        hd_fn Eigen::Matrix<T, 2, 3> projectJac(const Vec3<T> &v3D) const {
            const T x2 = v3D[0] * v3D[0], y2 = v3D[1] * v3D[1], z2 = v3D[2] * v3D[2];
            const T r2 = x2 + y2;
            const T r = sqrt(r2);
            const T r3 = r2 * r;
            const T theta = atan2(r, v3D[2]);

            const T theta2 = theta * theta, theta3 = theta2 * theta;
            const T theta4 = theta2 * theta2, theta5 = theta4 * theta;
            const T theta6 = theta2 * theta4, theta7 = theta6 * theta;
            const T theta8 = theta4 * theta4, theta9 = theta8 * theta;

            const T f = theta + theta3 * mvParameters[4] + theta5 * mvParameters[5] + theta7 * mvParameters[6] +
                    theta9 * mvParameters[7];
            const T fd = 1 + 3 * mvParameters[4] * theta2 + 5 * mvParameters[5] * theta4 + 7 * mvParameters[6] * theta6 +
                    9 * mvParameters[7] * theta8;

            Eigen::Matrix<T, 2, 3> JacGood;
            JacGood(0, 0) = mvParameters[0] * (fd * v3D[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
            JacGood(1, 0) =
                    mvParameters[1] * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) - f * v3D[1] * v3D[0] / r3);

            JacGood(0, 1) =
                    mvParameters[0] * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) - f * v3D[1] * v3D[0] / r3);
            JacGood(1, 1) = mvParameters[1] * (fd * v3D[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

            JacGood(0, 2) = -mvParameters[0] * fd * v3D[0] / (r2 + z2);
            JacGood(1, 2) = -mvParameters[1] * fd * v3D[1] / (r2 + z2);

            return JacGood;
        }

        hd_fn T uncertainty2(const Vec2<T> &p2D) const {
            return 1.0;
        }

    };



    using namespace graphite;

    // template<typename T>
    // hd_fn Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T,3,3> &R) {
    //     // Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    //     // return svd.matrixU() * svd.matrixV().transpose();
    //     // Sophus::SO3<T> so3(R);
    //     // so3.normalize();
    //     // return so3.matrix();
    //     return Sophus::SO3<T>::fitToSO3(R).matrix();
    // }

    template <typename T>
    hd_fn void EnsureDetPositive(Eigen::Matrix<T, 3, 3>& R) {
        T det = R.determinant();
        if (det < T(0)) {
            R.col(2) = -R.col(2);  // Flip last column to fix improper rotation
        }
    }

    template <typename T>
    hd_fn Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T, 3, 3>& R) {
        Mat3<T> RTR = R.transpose() * R;

        // First-order inverse sqrt approximation: inv_sqrt(RTR) ≈ 1.5 * I - 0.5 * RTR
        Mat3<T> inv_sqrt = Mat3<T>::Identity() * T(1.5) - RTR * T(0.5);
        // Mat3<T> inv_sqrt = RTR.inverse().cwiseSqrt();

        // Apply polar step: R_new = R * inv_sqrt
        Mat3<T> result = R * inv_sqrt;
        EnsureDetPositive(result);
        return result;
    }

    /*
    template <typename T>
    hd_fn void JacobiSymmetricRotate(Mat3<T>& A, Mat3<T>& V, const int p, const int q) {
        const T apq = A(p, q);
        const T eps = T(1e-12);
        if (abs(apq) <= eps) {
            return;
        }

        const T app = A(p, p);
        const T aqq = A(q, q);
        const T tau = (aqq - app) / (T(2) * apq);
        const T t = (tau >= T(0))
                        ? T(1) / (tau + sqrt(T(1) + tau * tau))
                        : -T(1) / (-tau + sqrt(T(1) + tau * tau));
        const T c = T(1) / sqrt(T(1) + t * t);
        const T s = t * c;

        Mat3<T> J = Mat3<T>::Identity();
        J(p, p) = c;
        J(q, q) = c;
        J(p, q) = s;
        J(q, p) = -s;

        A = J.transpose() * A * J;
        V = V * J;
    }

    template <typename T>
    hd_fn Mat3<T> NormalizeRotationJacobiSVD(const Mat3<T>& R) {
        Mat3<T> ATA = R.transpose() * R;
        Mat3<T> V = Mat3<T>::Identity();

        #pragma unroll
        for (int sweep = 0; sweep < 6; ++sweep) {
            JacobiSymmetricRotate(ATA, V, 0, 1);
            JacobiSymmetricRotate(ATA, V, 0, 2);
            JacobiSymmetricRotate(ATA, V, 1, 2);
        }

        T l0 = ATA(0, 0);
        T l1 = ATA(1, 1);
        T l2 = ATA(2, 2);
        if (l0 < T(0)) l0 = T(0);
        if (l1 < T(0)) l1 = T(0);
        if (l2 < T(0)) l2 = T(0);

        // Keep singular values ordered (descending) for stable sign correction.
        if (l0 < l1) {
            const T tmp = l0;
            l0 = l1;
            l1 = tmp;
            V.col(0).swap(V.col(1));
        }
        if (l1 < l2) {
            const T tmp = l1;
            l1 = l2;
            l2 = tmp;
            V.col(1).swap(V.col(2));
        }
        if (l0 < l1) {
            const T tmp = l0;
            l0 = l1;
            l1 = tmp;
            V.col(0).swap(V.col(1));
        }

        const T s0 = sqrt(l0);
        const T s1 = sqrt(l1);
        const T s2 = sqrt(l2);

        Mat3<T> U = R * V;
        const T eps = T(1e-9);
        if (s0 <= eps || s1 <= eps || s2 <= eps) {
            // Degenerate case fallback to iterative polar step.
            Mat3<T> result = R;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const Mat3<T> RTR = result.transpose() * result;
                const Mat3<T> inv_sqrt = Mat3<T>::Identity() * T(1.5) - RTR * T(0.5);
                result = result * inv_sqrt;
            }
            EnsureDetPositive(result);
            return result;
        }

        U.col(0) /= s0;
        U.col(1) /= s1;
        U.col(2) /= s2;

        Mat3<T> Rproj = U * V.transpose();
        if (Rproj.determinant() < T(0)) {
            Mat3<T> D = Mat3<T>::Identity();
            D(2, 2) = T(-1);
            Rproj = U * D * V.transpose();
        }

        return Rproj;
    }


    template <typename T>
    hd_fn Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T, 3, 3>& R) {
        // GPU-side 3x3 Jacobi SVD projection to SO(3): R = U * V^T.
        return NormalizeRotationJacobiSVD(R);
    }
    */

    // template <typename T>
    // hd_fn Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T, 3, 3>& R) {
    //     // Device-safe iterative polar decomposition (Newton-Schulz steps), no SVD required.
    //     Mat3<T> result = R;
    //     #pragma unroll
    //     for (int i = 0; i < 4; ++i) {
    //         const Mat3<T> RTR = result.transpose() * result;
    //         const Mat3<T> inv_sqrt = Mat3<T>::Identity() * T(1.5) - RTR * T(0.5);
    //         result = result * inv_sqrt;
    //     }
    //     EnsureDetPositive(result);
    //     return result;
    // }

    // template <typename T>
    // hd_fn Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T, 3, 3>& R) {
    //     Mat3<T> RTR = R.transpose() * R;

    //     // First-order inverse sqrt approximation: inv_sqrt(RTR) ≈ 1.5 * I - 0.5 * RTR
    //     Mat3<T> inv_sqrt = Mat3<T>::Identity() * T(1.5) - RTR * T(0.5);
    //     // Mat3<T> inv_sqrt = RTR.inverse().cwiseSqrt();

    //     // Apply polar step: R_new = R * inv_sqrt
    //     Mat3<T> result = R * inv_sqrt;
    //     EnsureDetPositive(result);
    //     return result;
    // }

    // template <typename T>
    // hd_fn Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T, 3, 3>& R0) {

    //     Mat3<T> r = R0;

    //     #pragma unroll
    //     for (int i = 0; i < 4; ++i) {
    //         Mat3<T> RTR = r.transpose() * r;
    //         r = T(1.5)*r - T(0.5)*r*RTR;
    //     }


    //     EnsureDetPositive(r);
    //     return r;
    // }

    // template<typename T>
    // hd_fn Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T,3,3> &R) {
    //     // Gram-Schmidt orthonormalization for device compatibility
    //     Eigen::Matrix<T,3,1> x = R.col(0);
    //     Eigen::Matrix<T,3,1> y = R.col(1);
    //     Eigen::Matrix<T,3,1> z = R.col(2);

    //     x.normalize();
    //     y = y - x * x.dot(y);
    //     y.normalize();
    //     z = x.cross(y);

    //     Eigen::Matrix<T,3,3> Rn;
    //     Rn.col(0) = x;
    //     Rn.col(1) = y;
    //     Rn.col(2) = z;
    //     return Rn;
    // }

    template <typename T>
    hd_fn Mat3<T> ExpSO3(const T x, const T y, const T z)
    {
        const T d2 = x*x+y*y+z*z;
        const T d = sqrt(d2);
        Mat3<T> W;
        W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
        if(d<1e-5)
        {
            Mat3<T> res = Mat3<T>::Identity() + W +0.5*W*W;
            return NormalizeRotation(res);
        }
        else
        {
            Mat3<T> res = Mat3<T>::Identity() + W*sin(d)/d + W*W*(T(1.0)-cos(d))/d2;
            return NormalizeRotation(res);
        }
    }

    template <typename T>
    hd_fn Mat3<T> ExpSO3(const Vec3<T> &w)
    {
        return ExpSO3<T>(w[0],w[1],w[2]);
    }

    template <typename T>
    hd_fn Vec3<T> LogSO3(const Mat3<T> &R) {
        const T tr = R(0,0)+R(1,1)+R(2,2);
        Vec3<T> w;
        w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
        const T costheta = (tr-1.0)*0.5f;
        if(costheta>1 || costheta<-1)
            return w;
        const T theta = acos(costheta);
        const T s = sin(theta);
        // if(fabs(s)<1e-5)
        if(abs(s)<1e-5)
            return w;
        else
            return theta*w/s;
    }

    template <typename T>
    hd_fn Mat3<T> InverseRightJacobianSO3(const T x, const T y, const T z)
    {
        const T d2 = x*x+y*y+z*z;
        const T d = sqrt(d2);

        Mat3<T> W;
        W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
        if(d<1e-5)
            return Mat3<T>::Identity();
        else
            return Mat3<T>::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
    }

    template <typename T>
    hd_fn Mat3<T> InverseRightJacobianSO3(const Vec3<T> &v)
    {
        return InverseRightJacobianSO3(v[0],v[1],v[2]);
    }

    template <typename T>
    hd_fn Mat3<T> RightJacobianSO3(const T x, const T y, const T z)
    {
        const T d2 = x*x+y*y+z*z;
        const T d = sqrt(d2);

        Mat3<T> W;
        W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
        if(d<1e-5)
        {
            return Mat3<T>::Identity();
        }
        else
        {
            return Mat3<T>::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
        }
    }
    template <typename T>
    hd_fn Mat3<T> RightJacobianSO3(const Vec3<T> &v)
    {
        return RightJacobianSO3(v[0],v[1],v[2]);
    }


    // Pose
    template <typename T, typename C>
    class ImuCamPose {
        public:
        // Methods
        ImuCamPose(){

            its = 0;
            num_cams = 1;
            pCamera[0] = nullptr;
            Rcw[0].setIdentity();
            tcw[0].setZero();
            tcb[0].setZero();
            Rcb[0].setIdentity();
            Rbc[0].setIdentity();
            tbc[0].setZero();
            bf = T(0);
            // For posegraph 4DoF
            Rwb0.setIdentity();
            DR.setIdentity();
            
        }
        
        ImuCamPose(ORB_SLAM3::KeyFrame *pKF, C** cameras) : its(0)
        {

            // Load IMU pose
            twb = pKF->GetImuPosition().cast<T>();
            Rwb = pKF->GetImuRotation().cast<T>();

            // Left camera
            tcw[0] = pKF->GetTranslation().cast<T>();
            Rcw[0] = pKF->GetRotation().cast<T>();
            tcb[0] = pKF->mImuCalib.mTcb.translation().cast<T>();
            Rcb[0] = pKF->mImuCalib.mTcb.rotationMatrix().cast<T>();
            Rbc[0] = Rcb[0].transpose();
            tbc[0] = pKF->mImuCalib.mTbc.translation().cast<T>();
            if (cameras) {
                pCamera[0] = cameras[0];
            }
            bf = pKF->mbf;

            if (cameras && cameras[1])
            {
                Mat4<T> Trl = pKF->GetRelativePoseTrl().matrix().cast<T>();
                Rcw[1] = Trl.template block<3,3>(0,0) * Rcw[0];
                tcw[1] = Trl.template block<3,3>(0,0) * tcw[0] + Trl.template block<3,1>(0,3);
                tcb[1] = Trl.template block<3,3>(0,0) * tcb[0] + Trl.template block<3,1>(0,3);
                Rcb[1] = Trl.template block<3,3>(0,0) * Rcb[0];
                Rbc[1] = Rcb[1].transpose();
                tbc[1] = -Rbc[1] * tcb[1];
                pCamera[1] = cameras[1];
                num_cams = 2;
            }
            else {
                num_cams = 1;
            }

            // For posegraph 4DoF
            Rwb0 = Rwb;
            DR.setIdentity();   
        }

        ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, ORB_SLAM3::KeyFrame* pKF): its(0)
        {
            // This is only for posegrpah, we do not care about multicamera
            // tcw.resize(1);
            // Rcw.resize(1);
            // tcb.resize(1);
            // Rcb.resize(1);
            // Rbc.resize(1);
            // tbc.resize(1);
            // pCamera.resize(1);

            tcb[0] = pKF->mImuCalib.mTcb.translation().cast<double>();
            Rcb[0] = pKF->mImuCalib.mTcb.rotationMatrix().cast<double>();
            Rbc[0] = Rcb[0].transpose();
            tbc[0] = pKF->mImuCalib.mTbc.translation().cast<double>();
            twb = _Rwc * tcb[0] + _twc;
            Rwb = _Rwc * Rcb[0];
            Rcw[0] = _Rwc.transpose();
            tcw[0] = -Rcw[0] * _twc;
            // pCamera[0] = pKF->mpCamera;
            pCamera[0] = nullptr; // didn't implement because we don't use it
            bf = pKF->mbf;
            num_cams = 1; // added

            // For posegraph 4DoF
            Rwb0 = Rwb;
            DR.setIdentity();
        }        

        hd_fn Vec2<T> Project(const Vec3<T> &Xw, int cam_idx) const
        {
            Vec3<T> Xc = Rcw[cam_idx] * Xw + tcw[cam_idx];

            return pCamera[cam_idx]->project(Xc);
        }

        hd_fn Vec3<T> ProjectStereo(const Vec3<T> &Xw, int cam_idx) const
        {
            Vec3<T> Pc = Rcw[cam_idx] * Xw + tcw[cam_idx];
            Vec3<T> pc;
            T invZ = 1/Pc(2);
            pc.head(2) = pCamera[cam_idx]->project(Pc);
            pc(2) = pc(0) - bf*invZ;
            return pc;
        }

        hd_fn void Update(const T *pu)
        {
            Vec3<T> ur, ut;
            ur << pu[0], pu[1], pu[2];
            ut << pu[3], pu[4], pu[5];

            // Update body pose
            twb += Rwb * ut;
            Rwb = Rwb * ExpSO3(ur);

            // Normalize rotation after 5 updates
            its++;
            if(its>=3)
            {
                NormalizeRotation(Rwb); // OS3
                // const auto Rwb_norm = NormalizeRotation(Rwb); // fixed?
                // Rwb = Rwb_norm;
                its=0;
            }

            // Update camera poses
            const Mat3<T> Rbw = Rwb.transpose();
            const Vec3<T> tbw = -Rbw * twb;

            for(int i=0; i< num_cams; i++)
            {
                Rcw[i] = Rcb[i] * Rbw;
                tcw[i] = Rcb[i] * tbw + tcb[i];
            }

        }

        hd_fn void UpdateW(const T *pu)
        {
            Vec3<T> ur, ut;
            ur << pu[0], pu[1], pu[2];
            ut << pu[3], pu[4], pu[5];


            const Mat3<T> dR = ExpSO3(ur);
            DR = dR * DR;
            Rwb = DR * Rwb0;
            // Update body pose
            twb += ut;

            // Normalize rotation after 5 updates
            its++;
            if(its>=5)
            {
                DR(0,2) = 0.0;
                DR(1,2) = 0.0;
                DR(2,0) = 0.0;
                DR(2,1) = 0.0;
                NormalizeRotation(DR);
                its = 0;
            }

            // Update camera pose
            const Mat3<T> Rbw = Rwb.transpose();
            const Vec3<T> tbw = -Rbw * twb;

            for(int i=0; i< num_cams; i++)
            {
                Rcw[i] = Rcb[i] * Rbw;
                tcw[i] = Rcb[i] * tbw+tcb[i];
            }
        }
        
        // template <typename P>
        // hd_fn void get_pgo_params(P* params) const
        // {
        //     Eigen::Map<Eigen::Matrix<P, 4, 1>> p(params);

        //     const auto tvec = twb;
        //     const auto rvec = LogSO3(DR);
        //     p(0) = tvec(0);
        //     p(1) = tvec(1);
        //     p(2) = tvec(2);
        //     p(3) = rvec(2);

        // }

        hd_fn bool isDepthPositive(const Vec3<T>& Xw, int cam_idx) const
        {
            return (Rcw[cam_idx].row(2) * Xw + tcw[cam_idx](2)) > 0.0;
        }

    public:
        // Member variables

        // For IMU
        Mat3<T> Rwb;
        Vec3<T> twb;

        // For set of cameras
        size_t num_cams;
        static constexpr size_t max_cams = 2;
        std::array<Mat3<T>, max_cams>  Rcw;
        std::array<Vec3<T>, max_cams> tcw;
        std::array<Mat3<T>, max_cams> Rcb, Rbc;
        std::array<Vec3<T>, max_cams> tcb, tbc;
        T bf;
        std::array<C*, max_cams> pCamera;

        // For posegraph 4DoF
        Mat3<T> Rwb0;
        Mat3<T> DR;

        int its;

    };

    template <typename T, typename C> struct PoseTraits {
        static constexpr size_t dimension = 6;
        using Vertex = ImuCamPose<T, C>;

        template <typename P>
        hd_fn static void parameters(const Vertex &vertex, P* params) {}

        hd_fn static void update(Vertex &vertex, const T *delta) {
            vertex.Update(delta);
        }
    };

    template <typename T, typename S, typename C>
    using PoseDescriptor = VertexDescriptor<T, S, PoseTraits<T, C>>;

}