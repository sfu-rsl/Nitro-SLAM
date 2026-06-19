#pragma once
#include "ImuTypes.h"

namespace gpu {


template <typename T>
class Bias
{

public:
    hd_fn Bias():bax(0),bay(0),baz(0),bwx(0),bwy(0),bwz(0){}
    hd_fn Bias(const T &b_acc_x, const T &b_acc_y, const T &b_acc_z,
            const T &b_ang_vel_x, const T &b_ang_vel_y, const T &b_ang_vel_z):
            bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z){}
    
    Bias(const ORB_SLAM3::IMU::Bias &b) {
        bax = b.bax;
        bay = b.bay;
        baz = b.baz;
        bwx = b.bwx;
        bwy = b.bwy;
        bwz = b.bwz;
    }

    hd_fn void CopyFrom(Bias &b) {
        bax = b.bax;
        bay = b.bay;
        baz = b.baz;
        bwx = b.bwx;
        bwy = b.bwy;
        bwz = b.bwz;
    }

public:
    T bax, bay, baz;
    T bwx, bwy, bwz;
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


//Preintegration of Imu Measurements
template <typename T>
class Preintegrated
{

public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Preintegrated(ORB_SLAM3::IMU::Preintegrated* pImuPre): b(pImuPre->b) {
        JRg = pImuPre->JRg.cast<T>();
        JVg = pImuPre->JVg.cast<T>();
        JPg = pImuPre->JPg.cast<T>();
        JVa = pImuPre->JVa.cast<T>();
        JPa = pImuPre->JPa.cast<T>();
        dR = pImuPre->dR.cast<T>();
        dV = pImuPre->dV.cast<T>();
        dP = pImuPre->dP.cast<T>();
    }

    hd_fn Bias<T> GetDeltaBias(const Bias<T> &b_) const {
        return Bias<T>(b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz,b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
    }

    // hd_fn Mat3<T> GetDeltaRotation(const Bias<T> &b_) const {
    //     Vec3<T> dbg;
    //     dbg << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz;
    //     return NormalizeRotation<T>(dR * Sophus::SO3<T>::exp(JRg * dbg).matrix());
    // }

    hd_fn Mat3<T> GetDeltaRotation(const Bias<T> &b_) const {
        Vec3<T> dbg;
        dbg << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz;
        if(dbg.array().isNaN()[0]){ // per https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/608
          dbg = Vec3<T>(0,0,0);
        }
        return NormalizeRotation<T>(dR * Sophus::SO3<T>::exp(JRg * dbg).matrix());
    }

    hd_fn Vec3<T> GetDeltaVelocity(const Bias<T> &b_) const {
        Vec3<T> dbg, dba;
        dbg << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz;
        dba << b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz;
        return dV + JVg * dbg + JVa * dba;
    }
    hd_fn Vec3<T> GetDeltaPosition(const Bias<T> &b_) const {
        Vec3<T> dbg, dba;
        dbg << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz;
        dba << b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz;
        return dP + JPg * dbg + JPa * dba;
    }


public:
    // T dT;
    // Eigen::Matrix<T,15,15> C;
    // Eigen::Matrix<T,15,15> Info;
    // Eigen::DiagonalMatrix<T,6> Nga, NgaWalk;

    // Values for the original bias (when integration was computed)
    Bias<T> b;
    Mat3<T> dR;
    Vec3<T> dV, dP;
    Mat3<T> JRg, JVg, JVa, JPg, JPa;
    // Vec3<T> avgA, avgW;


// private:
    // Updated bias
    // Bias<T> bu;
    // Dif between original and updated bias
    // This is used to compute the updated values of the preintegration
    // Eigen::Matrix<T,6,1> db;

//     struct integrable
//     {
//         template<class Archive>
//         void serialize(Archive & ar, const unsigned int version)
//         {
//             ar & boost::serialization::make_array(a.data(), a.size());
//             ar & boost::serialization::make_array(w.data(), w.size());
//             ar & t;
//         }

//         // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//         integrable(){}
//         integrable(const Eigen::Vector3f &a_, const Eigen::Vector3f &w_ , const float &t_):a(a_),w(w_),t(t_){}
//         Eigen::Vector3f a, w;
//         float t;
//     };

//     std::vector<integrable> mvMeasurements;

//     std::mutex mMutex;

};

}