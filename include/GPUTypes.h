#pragma once

/*
    This file will hold the GPU types needed for the optimization.
*/
#include <graphite/optimizer/levenberg_marquardt.hpp>
#include <graphite/types.hpp>
#include <graphite/utils.hpp>
#include "GPUPose.h"
#include "GPUImu.h"

namespace gpu {

using namespace graphite;

template <typename T>
using Velocity = Eigen::Matrix<T, 3, 1>;

template <typename T> struct VelocityTraits {
  static constexpr size_t dimension = 3;
  using Vertex = Velocity<T>;

  template <typename P>
  hd_fn static void parameters(const Vertex &vertex, P* params) {}

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};


template <typename T>
using GyroBias = Eigen::Matrix<T, 3, 1>;

template <typename T> struct GyroBiasTraits {
  static constexpr size_t dimension = 3;
  using Vertex = GyroBias<T>;

  template <typename P>
  hd_fn static void parameters(const Vertex &vertex, P* params) {}

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};


template <typename T>
using AccBias = Eigen::Matrix<T, 3, 1>;

template <typename T> struct AccBiasTraits {
  static constexpr size_t dimension = 3;
  using Vertex = AccBias<T>;

  template <typename P>
  hd_fn static void parameters(const Vertex &vertex, P* params) {}

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};

template <typename T, typename S>
using VelocityDescriptor = VertexDescriptor<T, S, VelocityTraits<T>>;

template <typename T, typename S>
using GyroBiasDescriptor = VertexDescriptor<T, S, GyroBiasTraits<T>>;

template <typename T, typename S>
using AccBiasDescriptor = VertexDescriptor<T, S, AccBiasTraits<T>>;

template <typename T, typename S, typename L> struct GyroRWConstraint {
  static constexpr size_t dimension = 3;
  using VertexDescriptors =
      std::tuple<GyroBiasDescriptor<T, S>, GyroBiasDescriptor<T, S>>;
  // using Observation = Eigen::Matrix<T, dimension, 3>; // TODO: This actually doesn't have an observation
  using Observation = Empty;
  using Data = Empty;
  using Loss = L;
//   using Differentiation = DifferentiationMode::Auto;
  using Differentiation = DifferentiationMode::Manual;

  template <typename D>
  hd_fn static void
  error(const GyroBias<T> & v1, const GyroBias<T> & v2, D *error) {
        Eigen::Map<Eigen::Matrix<D, dimension, 1>> e(error);
        e = v2 - v1;
  }

  template <typename D, size_t I>
  hd_fn static void jacobian(const GyroBias<T> &vg1, const GyroBias<T> &vg2, D *jacobian) {

    if constexpr (I == 0) {
        Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
        J = -Eigen::Matrix<D, 3, 3>::Identity();
    }
    else {
        Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
        J = Eigen::Matrix<D, 3, 3>::Identity();
    }

  }


};


template <typename T, typename S, typename L> struct AccRWConstraint {
  static constexpr size_t dimension = 3;
  using VertexDescriptors =
      std::tuple<AccBiasDescriptor<T, S>, AccBiasDescriptor<T, S>>;
  using Observation = Empty;
  using Data = Empty;
  using Loss = L;
  using Differentiation = DifferentiationMode::Manual;

  template <typename D>
  hd_fn static void
  error(const AccBias<T> & v1, const AccBias<T> & v2, D *error) {
        Eigen::Map<Eigen::Matrix<D, dimension, 1>> e(error);
        e = v2 - v1;
  }

  template <typename D, size_t I>
  hd_fn static void jacobian(const AccBias<T> &va1, const AccBias<T> &va2, D *jacobian) {

    if constexpr (I == 0) {
        Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
        J = -Eigen::Matrix<D, 3, 3>::Identity();
    }
    else {
        Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
        J = Eigen::Matrix<D, 3, 3>::Identity();
    }

  }

};

// Prior Gyro and Acc Bias Constraints

template <typename T, typename S, typename L> struct GyroRWPrior {
  static constexpr size_t dimension = 3;
  using VertexDescriptors =
      std::tuple<GyroBiasDescriptor<T, S>>;
  using Observation = Eigen::Matrix<T, dimension, 1>; // storing prior as observation
  using Data = Empty;
  using Loss = L;
  using Differentiation = DifferentiationMode::Manual;

  template <typename D>
  hd_fn static void
  error(const GyroBias<T>& v1, const Observation &obs, D *error) {
        Eigen::Map<Eigen::Matrix<D, dimension, 1>> e(error);
        e = obs - v1;
  }

  template <typename D, size_t I>
  hd_fn static void jacobian(const GyroBias<T> &vg1,
                             const Observation &obs, D *jacobian) {

    if constexpr (I == 0) {
        Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
        J = Eigen::Matrix<D, 3, 3>::Identity(); // shouldn't this be negative?
    }

  }

};

template <typename T, typename S, typename L> struct AccRWPrior {
  static constexpr size_t dimension = 3;
  using VertexDescriptors =
      std::tuple<AccBiasDescriptor<T, S>>;
  using Observation = Eigen::Matrix<T, dimension, 1>; // storing prior as observation
  using Data = Empty;
  using Loss = L;
  using Differentiation = DifferentiationMode::Manual;

  template <typename D>
  hd_fn static void
  error(const AccBias<T>& v1, const Observation &obs, D *error) {
        Eigen::Map<Eigen::Matrix<D, dimension, 1>> e(error);
        e = obs - v1;
  }

  template <typename D, size_t I>
  hd_fn static void jacobian(const AccBias<T> &va1,
                             const Observation &obs, D *jacobian) {

    if constexpr (I == 0) {
        Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
        J = Eigen::Matrix<D, 3, 3>::Identity(); // shouldn't this be negative?
    }

  }

};


// Map Point
template <typename T>
using SBAPointXYZ = Eigen::Matrix<T, 3, 1>;

template <typename T> struct SBAPointXYZTraits {
  static constexpr size_t dimension = 3;
  using Vertex = SBAPointXYZ<T>;

  template <typename P>
  hd_fn static void parameters(const Vertex &vertex, P* params) {}

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};

template <typename T, typename S>
using SBAPointXYZDescriptor = VertexDescriptor<T, S, SBAPointXYZTraits<T>>;


// Mono constraint
template <typename T, typename S, typename L, typename C> struct MonoConstraint {
  static constexpr size_t dimension = 2;
  using VertexDescriptors =
      std::tuple<SBAPointXYZDescriptor<T, S>, PoseDescriptor<T, S, C>>;
  using Observation = Vec2<T>; 
  using Data = int; // cam_idx
  using Loss = L;
  using Differentiation = DifferentiationMode::Manual;

  // Aliases
  using Pose = typename PoseDescriptor<T, S, C>::VertexType;
  using Point = typename SBAPointXYZDescriptor<T, S>::VertexType;

  // Can't use the parameterizations
  template <typename D>
  hd_fn static void
  error(const Point& VPoint, const Pose& VPose, const Observation &obs, const Data& data, D *error) {
        const int cam_idx = data;
        Eigen::Map<Vec2<T>> e(error);
        e = obs - VPose.Project(VPoint,cam_idx);
  }

  template <typename D, size_t I>
  hd_fn static void jacobian(const Point &VPoint, const Pose &VPose,
                             const Observation &obs, const Data &data, D *jacobian) {

      const int cam_idx = data;
      const Mat3<D> &Rcw = VPose.Rcw[cam_idx];
      const Vec3<D> &tcw = VPose.tcw[cam_idx];
      const Vec3<D> Xc = Rcw*(VPoint) + tcw;
      const Vec3<D> Xb = VPose.Rbc[cam_idx]*Xc+VPose.tbc[cam_idx];
      const Mat3<D> &Rcb = VPose.Rcb[cam_idx];

      const Eigen::Matrix<D,2,3> proj_jac = VPose.pCamera[cam_idx]->projectJac(Xc);


      if constexpr (I == 0) {
        Eigen::Map<Eigen::Matrix<D, 2, 3>> J(jacobian);
        J = -proj_jac * Rcw;
      }
      else {
          Eigen::Map<Eigen::Matrix<D, 2, 6>> J(jacobian);
          Eigen::Matrix<D,3,6> SE3deriv;
          D x = Xb(0);
          D y = Xb(1);
          D z = Xb(2);

          SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
                  -z , 0.0, x, 0.0, 1.0, 0.0,
                  y ,  -x , 0.0, 0.0, 0.0, 1.0;

          J = proj_jac * Rcb * SE3deriv;
      }

  }
};

// Stereo Constraint
template <typename T, typename S, typename L, typename C> struct StereoConstraint {
  static constexpr size_t dimension = 3;
  using VertexDescriptors =
      std::tuple<SBAPointXYZDescriptor<T, S>, PoseDescriptor<T, S, C>>;
  using Observation = Vec3<T>; 
  using Data = int; // cam_idx
  using Loss = L;
  using Differentiation = DifferentiationMode::Manual;

  // Aliases
  using Pose = typename PoseDescriptor<T, S, C>::VertexType;
  using Point = typename SBAPointXYZDescriptor<T, S>::VertexType;

  // Can't use the parameterizations
  template <typename D>
  hd_fn static void
  error(const Point& VPoint, const Pose& VPose, const Observation &obs, const Data& data, D *error) {
        const int cam_idx = data;
        Eigen::Map<Vec3<D>> e(error);
        e = obs - VPose.ProjectStereo(VPoint,cam_idx);
  }

  template <typename D, size_t I>
  hd_fn static void jacobian(const Point &VPoint, const Pose &VPose,
                             const Observation &obs, const Data &data, D *jacobian) {

      const int cam_idx = data;
      const Mat3<D> &Rcw = VPose.Rcw[cam_idx];
      const Vec3<D> &tcw = VPose.tcw[cam_idx];
      const Vec3<D> Xc = Rcw*(VPoint) + tcw;
      const Vec3<D> Xb = VPose.Rbc[cam_idx]*Xc+VPose.tbc[cam_idx];
      const Mat3<D> &Rcb = VPose.Rcb[cam_idx];
      const D bf = VPose.bf;
      const D inv_z2 = 1.0/(Xc(2)*Xc(2));

      Mat3<D> proj_jac;
      proj_jac.block<2,3>(0,0) = VPose.pCamera[cam_idx]->projectJac(Xc);
      proj_jac.block<1,3>(2,0) = proj_jac.block<1,3>(0,0);
      proj_jac(2,2) += bf*inv_z2;


      if constexpr (I == 0) {
        Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
        J = -proj_jac * Rcw;
      }
      else {
          Eigen::Map<Eigen::Matrix<D, 3, 6>> J(jacobian);
          Eigen::Matrix<T,3,6> SE3deriv;
          T x = Xb(0);
          T y = Xb(1);
          T z = Xb(2);

          SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
                  -z , 0.0, x, 0.0, 1.0, 0.0,
                  y ,  -x , 0.0, 0.0, 0.0, 1.0;

          J = proj_jac * Rcb * SE3deriv;
      }

  }

};


template <typename T>
class InertialConstraintData {
  public:
  InertialConstraintData(ORB_SLAM3::IMU::Preintegrated *pInt):
    // JRg(pInt->JRg.cast<T>()),
    // JVg(pInt->JVg.cast<T>()), JPg(pInt->JPg.cast<T>()), JVa(pInt->JVa.cast<T>()),
    // JPa(pInt->JPa.cast<T>()), 
    mpInt(pInt), dt(pInt->dT)
  {
        g << 0, 0, -ORB_SLAM3::IMU::GRAVITY_VALUE;
  }

  template <typename S>
  Eigen::Matrix<S, 9, 9> get_information_matrix(ORB_SLAM3::IMU::Preintegrated *pInt) const {
    Eigen::Matrix<double, 9, 9> Info = pInt->C.block<9,9>(0,0).cast<double>().inverse();
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
    for(int i=0;i<9;i++)
        if(eigs[i]<1e-12)
            eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    return Info.cast<S>();
  }

  public:
  // const Mat3<T> JRg, JVg, JPg, JVa, JPa; // already stored in Preintegrated
  // Preintegrated<T>* mpInt;
  Preintegrated<T> mpInt; // FIX: This should be a pointer, but we use a copy for simplicity
  const T dt;
  Vec3<T> g;
};

// Inertial Constraint
template <typename T, typename S, typename L, 
typename PoseDescriptor> struct InertialConstraint {
  static constexpr size_t dimension = 9;
  using VertexDescriptors =
      std::tuple<PoseDescriptor, 
                 VelocityDescriptor<T, S>,
                 GyroBiasDescriptor<T, S>,
                 AccBiasDescriptor<T, S>,
                 PoseDescriptor,
                 VelocityDescriptor<T, S>>;
  using Observation = Empty; 
  using Data = InertialConstraintData<T>;
  using Loss = L;
  using Differentiation = DifferentiationMode::Manual;

  // Aliases
  using Pose = typename PoseDescriptor::VertexType;

  // Can't use the parameterizations
  // TODO: Implement alternate error function w/o parameterizations
  template <typename D>
  hd_fn static void
  error(const Pose& VP1, const Velocity<T> & VV1, const GyroBias<T> & VG1, 
        const AccBias<T> & VA1, const Pose &VP2, const Velocity<T> & VV2,
        const Data& data, D *error) {

        const auto mpInt = &(data.mpInt);

        const Bias<D> b1(VA1[0],VA1[1],VA1[2],VG1[0],VG1[1],VG1[2]);
        const Mat3<D> dR = mpInt->GetDeltaRotation(b1).template cast<D>();
        const Vec3<D> dV = mpInt->GetDeltaVelocity(b1).template cast<D>();
        const Vec3<D> dP = mpInt->GetDeltaPosition(b1).template cast<D>();

        const auto g = data.g;
        const auto dt = data.dt;
        const Vec3<D> er = LogSO3<D>(dR.transpose()*VP1.Rwb.transpose()*VP2.Rwb);
        const Vec3<D> ev = VP1.Rwb.transpose()*(VV2 - VV1 - g*dt) - dV;
        const Vec3<D> ep = VP1.Rwb.transpose()*(VP2.twb - VP1.twb
                                                                  - VV1*dt - g*dt*dt/2) - dP;

        Eigen::Map<Eigen::Matrix<D, dimension, 1>> e(error);
        e << er, ev, ep;
  }

  template <typename D, size_t I>
  hd_fn static void jacobian(const Pose &VP1, const Velocity<T>& VV1, const GyroBias<T>& VG1,
                            const AccBias<T>& VA1, const Pose& VP2, const Velocity<T>& VV2,
                            const Data &data, D *jacobian) {
    const auto mpInt = &(data.mpInt);
    const auto g = data.g;
    const auto dt = data.dt;

    const Bias<D> b1(VA1[0],VA1[1],VA1[2],VG1[0],VG1[1],VG1[2]);
    const auto db = mpInt->GetDeltaBias(b1);
    Vec3<D> dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Mat3<D> Rwb1 = VP1.Rwb;
    const Mat3<D> Rbw1 = Rwb1.transpose();
    const Mat3<D> Rwb2 = VP2.Rwb;

    const Mat3<D> dR = mpInt->GetDeltaRotation(b1).template cast<D>();
    const Mat3<D> eR = dR.transpose()*Rbw1*Rwb2;
    const Vec3<D> er = LogSO3(eR);
    const Mat3<D> invJr = InverseRightJacobianSO3(er);

    if constexpr (I == 0) {
      Eigen::Map<Eigen::Matrix<D, 9, 6>> J(jacobian);
      // Jacobians wrt Pose 1
      J.setZero();
      // rotation
      J.block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1; // OK
      J.block<3,3>(3,0) = Sophus::SO3<D>::hat(Rbw1*(VV2 - VV1 - g*dt)).template cast<D>(); // OK
      J.block<3,3>(6,0) = Sophus::SO3<D>::hat(Rbw1*(VP2.twb - VP1.twb
                                                    - VV1*dt - 0.5*g*dt*dt)).template cast<D>(); // OK
      // translation
      J.block<3,3>(6,3) = -Mat3<D>::Identity(); // OK
    }
    else if constexpr (I == 1) {
      Eigen::Map<Eigen::Matrix<D, 9, 3>> J(jacobian);
      // Jacobians wrt Velocity 1
      J.setZero();
      J.block<3,3>(3,0) = -Rbw1; // OK
      J.block<3,3>(6,0) = -Rbw1*dt; // OK
    }
    else if constexpr (I == 2) {
      Eigen::Map<Eigen::Matrix<D, 9, 3>> J(jacobian);
      // Jacobians wrt Gyro 1
      J.setZero();
      J.block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3<D>(mpInt->JRg*dbg)*mpInt->JRg; // OK
      J.block<3,3>(3,0) = -mpInt->JVg; // OK
      J.block<3,3>(6,0) = -mpInt->JPg; // OK
    }
    else if constexpr (I == 3) {
      Eigen::Map<Eigen::Matrix<D, 9, 3>> J(jacobian);
      // Jacobians wrt Accelerometer 1
      J.setZero();
      J.block<3,3>(3,0) = -mpInt->JVa; // OK
      J.block<3,3>(6,0) = -mpInt->JPa; // OK
    }
    else if constexpr (I == 4) {
      Eigen::Map<Eigen::Matrix<D, 9, 6>> J(jacobian);
      // Jacobians wrt Pose 2
      J.setZero();
      // rotation
      J.template block<3,3>(0,0) = invJr; // OK
      // translation
      J.template block<3,3>(6,3) = Rbw1*Rwb2; // OK
    }
    else {
      Eigen::Map<Eigen::Matrix<D, 9, 3>> J(jacobian);
      // Jacobians wrt Velocity 2
      J.setZero();
      J.template block<3,3>(3,0) = Rbw1; // OK
    }

  }
};

template <typename T, typename S, typename L, 
typename PoseDescriptor>
using InertialConstraintDescriptor = graphite::FactorDescriptor<T, S, InertialConstraint<T, S, L, PoseDescriptor>>;

template<typename T, typename S, typename L>
using GyroRWConstraintDescriptor = graphite::FactorDescriptor<T, S, GyroRWConstraint<T, S, L>>;

template<typename T, typename S, typename L>
using AccRWConstraintDescriptor = graphite::FactorDescriptor<T, S, AccRWConstraint<T, S, L>>;

template<typename T, typename S, typename L>
using GyroRWPriorDescriptor = graphite::FactorDescriptor<T, S, GyroRWPrior<T, S, L>>;

template<typename T, typename S, typename L>
using AccRWPriorDescriptor = graphite::FactorDescriptor<T, S, AccRWPrior<T, S, L>>;

template <typename T, typename S, typename L, typename C>
using MonoConstraintDescriptor = graphite::FactorDescriptor<T, S, MonoConstraint<T, S, L, C>>;

template <typename T, typename S, typename L, typename C>
using StereoConstraintDescriptor = graphite::FactorDescriptor<T, S, StereoConstraint<T, S, L, C>>;


// ---- Pose-only constraints (fixed world point, optimize only pose) ----

template <typename T>
struct PoseOnlyData {
    Vec3<T> Xw;
    int cam_idx;
};

template <typename T, typename S, typename L, typename C>
struct MonoConstraintOnlyPose {
    static constexpr size_t dimension = 2;
    using VertexDescriptors = std::tuple<PoseDescriptor<T, S, C>>;
    using Observation = Vec2<T>;
    using Data = PoseOnlyData<T>;
    using Loss = L;
    using Differentiation = DifferentiationMode::Manual;

    using Pose = typename PoseDescriptor<T, S, C>::VertexType;

    template <typename D>
    hd_fn static void error(const Pose& VPose, const Observation& obs, const Data& d, D* e_ptr) {
        Eigen::Map<Vec2<T>> e(e_ptr);
        e = obs - VPose.Project(d.Xw, d.cam_idx);
    }

    template <typename D, size_t I>
    hd_fn static void jacobian(const Pose& VPose, const Observation& obs, const Data& d, D* jac_ptr) {
        const int cam_idx = d.cam_idx;
        const Mat3<D>& Rcw = VPose.Rcw[cam_idx];
        const Vec3<D>& tcw = VPose.tcw[cam_idx];
        const Vec3<D> Xc = Rcw * d.Xw + tcw;
        const Vec3<D> Xb = VPose.Rbc[cam_idx] * Xc + VPose.tbc[cam_idx];
        const Mat3<D>& Rcb = VPose.Rcb[cam_idx];
        const Eigen::Matrix<D, 2, 3> proj_jac = VPose.pCamera[cam_idx]->projectJac(Xc);

        Eigen::Map<Eigen::Matrix<D, 2, 6>> J(jac_ptr);
        Eigen::Matrix<T, 3, 6> SE3deriv;
        T x = Xb(0), y = Xb(1), z = Xb(2);
        SE3deriv << T(0), z, -y, T(1), T(0), T(0),
                    -z, T(0), x, T(0), T(1), T(0),
                     y, -x, T(0), T(0), T(0), T(1);
        J = proj_jac * Rcb * SE3deriv;
    }
};

template <typename T, typename S, typename L, typename C>
struct StereoConstraintOnlyPose {
    static constexpr size_t dimension = 3;
    using VertexDescriptors = std::tuple<PoseDescriptor<T, S, C>>;
    using Observation = Vec3<T>;
    using Data = PoseOnlyData<T>;
    using Loss = L;
    using Differentiation = DifferentiationMode::Manual;

    using Pose = typename PoseDescriptor<T, S, C>::VertexType;

    template <typename D>
    hd_fn static void error(const Pose& VPose, const Observation& obs, const Data& d, D* e_ptr) {
        Eigen::Map<Vec3<T>> e(e_ptr);
        e = obs - VPose.ProjectStereo(d.Xw, d.cam_idx);
    }

    template <typename D, size_t I>
    hd_fn static void jacobian(const Pose& VPose, const Observation& obs, const Data& d, D* jac_ptr) {
        const int cam_idx = d.cam_idx;
        const Mat3<D>& Rcw = VPose.Rcw[cam_idx];
        const Vec3<D>& tcw = VPose.tcw[cam_idx];
        const Vec3<D> Xc = Rcw * d.Xw + tcw;
        const Vec3<D> Xb = VPose.Rbc[cam_idx] * Xc + VPose.tbc[cam_idx];
        const Mat3<D>& Rcb = VPose.Rcb[cam_idx];
        const D bf = VPose.bf;
        const D inv_z2 = D(1) / (Xc(2) * Xc(2));

        Mat3<D> proj_jac;
        proj_jac.template block<2, 3>(0, 0) = VPose.pCamera[cam_idx]->projectJac(Xc);
        proj_jac.template block<1, 3>(2, 0) = proj_jac.template block<1, 3>(0, 0);
        proj_jac(2, 2) += bf * inv_z2;

        Eigen::Map<Eigen::Matrix<D, 3, 6>> J(jac_ptr);
        Eigen::Matrix<T, 3, 6> SE3deriv;
        T x = Xb(0), y = Xb(1), z = Xb(2);
        SE3deriv << T(0), z, -y, T(1), T(0), T(0),
                    -z, T(0), x, T(0), T(1), T(0),
                     y, -x, T(0), T(0), T(0), T(1);
        J = proj_jac * Rcb * SE3deriv;
    }
};

template <typename T, typename S, typename L, typename C>
using MonoConstraintOnlyPoseDescriptor = graphite::FactorDescriptor<T, S, MonoConstraintOnlyPose<T, S, L, C>>;

template <typename T, typename S, typename L, typename C>
using StereoConstraintOnlyPoseDescriptor = graphite::FactorDescriptor<T, S, StereoConstraintOnlyPose<T, S, L, C>>;

}