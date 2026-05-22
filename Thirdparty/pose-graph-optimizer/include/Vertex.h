#ifndef OPTIMIZER_VERTEX_H
#define OPTIMIZER_VERTEX_H

#include "types.hpp"

namespace optimizer {

class Vertex {

public:
    Vertex(int id, Mat3d& _Rcw, Mat3d& _Rcb, Mat3d& _Rbc, Mat3d& _Rwb, Vec3d& _tcw, Vec3d& _tcb, Vec3d& _tbc, Vec3d& _twb, double _bf);

    int getID() const;
    Mat3d getRcw() const;
    Mat3d getRcb() const;
    Mat3d getRbc() const;
    Mat3d getRwb() const;
    Vec3d gettcw() const;
    Vec3d gettcb() const;
    Vec3d gettbc() const;
    Vec3d gettwb() const;

    Mat3d getDR() const;

    bool getIsFixed() const;

    void setFixed(bool fixed);

private:
    Mat3d Rcw, Rcb, Rbc, Rwb;
    Vec3d tcw, tcb, tbc, twb;
    Mat3d DR; // Identity matrix
    Mat3d Rwb0; // initial rotation of world frame
    double bf;
    int id;

    bool isFixed;
};
} // namespace optimizer

#endif  // OPTIMIZER_VERTEX_H