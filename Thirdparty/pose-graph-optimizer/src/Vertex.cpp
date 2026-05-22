#include "../include/Vertex.h"
#include <iostream>

namespace optimizer {

Vertex::Vertex(int id, Mat3d& _Rcw, Mat3d& _Rcb, Mat3d& _Rbc, Mat3d& _Rwb, Vec3d& _tcw, Vec3d& _tcb, Vec3d& _tbc, Vec3d& _twb, double _bf)
    : id(id), Rcw(_Rcw), Rcb(_Rcb), Rbc(_Rbc), Rwb(_Rwb), tcw(_tcw), tcb(_tcb), tbc(_tbc), twb(_twb), bf(_bf)
{
    DR.setIdentity();
    Rwb0 = Rwb;
    isFixed = false;
}

int Vertex::getID() const {
    return id;
}

Mat3d Vertex::getRcw() const {
    return Rcw;
}

Mat3d Vertex::getRcb() const {
    return Rcb;
}

Mat3d Vertex::getRbc() const {
    return Rbc;
}

Mat3d Vertex::getRwb() const {
    return Rwb;
}

Vec3d Vertex::gettcw() const {
    return tcw;
}

Vec3d Vertex::gettcb() const {
    return tcb;
}

Vec3d Vertex::gettbc() const {
    return tbc;
}

Vec3d Vertex::gettwb() const {
    return twb;
}

Mat3d Vertex::getDR() const {
    return DR;
}

bool Vertex::getIsFixed() const {
    return isFixed;
}

void Vertex::setFixed(bool fixed) {
    isFixed = fixed;
}

} // namespace optimizer