#ifndef OPTIMIZER_EDGE_H
#define OPTIMIZER_EDGE_H

#include "types.hpp"

namespace optimizer {

class Edge {
public:
    Edge(int id, int vertexXid, int vertexYid, const Mat3d& rotation, const Vec3d& translation);

    Mat3d getRotation() const;
    Vec3d getTranslation() const;

    int getVertexXid() const;

    int getVertexYid() const;

    int getID() const;

private:
    Mat3d rotation;
    Vec3d translation;
    int vertexXid;
    int vertexYid;
    int id;
};

} // namespace optimizer

#endif  // OPTIMIZER_EDGE_H
