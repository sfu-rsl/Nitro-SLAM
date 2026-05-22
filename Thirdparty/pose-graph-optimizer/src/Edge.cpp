#include "../include/Edge.h"

namespace optimizer {

Edge::Edge(int id, int vertexXid, int vertexYid, const Mat3d& rotation, const Vec3d& translation)
    : id(id), vertexXid(vertexXid), vertexYid(vertexYid), rotation(rotation), translation(translation)
{

}

Mat3d Edge::getRotation() const {
    return rotation;
}

Vec3d Edge::getTranslation() const {
    return translation;
}

int Edge::getVertexXid() const {
    return vertexXid;
}

int Edge::getVertexYid() const {
    return vertexYid;
}

int Edge::getID() const {
    return id;
}

} // namespace optimizer

