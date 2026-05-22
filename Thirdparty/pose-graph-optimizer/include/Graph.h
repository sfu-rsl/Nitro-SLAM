#ifndef OPTIMIZER_GRAPH_H
#define OPTIMIZER_GRAPH_H

#include "GraphInterface.h"
#include <sycl/sycl.hpp>

namespace optimizer {

class Graph : public GraphInterface {

public:
    Graph();

    void addEdge(const Edge& edge) override;
    
    void addVertex(const Vertex& vertex) override;
    void initializeBuffers() override;
    void ExpSO3Test() override;
    void LogSO3Test() override;
    // void updateVertex(int dimension, bool isPositive);
    // void computeTranslationError(int dimension);
    // void computeRotationalError(int dimension);

    void update(int dimension) override;

    void update(double* values) override;
    void computeError(int dimension) override;

    void computeJacobians() override;

    void printJacobianBuffers() override;

    void jacobianTransferTime(std::vector<double>& hostData) override;

    // Getters
    sycl::buffer<double, 1>& getjacobianBufferX();
    sycl::buffer<double, 1>& getjacobianBufferY();

    sycl::buffer<int, 1>& getVertexXMapBuffer();
    sycl::buffer<int, 1>& getVertexYMapBuffer();

    std::map<int, int>& getVertexIDMap() override;
    std::map<int, int>& getVertexIDReverseMap() override;

    std::map<int, int>& getEdgeIDMap() override;
    std::map<int, int>& getEdgeIDReverseMap() override;

    void getJacobianX(std::vector<double>& hostData) override;
    void getJacobianY(std::vector<double>& hostData) override;

    int getNumEdges() override;
    int getNumVertices() override;

private:

    // Vertex ID to index mapping
    int currentVertexIndex;
    std::map<int, int> vertexIDMap;
    std::map<int, int> vertexIDReverseMap;

    // Edge ID to index mapping
    int currentEdgeIndex;
    std::map<int, int> edgeIDMap;
    std::map<int, int> edgeIDReverseMap;

    sycl::queue queue;
    size_t numEdges;
    size_t numVertices;

    // Edge data stored in column-major format
    std::vector<Mat3d> rotationMatrices;
    std::vector<Vec3d> translationVectors;

    // Edge data buffers
    std::unique_ptr<sycl::buffer<Mat3d, 1>> rotationBuffer;
    std::unique_ptr<sycl::buffer<Vec3d, 1>> translationBuffer;
    // sycl::buffer<Mat3d, 1> rotationBuffer;
    // sycl::buffer<Vec3d, 1> translationBuffer;

    // Vertex data stored in column-major format
    std::vector<Mat3d> Rcw, Rcb, Rbc, Rwb, Rwb0; // Rotation matrices
    std::vector<Vec3d> tcw, tcb, tbc, twb; // Translation vectors

    // Vertex data buffers
    std::unique_ptr<sycl::buffer<Mat3d, 1>> RcwBuffer, RcbBuffer, RbcBuffer, RwbBuffer, Rwb0Buffer;
    std::unique_ptr<sycl::buffer<Vec3d, 1>> tcwBuffer, tcbBuffer, tbcBuffer, twbBuffer;

    std::vector<Mat3d> RcwUpdate;
    std::vector<Vec3d> tcwUpdate;

    // intermediate Rcw and tcw update values
    std::unique_ptr<sycl::buffer<Mat3d, 1>> RcwUpdatePositiveBuffer, RcwUpdateNegativeBuffer;
    std::unique_ptr<sycl::buffer<Vec3d, 1>> tcwUpdatePositiveBuffer, tcwUpdateNegativeBuffer;
    

    std::vector<int> vertexXMap, vertexYMap;

    // Edge-vertex mapping
    std::unique_ptr<sycl::buffer<int, 1>> vertexXMapBuffer;
    std::unique_ptr<sycl::buffer<int, 1>> vertexYMapBuffer;
    // sycl::buffer<int, 1> vertexXMapBuffer;
    // sycl::buffer<int, 1> vertexYMapBuffer;

    
    // Jacobian data buffers
    std::unique_ptr<sycl::buffer<double, 1>> jacobianBufferX;
    std::unique_ptr<sycl::buffer<double, 1>> jacobianBufferY;
    // sycl::buffer<double, 1> jacobianBufferX, jacobianBufferY;

    std::vector<int> fixedVertices;

    std::unique_ptr<sycl::buffer<int, 1>> fixedVerticesBuffer;

};

} // namespace optimizer

#endif  // OPTIMIZER_GRAPH_H