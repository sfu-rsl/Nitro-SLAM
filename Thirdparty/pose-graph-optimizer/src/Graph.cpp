#include "../include/types.hpp"
#include "../include/Graph.h"
#include "../include/Kernel.h"
#include <sycl/sycl.hpp>
#include <fstream>


namespace optimizer {

Graph::Graph() : GraphInterface(),
        queue(),
        vertexIDMap(std::map<int, int>()),
        vertexIDReverseMap(std::map<int, int>()),
        edgeIDMap(std::map<int, int>()),
        edgeIDReverseMap(std::map<int, int>()),
        currentVertexIndex(0),
        currentEdgeIndex(0),
        numEdges(0),
        numVertices(0)

{
    std::cout << "Running on "
        << queue.get_device().get_info<sycl::info::device::name>()
        << "\n";
}

void Graph::addEdge(const Edge& edge) {
    rotationMatrices.push_back(edge.getRotation());
    translationVectors.push_back(edge.getTranslation());

    vertexXMap.push_back(vertexIDMap[edge.getVertexXid()]);
    vertexYMap.push_back(vertexIDMap[edge.getVertexYid()]);
    edgeIDMap[edge.getID()] = currentEdgeIndex;
    edgeIDReverseMap[currentEdgeIndex] = edge.getID();
    currentEdgeIndex++;
    numEdges++;
}

void Graph::addVertex(const Vertex& vertex) {
    Rcw.push_back(vertex.getRcw());
    Rcb.push_back(vertex.getRcb());
    Rbc.push_back(vertex.getRbc());
    Rwb.push_back(vertex.getRwb());
    Rwb0.push_back(vertex.getRwb());
    tcw.push_back(vertex.gettcw());
    tcb.push_back(vertex.gettcb());
    tbc.push_back(vertex.gettbc());
    twb.push_back(vertex.gettwb());
    fixedVertices.push_back(vertex.getIsFixed());
    vertexIDMap[vertex.getID()] = currentVertexIndex;
    vertexIDReverseMap[currentVertexIndex] = vertex.getID();
    currentVertexIndex++;
    numVertices++;
}

// Copy data to buffers
void Graph::initializeBuffers() {

    rotationBuffer = std::make_unique<sycl::buffer<Mat3d, 1>>(rotationMatrices.data(), sycl::range<1>(numEdges));
    translationBuffer = std::make_unique<sycl::buffer<Vec3d, 1>>(translationVectors.data(), sycl::range<1>(numEdges));

    RcwBuffer = std::make_unique<sycl::buffer<Mat3d, 1>>(Rcw.data(), sycl::range<1>(numVertices));
    RcbBuffer = std::make_unique<sycl::buffer<Mat3d, 1>>(Rcb.data(), sycl::range<1>(numVertices));
    RbcBuffer = std::make_unique<sycl::buffer<Mat3d, 1>>(Rbc.data(), sycl::range<1>(numVertices));
    RwbBuffer = std::make_unique<sycl::buffer<Mat3d, 1>>(Rwb.data(), sycl::range<1>(numVertices));
    tcwBuffer = std::make_unique<sycl::buffer<Vec3d, 1>>(tcw.data(), sycl::range<1>(numVertices));
    tcbBuffer = std::make_unique<sycl::buffer<Vec3d, 1>>(tcb.data(), sycl::range<1>(numVertices));
    tbcBuffer = std::make_unique<sycl::buffer<Vec3d, 1>>(tbc.data(), sycl::range<1>(numVertices));
    twbBuffer = std::make_unique<sycl::buffer<Vec3d, 1>>(twb.data(), sycl::range<1>(numVertices));
    Rwb0Buffer = std::make_unique<sycl::buffer<Mat3d, 1>>(Rwb0.data(), sycl::range<1>(numVertices));
    RcwUpdatePositiveBuffer = std::make_unique<sycl::buffer<Mat3d, 1>>(sycl::range<1>(numVertices));
    tcwUpdatePositiveBuffer = std::make_unique<sycl::buffer<Vec3d, 1>>(sycl::range<1>(numVertices));
    RcwUpdateNegativeBuffer = std::make_unique<sycl::buffer<Mat3d, 1>>(sycl::range<1>(numVertices));
    tcwUpdateNegativeBuffer = std::make_unique<sycl::buffer<Vec3d, 1>>(sycl::range<1>(numVertices));
    fixedVerticesBuffer = std::make_unique<sycl::buffer<int, 1>>(fixedVertices.data(), sycl::range<1>(numVertices));

    vertexXMapBuffer = std::make_unique<sycl::buffer<int, 1>>(vertexXMap.data(), sycl::range<1>(numEdges));
    vertexYMapBuffer = std::make_unique<sycl::buffer<int, 1>>(vertexYMap.data(), sycl::range<1>(numEdges));

    // jacobianBufferX = std::make_unique<sycl::buffer<double, 1>>(sycl::range<1>(numEdges*6*4));
    // jacobianBufferY = std::make_unique<sycl::buffer<double, 1>>(sycl::range<1>(numEdges*6*4));

     // 6 for first dimension and then 3 for each of the 3 dimensions
    jacobianBufferX = std::make_unique<sycl::buffer<double, 1>>(sycl::range<1>(numEdges*15));
    jacobianBufferY = std::make_unique<sycl::buffer<double, 1>>(sycl::range<1>(numEdges*15));

    queue.submit([&](sycl::handler& cgh) {
        auto RcwUpdatePosAcc = RcwUpdatePositiveBuffer->get_access<sycl::access::mode::discard_write>(cgh);
        auto tcwUpdatePosAcc = tcwUpdatePositiveBuffer->get_access<sycl::access::mode::discard_write>(cgh);
        auto RcwUpdateNegAcc = RcwUpdateNegativeBuffer->get_access<sycl::access::mode::discard_write>(cgh);
        auto tcwUpdateNegAcc = tcwUpdateNegativeBuffer->get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<class InitializeBuffers>(sycl::range<1>(numVertices), [=](sycl::id<1> i) {
            RcwUpdatePosAcc[i] = Mat3d::Zero();
            tcwUpdatePosAcc[i] = Vec3d::Zero();
            RcwUpdateNegAcc[i] = Mat3d::Zero();
            tcwUpdateNegAcc[i] = Vec3d::Zero();
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto accX = jacobianBufferX->get_access<sycl::access::mode::discard_write>(cgh);
        auto accY = jacobianBufferY->get_access<sycl::access::mode::discard_write>(cgh);
        // cgh.parallel_for<class InitializeBuffers>(sycl::range<1>(numEdges*6*4), [=](sycl::id<1> i) {
        cgh.parallel_for<class InitializeBuffers>(sycl::range<1>(numEdges*15), [=](sycl::id<1> i) {
            accX[i] = 0.0;
            accY[i] = 0.0;
        });
    });

    queue.wait();
}


void Graph::ExpSO3Test(){
    Kernel::ExpSO3Test(queue, *rotationBuffer , *translationBuffer, numEdges);
}

void Graph::LogSO3Test(){
    Kernel::LogSO3Test(queue, *rotationBuffer, numEdges);
}

void Graph::update(int dimension){
    // Kernel::update(queue, dimension, isPositive, *Rwb0Buffer, *twbBuffer, *RcwBuffer, *RcbBuffer, *tcwBuffer, *tcbBuffer, *RcwUpdatePositiveBuffer, *tcwUpdatePositiveBuffer, *RcwUpdateNegativeBuffer, *tcwUpdateNegativeBuffer, *vertexXMapBuffer, *vertexYMapBuffer, numEdges, numVertices);
    Kernel::update(queue, dimension, *Rwb0Buffer, *twbBuffer, *RcwBuffer, *RcbBuffer, *tcwBuffer, *tcbBuffer, *RcwUpdatePositiveBuffer, *tcwUpdatePositiveBuffer, *RcwUpdateNegativeBuffer, *tcwUpdateNegativeBuffer, numEdges, numVertices);
}

void Graph::update(double* values){
    Kernel::update(queue, values, *Rwb0Buffer, *twbBuffer, *RcwBuffer, *RcbBuffer, *tcwBuffer, *tcbBuffer, numEdges, numVertices);
}

void Graph::computeError(int dimension){
    // Kernel::computeError(queue, dimension, isPositive, *translationBuffer, *rotationBuffer, *RcwBuffer, *tcwBuffer, *RcwUpdatePositiveBuffer, *tcwUpdatePositiveBuffer, *vertexXMapBuffer, *vertexYMapBuffer, *jacobianBufferX, *jacobianBufferY, *fixedVerticesBuffer, numEdges);
    Kernel::computeError(queue, dimension, *translationBuffer, *rotationBuffer, *RcwBuffer, *tcwBuffer, *RcwUpdatePositiveBuffer, *tcwUpdatePositiveBuffer, *RcwUpdateNegativeBuffer, *tcwUpdateNegativeBuffer, *vertexXMapBuffer, *vertexYMapBuffer, *jacobianBufferX, *jacobianBufferY, *fixedVerticesBuffer, numEdges);
}

sycl::buffer<double, 1>& Graph::getjacobianBufferX(){
    return *jacobianBufferX;
}

sycl::buffer<double, 1>& Graph::getjacobianBufferY(){
    return *jacobianBufferY;
}

sycl::buffer<int, 1>& Graph::getVertexXMapBuffer(){
    return *vertexXMapBuffer;
}

sycl::buffer<int, 1>& Graph::getVertexYMapBuffer(){
    return *vertexYMapBuffer;
}

std::map<int, int>& Graph::getVertexIDMap(){
    return vertexIDMap;
}

std::map<int, int>& Graph::getVertexIDReverseMap(){
    return vertexIDReverseMap;
}

std::map<int, int>& Graph::getEdgeIDMap(){
    return edgeIDMap;
}

std::map<int, int>& Graph::getEdgeIDReverseMap(){
    return edgeIDReverseMap;
}

int Graph::getNumEdges(){
    return numEdges;
}

int Graph::getNumVertices(){
    return numVertices;
}

void Graph::getJacobianX(std::vector<double>& hostData){
    queue.submit([&](sycl::handler& cgh) {
        auto hostAccess = jacobianBufferX->get_access<sycl::access::mode::read>(cgh);
        cgh.copy(hostAccess, hostData.data());
    }).wait();
}

void Graph::getJacobianY(std::vector<double>& hostData){
    queue.submit([&](sycl::handler& cgh) {
        auto hostAccess = jacobianBufferY->get_access<sycl::access::mode::read>(cgh);
        cgh.copy(hostAccess, hostData.data());
    }).wait();
}

void Graph::computeJacobians(){
    for(int i = 0; i < 4; i++){
        update(i);
        computeError(i);
    }
    queue.wait();
}

void Graph::jacobianTransferTime(std::vector<double>& hostData){
    auto start = std::chrono::high_resolution_clock::now();
    {
        queue.submit([&](sycl::handler& cgh) {
            auto hostAccess = jacobianBufferX->get_access<sycl::access::mode::read>(cgh);
            cgh.copy(hostAccess, hostData.data());
        });
        queue.wait(); // Wait for all commands in the queue to finish
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Data transfer time: " << diff.count() << " seconds\n";
}

void Graph::printJacobianBuffers()
{
    auto jacobianXAcc = jacobianBufferX->get_access<sycl::access::mode::read>();
    auto jacobianYAcc = jacobianBufferY->get_access<sycl::access::mode::read>();

    for (int edgeIdx = 0; edgeIdx < numEdges; ++edgeIdx) {
        std::cout << "Edge " << edgeIdx << std::endl;

        std::cout << "Jacobian X: " << std::endl;
        for(int row = 0; row < 6; ++row) { // 6 rows
            for(int col = 0; col < 4; ++col) { // 4 columns
                int index = edgeIdx * 4 * 6 + col * 6 + row; // Convert 3D indices to 1D index
                std::cout << jacobianXAcc[index] << " | ";
            }
            std::cout << std::endl;
        }

        std::cout << "Jacobian Y: " << std::endl;
        for(int row = 0; row < 6; ++row) {
            for(int col = 0; col < 4; ++col) {
                int index = edgeIdx * 4 * 6 + col * 6 + row; // Convert 3D indices to 1D index
                std::cout << jacobianYAcc[index] << " | ";
            }
            std::cout << std::endl;
        }
    }
}


} // namespace optimizer
