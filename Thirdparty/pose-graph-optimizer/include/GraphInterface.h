#ifndef OPTIMIZER_GRAPH_INTERFACE_H
#define OPTIMIZER_GRAPH_INTERFACE_H

#include "types.hpp"
#include "Edge.h"
#include "Vertex.h"
#include <map>
#include <memory>
#include <vector>

namespace optimizer {

class GraphInterface {

public:
    virtual ~GraphInterface() = default;
    
    virtual void addEdge(const Edge& edge) = 0;
    
    virtual void addVertex(const Vertex& vertex) = 0;
    
    virtual void initializeBuffers() = 0;
    
    virtual void ExpSO3Test() = 0;
    
    virtual void LogSO3Test() = 0;

    virtual void update(int dimension) = 0;

    virtual void update(double* values) = 0;
    
    virtual void computeError(int dimension) = 0;

    virtual void computeJacobians() = 0;

    virtual void printJacobianBuffers() = 0;

    virtual void jacobianTransferTime(std::vector<double>& hostData) = 0;

    virtual std::map<int, int>& getVertexIDMap() = 0;
    
    virtual std::map<int, int>& getVertexIDReverseMap() = 0;

    virtual std::map<int, int>& getEdgeIDMap() = 0;

    virtual std::map<int, int>& getEdgeIDReverseMap() = 0;

    virtual void getJacobianX(std::vector<double>& hostData) = 0;

    virtual void getJacobianY(std::vector<double>& hostData) = 0;

    virtual int getNumEdges() = 0;

    virtual int getNumVertices() = 0;

    static std::unique_ptr<GraphInterface> create();
};

} // namespace optimizer

#endif  // OPTIMIZER_GRAPH_INTERFACE_H
