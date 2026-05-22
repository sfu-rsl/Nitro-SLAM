#include "include/Edge.h"
#include "include/Graph.h"
#include "include/DataParser.h"
#include "include/GraphInterface.h"
#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <unistd.h>
#include <iostream>
#include <array>
#include <fstream>
#include <string>
#include <iomanip>

/*

void dumpJacobianResults(optimizer::Graph& graph, const std::string& filename) 
{
    std::fstream file;
    file.open(filename, std::ios::out);
    if(!file.is_open()) {
        std::cout << "Could not open file: " << filename << std::endl;
        return;
    }

    auto jacobianXHost = graph.getjacobianBufferX().template get_access<sycl::access::mode::read>();
    auto jacobianYHost = graph.getjacobianBufferY().template get_access<sycl::access::mode::read>();
    auto vertexXMapHost = graph.getVertexXMapBuffer().template get_access<sycl::access::mode::read>();
    auto vertexYMapHost = graph.getVertexYMapBuffer().template get_access<sycl::access::mode::read>();

    std::size_t numElements = jacobianXHost.get_count(); // Assuming both buffers are of the same size
    for(std::size_t i = 0; i < numElements; i += 24){
        file << graph.getVertexIDReverseMap()[vertexXMapHost[i/24]] << " " << graph.getVertexIDReverseMap()[vertexYMapHost[i/24]] << ": ";
        for(std::size_t j = i; j < i + 24 && j < numElements; ++j){
            file << std::setprecision(15) << jacobianXHost[j] << " ";
        }
        file << ", ";
        for(std::size_t j = i; j < i + 24 && j < numElements; ++j){
            file << std::setprecision(15) <<  jacobianYHost[j] << " ";
        }
        
        file << std::endl; // A new line after a block of 24 from each buffer
    }

    file.close();
}

*/


void dumpJacobianResults(optimizer::Graph& graph, const std::string& filename) 
{
    std::fstream file;
    file.open(filename, std::ios::out);
    if(!file.is_open()) {
        std::cout << "Could not open file: " << filename << std::endl;
        return;
    }

    auto jacobianXHost = graph.getjacobianBufferX().template get_access<sycl::access::mode::read>();
    auto jacobianYHost = graph.getjacobianBufferY().template get_access<sycl::access::mode::read>();
    auto vertexXMapHost = graph.getVertexXMapBuffer().template get_access<sycl::access::mode::read>();
    auto vertexYMapHost = graph.getVertexYMapBuffer().template get_access<sycl::access::mode::read>();

    std::size_t numElements = jacobianXHost.get_count(); // Assuming both buffers are of the same size
    for(std::size_t i = 0; i < numElements; i += 15){
        file << graph.getVertexIDReverseMap()[vertexXMapHost[i/15]] << " " << graph.getVertexIDReverseMap()[vertexYMapHost[i/15]] << ": ";
        
        // For the first dimension, print both rotational and translational values
        for(std::size_t j = i; j < i + 6 && j < numElements; ++j){
            file << std::setprecision(15) << jacobianXHost[j] << " ";
        }

        // For the next three dimensions, print 0's for rotational values and then the translational values
        for(int dim = 1; dim < 4; ++dim) {
            // Print 0's for rotational values
            for(int zeroCount = 0; zeroCount < 3; ++zeroCount) {
                file << "0 ";
            }
            // Print the translational values
            for(std::size_t j = i + 6 + (dim - 1) * 3; j < i + 6 + dim * 3 && j < numElements; ++j){
                file << std::setprecision(15) << jacobianXHost[j] << " ";
            }
        }
        
        file << ", ";
        
        // Repeat the same for jacobianYHost
        for(std::size_t j = i; j < i + 6 && j < numElements; ++j){
            file << std::setprecision(15) << jacobianYHost[j] << " ";
        }
        for(int dim = 1; dim < 4; ++dim) {
            for(int zeroCount = 0; zeroCount < 3; ++zeroCount) {
                file << "0 ";
            }
            for(std::size_t j = i + 6 + (dim - 1) * 3; j < i + 6 + dim * 3 && j < numElements; ++j){
                file << std::setprecision(15) << jacobianYHost[j] << " ";
            }
        }
        
        file << std::endl; // A new line after a block of 15 from each buffer
    }

    file.close();
}


void compareJacobians(const std::map<std::pair<int, int>, std::pair<Mat6x4d, Mat6x4d>>& map1, 
                      const std::map<std::pair<int, int>, std::pair<Mat6x4d, Mat6x4d>>& map2, 
                      const std::string& filename, double eps = 1e-4) 
{
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (const auto& [vertex_pair, jacobians] : map1) {
        auto found = map2.find(vertex_pair);
        if (found != map2.end()) {
            double frobenius_norm_x = (jacobians.first - found->second.first).norm();
            double frobenius_norm_y = (jacobians.second - found->second.second).norm();

            if(vertex_pair.first == 662 || vertex_pair.second == 662) continue;
            // if(vertex_pair.first == 385 || vertex_pair.second == 385) continue;
            if (frobenius_norm_x > 5 || frobenius_norm_y > 5) {
                std::cout << "Vertex pair: " << vertex_pair.first << ", " << vertex_pair.second
                          << ", " << frobenius_norm_x
                          << ", " << frobenius_norm_y << std::endl;
            }
            file << vertex_pair.first << ", " << vertex_pair.second
                        << ", " << frobenius_norm_x
                        << ", " << frobenius_norm_y << std::endl;
        }
    }
}

int main() {

    try {
    sycl::property_list propList {sycl::property::queue::enable_profiling() };

    // Create a SYCL queue.
    // sycl::queue q;
    // try {
    // sycl::device gpuDevice = sycl::device(sycl::gpu_selector{});
    // q = sycl::queue(gpuDevice, propList);
    // } catch (sycl::runtime_error &e) {
    // std::cout << "No GPU device found, falling back to CPU.\n";
    // sycl::device cpuDevice = sycl::device(sycl::cpu_selector{});
    // q = sycl::queue(cpuDevice);
    // }

    // std::cout << "Running on "
    //     << q.get_device().get_info<sycl::info::device::name>()
    //     << "\n";

    int iteration = 5;
    std::string dataset = "outdoors";
    std::string inputData = "../Data/" + dataset + "/opt_" + std::to_string(iteration) + ".txt";
    std::string jacobianInput = "../Data/" + dataset + "/jacobians_" + std::to_string(iteration) + ".txt";
    // std::string jacobianOutput = "./../Data/" + dataset + "/JacobianResults_" + std::to_string(iteration) + ".txt";
    std::string jacobianOutput = "./../JacobianResultsNew_" + std::to_string(iteration) + ".txt";
    // std::string jacobianDifference = "./../Data/" + dataset + "/JacobianDifferenceNormFixed_" + std::to_string(iteration) + ".txt";
    std::string jacobianDifference = "./../JacobianDifferenceNormFixedNew_" + std::to_string(iteration) + ".txt";

    std::ifstream dataFile(inputData);

    if(!dataFile.is_open()) {
        std::cout << "Could not open dataFile" << std::endl;
        return 1;
    }

    int num_edges = 0;
    int num_vertices = 0;

    std::vector<optimizer::Vertex> vertices;
    std::vector<optimizer::Edge> edges;

    std::string line;
    while(std::getline(dataFile, line)) {
        if (line[0] == 'V') { // This line represents a vertex.
            optimizer::Vertex v = optimizer::DataParser::parseVertexLine(line);
            vertices.push_back(v);
            num_vertices++;
            
            // Peek at the next line; if it's a FIX for this vertex, mark the vertex as fixed.
            if (dataFile.peek() != EOF) {
                std::streampos oldPos = dataFile.tellg();  // Remember our position in the file.
                std::getline(dataFile, line);
                if (line.substr(0, 3) == "FIX" && std::stoi(line.substr(4)) == v.getID()) {
                    vertices.back().setFixed(true);  // Fix this vertex.
                } else {
                    // It's not a FIX line; go back to where we were so we don't skip a line.
                    dataFile.seekg(oldPos);
                }
            }
        } else if (line[0] == 'E') { // This line represents an edge.
            optimizer::Edge e = optimizer::DataParser::parseEdgeLine(num_edges, line);
            edges.push_back(e);
            num_edges++;
        }
    }
    std::cout << "Number of edges: " << num_edges << std::endl;
    std::cout << "Number of vertices: " << num_vertices << std::endl;


    // Create a graph with the edges
    optimizer::Graph graph;

    for(int i = 0; i < num_vertices; ++i){
        graph.addVertex(vertices[i]);
        if(vertices[i].getIsFixed()){
            std::cout << "Vertex " << vertices[i].getID() << " is fixed" << std::endl;
        }
    }
    for (int i = 0; i < num_edges; ++i) {
        graph.addEdge(edges[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // // Update the buffers
    graph.initializeBuffers();

    auto mid = std::chrono::high_resolution_clock::now();

    graph.computeJacobians();

    auto end = std::chrono::high_resolution_clock::now();
    auto dataTransfer = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto compute = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);

    std::cout << "Time taken: " << dataTransfer.count() << " and " << compute.count() << " microseconds" << std::endl;

    dumpJacobianResults(graph, jacobianOutput);

    std::ifstream ORB_SLAM3JacobianData(jacobianInput), jacobianData(jacobianOutput);

    std::map<std::pair<int, int>, std::pair<Mat6x4d, Mat6x4d>> ORB_SLAM3JacobianMap;
    std::map<std::pair<int, int>, std::pair<Mat6x4d, Mat6x4d>> jacobianMap;

    while(getline(ORB_SLAM3JacobianData, line)) {
        auto [vxID, vyID, jacobianX, jacobianY] = optimizer::DataParser::parseJacobianLine(line);
        ORB_SLAM3JacobianMap[std::make_pair(vxID, vyID)] = std::make_pair(jacobianX, jacobianY);
    }

    while(getline(jacobianData, line)) {
        auto [vxID, vyID, jacobianX, jacobianY] = optimizer::DataParser::parseJacobianLine(line);
        jacobianMap[std::make_pair(vxID, vyID)] = std::make_pair(jacobianX, jacobianY);
    }

    std::cout << "ORB_SLAM3 Jacobian Map size: " << ORB_SLAM3JacobianMap.size() << std::endl;
    std::cout << "Jacobian Map size: " << jacobianMap.size() << std::endl;

    compareJacobians(ORB_SLAM3JacobianMap, jacobianMap, jacobianDifference);

    } catch (sycl::exception& e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}