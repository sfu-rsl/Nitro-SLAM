#include "../include/DataParser.h"
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

using Mat3d = Eigen::Matrix3d;
using Vec3d = Eigen::Vector3d;

namespace optimizer{


        // Expected Format: V id:Rcw,Rcb,Rbc,tcw,tcb,tbc,bf
        Vertex DataParser::parseVertexLine(const std::string& line) {
            std::stringstream ss(line);
            std::string id_str, data_str;
            std::getline(ss, id_str, ':'); // split the string at ':'
            std::getline(ss, data_str, ':');

            int id = std::stoi(id_str.substr(1)); // get the vertex id
            std::vector<std::string> data_items;
            std::stringstream data_ss(data_str);

            while(std::getline(data_ss, data_str, ',')) // split the data string at ','
                data_items.push_back(data_str);

            // parse the matrices and vectors here using data_items
            Mat3d Rcw = readMatrix3d(data_items[0]);
            Mat3d Rcb = readMatrix3d(data_items[1]);
            Mat3d Rbc = readMatrix3d(data_items[2]);
            Mat3d Rwb = readMatrix3d(data_items[3]);
            Vec3d tcw = readVector3d(data_items[4]);
            Vec3d tcb = readVector3d(data_items[5]);
            Vec3d tbc = readVector3d(data_items[6]);
            Vec3d twb = readVector3d(data_items[7]);
            double bf = std::stod(data_items[8]);

            return Vertex(id, Rcw, Rcb, Rbc, Rwb, tcw, tcb, tbc, twb, bf);
        }

        // Expected Format: E id1,id2:dRij,dtij
        Edge DataParser::parseEdgeLine(int id, const std::string& line) {
            std::stringstream ss(line);
            std::string id_str, data_str;
            std::getline(ss, id_str, ':'); // split the string at ':'
            std::getline(ss, data_str, ':');

            std::stringstream id_ss(id_str); // create a new stringstream for id_str
            int vertexXid, vertexYid;
            char E;
            id_ss >> E >> vertexXid >> vertexYid; // read the values

            std::vector<std::string> data_items;
            std::stringstream data_ss(data_str);

            while(std::getline(data_ss, data_str, ',')) // split the data string at ','
                data_items.push_back(data_str);

            // parse the rotation and translation matrices here using data_items
            Mat3d dRij = readMatrix3d(data_items[0]);
            Vec3d dtij = readVector3d(data_items[1]);

            return Edge(id, vertexXid, vertexYid, dRij, dtij);
        }

        // Expected Format: id1,id2: jacobianX, jacobianY
        std::tuple<int, int, Mat6x4d, Mat6x4d> DataParser::parseJacobianLine(const std::string& line) {
            std::stringstream ss(line);
            std::string id_str, jacobianX_str, jacobianY_str;
            
            std::getline(ss, id_str, ':'); // split the string at ':'
            std::getline(ss, jacobianX_str, ','); // split the string at ','
            std::getline(ss, jacobianY_str);

            std::stringstream id_ss(id_str); // create a new stringstream for id_str
            int vertexXid, vertexYid;
            id_ss >> vertexXid >> vertexYid; // read the values

            Mat6x4d jacobianX = readJacobian6x4(jacobianX_str);
            Mat6x4d jacobianY = readJacobian6x4(jacobianY_str);

            return std::make_tuple(vertexXid, vertexYid, jacobianX, jacobianY);
        }


        Mat3d DataParser::readMatrix3d(const std::string& input) {
            std::vector<double> elements = parseStringToDoubles(input);
            Mat3d matrix;
            for(int i = 0; i < 3; ++i) {
                for(int j = 0; j < 3; ++j)
                    matrix(i, j) = elements[3*i+j];
            }
            return matrix;
        }

        Vec3d DataParser::readVector3d(const std::string& input) {
            std::vector<double> elements = parseStringToDoubles(input);
            Vec3d vector;
            for(int i = 0; i < 3; ++i) {
                vector[i] = elements[i];
            }
            return vector;
        }

        Mat6x4d DataParser::readJacobian6x4(const std::string& input) {
            std::vector<double> elements = parseStringToDoubles(input);
            Mat6x4d matrix;
            for(int j = 0; j < 4; ++j) { // Loop over columns first
                for(int i = 0; i < 6; ++i) { // Then loop over rows
                    matrix(i, j) = elements[6*j+i]; // Access elements in column-major order
                }
            }
            return matrix;
        }

        std::vector<double> DataParser::parseStringToDoubles(const std::string& input) {
            std::vector<double> output;
            std::stringstream ss(input);

            double value;
            while (ss >> value) {
                output.push_back(value);
            }

            return output;
        }
} // namespace optimizer
