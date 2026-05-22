#ifndef DATA_PARSER_H
#define DATA_PARSER_H

#include "types.hpp"
#include "Edge.h"
#include "Vertex.h"

namespace optimizer {

    class DataParser {
    public:
        // Parses a line from a file that describes a Vertex
        static optimizer::Vertex parseVertexLine(const std::string& line);

        // Parses a line from a file that describes an Edge
        static optimizer::Edge parseEdgeLine(int id, const std::string& line);

        static std::tuple<int, int, Mat6x4d, Mat6x4d> parseJacobianLine(const std::string& line);

        // Parses a string to form a 3x3 matrix
        static Mat3d readMatrix3d(const std::string& input);

        // Parses a string to form a 3x1 vector
        static Vec3d readVector3d(const std::string& input);

        static Mat6x4d readJacobian6x4(const std::string& input);

        // Helper function that parses a string into a vector of doubles
        static std::vector<double> parseStringToDoubles(const std::string& input);

    };

} // namespace optimizer

#endif  // DATA_PARSER_H