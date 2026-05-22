#include "GraphInterface.h"
#include "Graph.h"

namespace optimizer {

    std::unique_ptr<GraphInterface> GraphInterface::create() {
        return std::make_unique<Graph>();
    }
}