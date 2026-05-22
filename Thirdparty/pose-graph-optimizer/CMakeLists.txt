cmake_minimum_required(VERSION 3.10)
project(optimizer)

set(CMAKE_CXX_COMPILER "syclcc")
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -I /usr/local/include/eigen3/ --hipsycl-targets=cuda:sm_86")

set(SOURCES
    src/Edge.cpp
    src/Vertex.cpp
    src/Kernel.cpp
    src/Graph.cpp
    src/DataParser.cpp
    src/GraphInterface.cpp
)

add_library(optimizer SHARED ${SOURCES})

target_include_directories(optimizer PUBLIC include)

# add_executable(main main.cpp ${SOURCES})
