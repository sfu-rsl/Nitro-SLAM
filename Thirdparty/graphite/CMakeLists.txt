cmake_minimum_required(VERSION 3.21)
project(Graphite LANGUAGES CUDA CXX VERSION 0.4.0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES EQUAL 52) # Newer versions default to 52 which is incompatible
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 120)
  endif()
  message(STATUS "No CMAKE_CUDA_ARCHITECTURES specified, defaulting to: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CMAKE_CUDA_ARCHITECTURES specified as: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Check WSL2
message(STATUS "CMake ${CMAKE_VERSION}  kernel: ${CMAKE_HOST_SYSTEM_VERSION}")
if(CMAKE_HOST_SYSTEM_VERSION MATCHES "microsoft-standard-WSL2$")
  add_compile_definitions(__WSL2__)
endif()

# Libraries
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(cudss 0.7.0 REQUIRED)

# Setup graphite library
add_library(graphite src/stream.cpp src/eigen_solver.cpp)
target_include_directories(graphite PUBLIC include)
target_link_libraries(graphite PUBLIC CUDA::cublas Eigen3::Eigen cudss)


# Examples
option(BUILD_EXAMPLES "Build example applications" OFF)

if(BUILD_EXAMPLES)
  message(STATUS "Configured to build example applications")

  add_executable(circle examples/circle.cu)
  target_link_libraries(circle graphite)


  add_executable(bal examples/bal.cu)
  target_link_libraries(bal graphite)
endif()

option(BUILD_DOCS "Build documentation" ON)

if (BUILD_DOCS)
  find_package(Doxygen)
  if (DOXYGEN_FOUND)

    FetchContent_Declare(
        doxygen-awesome-css
        URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/heads/main.zip
    )
    FetchContent_MakeAvailable(doxygen-awesome-css)

    FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)


    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
  else()
    message(WARNING "Doxygen not found! 'docs' target will not be available.")
  endif()

endif()

option(BUILD_TESTS "Build unit tests" OFF)

if (BUILD_TESTS)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(
    tests
    tests/dual.cu
    tests/factor.cu
    tests/vector.cu
    tests/vertex.cu
  )

  target_link_libraries(
    tests
    graphite
    GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(tests)

endif()



# Formatter
find_program(CLANG_FORMAT_BIN NAMES clang-format)
if(CLANG_FORMAT_BIN)
  file(GLOB_RECURSE FORMAT_FILES
    "include/graphite/*.cpp" "include/graphite/*.hpp" "include/graphite/*.c" "include/graphite/*.h" "include/graphite/*.cu" "include/graphite/*.cuh"
    "examples/*.cpp" "examples/*.hpp" "examples/*.c" "examples/*.h" "examples/*.cu" "examples/*.cuh" "tests/*.cu"
  )
  list(FILTER FORMAT_FILES EXCLUDE REGEX "examples/argparse/*")
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_BIN}
      -i
      --style=file
      ${FORMAT_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting source files"
  )
else()
  message(WARNING "clang-format not found! 'format' target will not be available.")
endif()
