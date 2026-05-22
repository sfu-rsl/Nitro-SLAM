cmake_minimum_required(VERSION 3.4.1)
project(compute_engine VERSION 0.1.0)
add_compile_definitions(SPDLOG_ACTIVE_LEVEL=5)
add_compile_definitions(KOMPUTE_VK_API_MAJOR_VERSION=1 KOMPUTE_VK_API_MINOR_VERSION=2)
set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# add_compile_options(-fsanitize=thread)
# add_link_options(-fsanitize=thread)
include(CheckLanguage)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

FetchContent_Declare(
  googlebench
  URL https://github.com/google/benchmark/archive/refs/tags/v1.6.1.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googlebench)

include(GoogleTest)

# Kompute
set(KOMPUTE_OPT_LOG_LEVEL "Off" CACHE STRING "Kompute logging level" FORCE)
set(KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK ON)
set(KOMPUTE_OPT_RELAX ON)
set(KOMPUTE_OPT_USE_BUILT_IN_GOOGLE_TEST OFF)
set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS ON)

# General
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries." FORCE)

# It is necessary to pass the DEBUG or RELEASE flag accordingly to Kompute
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG=1 ${KOMPUTE_EXTRA_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DRELEASE=1 ${KOMPUTE_EXTRA_CXX_FLAGS}")

add_subdirectory(external/kompute)

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Threads REQUIRED)

# library
add_library(compute SHARED
  src/backend/vulkan/compute_engine.cpp
  src/CustomOp.cpp
)

# compile shaders
find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE SHADERS
  "shaders/*.comp"
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data-pgo/
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data-pgo/)

add_custom_target(compile_shaders
  COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/src/include/solver/shaders
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/external/kompute/scripts/convert_shaders_custom.py -p shaders -s glslangValidator -c ${CMAKE_CURRENT_SOURCE_DIR}/src/include/solver/shaders/
  DEPENDS ${SHADERS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  message("CUDA IS AVAILABLE")

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(compute CUDA::cusparse CUDA::cusolver CUDA::cudart)
  add_compile_definitions(CUDA_ENABLED)
else()
  message("CUDA IS NOT AVAILABLE")
endif()

add_dependencies(compute
  compile_shaders
)

target_link_libraries(compute
  kompute::kompute
  Threads::Threads
)

target_include_directories(compute PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  PUBLIC external/kompute/src/include
  PUBLIC src/include
)

# Benchmarks
add_executable(sbm_bench
  src/sbm_bench.cpp
)

target_link_libraries(sbm_bench
  kompute::kompute
  benchmark::benchmark
  compute
)

enable_testing()

# Tests
add_executable(
  tests
  src/tests.cpp
)
target_link_libraries(tests
  gtest_main
  compute
)

gtest_discover_tests(tests PROPERTIES DISCOVERY_TIMEOUT 600)
