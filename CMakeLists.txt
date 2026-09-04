cmake_minimum_required(VERSION 3.5)
project(ORB_SLAM3 LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda --default-stream per-thread")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 120)
  set(CMAKE_CUDA_ARCHITECTURES native) # RTX 3000 Series -> 86, RTX 5080 -> 120
endif()

# Integrated Tegra (Jetson) GPUs report concurrentManagedAccess == 0, so
# cudaMemPrefetchAsync is unsupported there and fails with cudaErrorInvalidDevice.
# DEVICE_JETSON compiles those prefetch hints out. Override explicitly with
# -DDEVICE_JETSON=ON or -DDEVICE_JETSON=OFF; with no override we detect the build
# host and default to OFF (i.e. not a Jetson) when nothing identifies one.
if(DEFINED DEVICE_JETSON)
  message(STATUS "DEVICE_JETSON overridden: ${DEVICE_JETSON}")
else()
  set(DEVICE_JETSON OFF)
  if(EXISTS "/etc/nv_tegra_release")
    # Canonical JetPack/L4T marker.
    set(DEVICE_JETSON ON)
  elseif(EXISTS "/proc/device-tree/model")
    # file(STRINGS) rather than file(READ): device-tree nodes are NUL-terminated.
    file(STRINGS "/proc/device-tree/model" DEVICE_TREE_MODEL LIMIT_COUNT 1)
    if(DEVICE_TREE_MODEL MATCHES "Jetson|Tegra")
      set(DEVICE_JETSON ON)
    endif()
  endif()
  message(STATUS "DEVICE_JETSON auto-detected: ${DEVICE_JETSON}")
endif()

if(DEVICE_JETSON)
  add_definitions(-DDEVICE_JETSON)
endif()



IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

add_compile_options(-w)
# -fsanitize=address for debugging
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3  ${OpenMP_CXX_FLAGS} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

# Per-thread profiling instrumentation. These gate the Stats/ singletons that write
# the per-frame / per-iteration timing breakdowns under <strStatsFile>/{Tracking,
# LocalMapping,LoopClosing}/data/. REGISTER_STATS turns on all three at once.
# They are mutually exclusive with the upstream REGISTER_TIMES instrumentation:
# both declare identically-named time_point locals in the same scope.
option(REGISTER_STATS               "Enable all Nitro-SLAM profiling stats"   OFF)
option(REGISTER_TRACKING_STATS      "Enable FastTrack profiling stats"        OFF)
option(REGISTER_LOCAL_MAPPING_STATS "Enable TurboMap profiling stats"         OFF)
option(REGISTER_LOOP_CLOSING_STATS  "Enable FastLoop profiling stats"         OFF)
option(REGISTER_TIMES               "Enable upstream ORB-SLAM3 REGISTER_TIMES" OFF)

if(REGISTER_STATS)
  set(REGISTER_TRACKING_STATS ON)
  set(REGISTER_LOCAL_MAPPING_STATS ON)
  set(REGISTER_LOOP_CLOSING_STATS ON)
endif()

if(REGISTER_TIMES AND (REGISTER_TRACKING_STATS OR REGISTER_LOCAL_MAPPING_STATS OR REGISTER_LOOP_CLOSING_STATS))
  message(FATAL_ERROR "REGISTER_TIMES cannot be combined with the REGISTER_*_STATS options "
                      "(duplicate time_point declarations in LocalMapping/LoopClosing). Pick one.")
endif()

foreach(_stat REGISTER_TRACKING_STATS REGISTER_LOCAL_MAPPING_STATS REGISTER_LOOP_CLOSING_STATS REGISTER_TIMES)
  if(${_stat})
    add_compile_definitions(${_stat})
    message(STATUS "Profiling: ${_stat} enabled")
  endif()
endforeach()

add_compile_definitions(SPDLOG_ACTIVE_LEVEL=5)
add_compile_definitions(KOMPUTE_VK_API_MAJOR_VERSION=1 KOMPUTE_VK_API_MINOR_VERSION=2)
add_compile_definitions(KOMPUTE_DISABLE_VK_DEBUG_LAYERS)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenMP REQUIRED)
find_package(realsense2)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# new - don't need these anymore (used for Vulkan LBA)
# ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/compute-engine/src/include
# ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/compute-engine/external/kompute/src/include
# ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/compute-engine/external/kompute/external/VulkanMemoryAllocator/include
)

set(BUILD_EXAMPLES OFF)
set(BUILD_DOCS OFF)
set(BUILD_TESTS OFF)
add_subdirectory(Thirdparty/graphite)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/fast.cu
src/orientation.cu
src/resize.cu
src/gaussian_blur.cu
src/descriptor.cu
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Optimizer.cu
src/LocalBA.cu
src/LIBA.cu
src/FIBA.cu
src/PoseOptimizationFused.cu
src/PoseOptTiming.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/Stats/TrackingStats.cc
src/Stats/LocalMappingStats.cc
src/Stats/LoopClosingStats.cc
src/Kernels/CudaUtils.cu
src/Kernels/CudaWrappers/CudaMapPoint.cu
src/Kernels/CudaWrappers/CudaFrame.cu
src/Kernels/CudaWrappers/CudaKeyFrame.cu
src/Kernels/StereoMatchKernel.cu
src/Kernels/SearchLocalPointsKernel.cu
src/Kernels/PoseEstimationKernel.cu
src/Kernels/SearchForTriangulationKernel.cu
src/Kernels/TrackingKernelController.cu
src/Kernels/MappingKernelController.cu
src/Kernels/CudaKeyFrameAllocator.cu
src/Kernels/FuseKernel.cu
src/Kernels/CudaUtils.cu
src/Kernels/LoopClosingKernelController.cu
src/Kernels/CudaWrappers/CudaMapPoint.cu
src/Kernels/CudaWrappers/CudaKeyFrame.cu
src/Kernels/SearchByProjectionKernel.cu
src/Kernels/SearchAndFuseKernel.cu
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/fast.h
include/orientation.h
include/resize.h
include/gaussian_blur.h
include/descriptor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/Stats/StatsInterface.h
include/Stats/TrackingStats.h
include/Stats/LocalMappingStats.h
include/Stats/LoopClosingStats.h
include/Kernels/CudaUtils.h
include/Kernels/CudaWrappers/CudaMapPoint.h
include/Kernels/CudaWrappers/CudaKeyPoint.h
include/Kernels/CudaWrappers/CudaFrame.h
include/Kernels/CudaWrappers/CudaKeyFrame.h
include/Kernels/CudaWrappers/CudaCamera.h
include/Kernels/StereoMatchKernel.h
include/Kernels/SearchLocalPointsKernel.h
include/Kernels/PoseEstimationKernel.h
include/Kernels/SearchForTriangulationKernel.h
include/Kernels/KernelInterface.h
include/Kernels/FuseKernel.h
include/Kernels/TrackingKernelController.h
include/Kernels/MappingKernelController.h
include/Kernels/CudaKeyFrameAllocator.h
include/Kernels/CudaUtils.h
include/Kernels/LoopClosingKernelController.h
include/Kernels/CudaWrappers/CudaMapPoint.h
include/Kernels/CudaWrappers/CudaCamera.h
include/Kernels/CudaWrappers/CudaKeyFrame.h
include/Kernels/CudaWrappers/CudaKeyPoint.h
include/Kernels/SearchByProjectionKernel.h
include/Kernels/SearchAndFuseKernel.h
include/GPUPose.h
include/PoseOptTiming.h
include/FusedLM.h
include/GPUTypes.h
include/GPUImu.h
include/PGOTypes.h
include/PGOInterface.h
include/LIBAInterface.h
)

set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_compile_options(-w)

add_compile_definitions(SPDLOG_ACTIVE_LEVEL=5)
add_subdirectory(Thirdparty/g2o)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
message("CUDA IS AVAILABLE")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} CUDA::cusparse CUDA::cusolver CUDA::cudart)
add_compile_definitions(CUDA_ENABLED)
else()
message("CUDA IS NOT AVAILABLE")
endif()

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
OpenMP::OpenMP_CXX
# new
# compute # disable this because we don't use Vulkan LBA anymore
# fmt::fmt
graphite
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()

#Old examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)

add_executable(rgbd_tum_old
        Examples_old/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i_old
            Examples_old/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i_old
            Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)

add_executable(stereo_kitti_old
        Examples_old/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

add_executable(stereo_euroc_old
        Examples_old/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

add_executable(stereo_tum_vi_old
        Examples_old/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265_old
            Examples_old/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i_old
            Examples_old/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)

add_executable(mono_tum_old
        Examples_old/Monocular/mono_tum.cc)
target_link_libraries(mono_tum_old ${PROJECT_NAME})

add_executable(mono_kitti_old
        Examples_old/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti_old ${PROJECT_NAME})

add_executable(mono_euroc_old
        Examples_old/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc_old ${PROJECT_NAME})

add_executable(mono_tum_vi_old
        Examples_old/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265_old
            Examples_old/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_realsense_D435i_old
            Examples_old/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)

add_executable(mono_inertial_euroc_old
        Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi_old
        Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)

add_executable(stereo_inertial_euroc_old
        Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi_old
        Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

# Tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Tests)

add_executable(test_normalize_rotation
        Examples/Tests/test_normalize_rotation.cu)
target_link_libraries(test_normalize_rotation ${PROJECT_NAME})

add_executable(test_orb_extraction
        Examples/Tests/test_orb_extraction.cc)
target_link_libraries(test_orb_extraction ${PROJECT_NAME})

add_executable(bench_orb_extraction
        Examples/Tests/bench_orb_extraction.cc)
target_link_libraries(bench_orb_extraction ${PROJECT_NAME})
