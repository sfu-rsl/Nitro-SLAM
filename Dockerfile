# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
# FROM ubuntu:22.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Install git, compiler, cmake
RUN apt-get update && apt-get -y install \
build-essential cmake git git-lfs && \
# Install pangolin pre-reqs
apt-get -y install libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev \
libc++-dev libglew-dev libeigen3-dev cmake g++ ninja-build \
libjpeg-dev libpng-dev \
libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev

# Install other ORB-SLAM3 and CUDA dependencies
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -y install libopencv-dev libopencv-core-dev libeigen3-dev libboost-serialization-dev libssl-dev 

# RUN apt-get -y install nvidia-cuda-toolkit nvidia-cuda-dev nvidia-cuda-gdb

# Install CUDA Toolkit 12.6
# RUN apt-get -y install cuda-toolkit-12-6 cuda-gdb-12-6
RUN apt-get -y install cuda-toolkit-12-8 cuda-gdb-12-8 cudss-cuda-12

# ORB-SLAM3 Stuff
# Install pangolin
RUN apt-get -y install python3-dev python3-setuptools
RUN git clone --branch v0.6 --recursive https://github.com/stevenlovegrove/Pangolin.git && \
cd Pangolin && \
cmake -B build -GNinja && \
cmake --build build && \
cd build && ninja install

# Things for vulkan
RUN apt-get update && apt-get -y install libvulkan-dev vulkan-tools glslang-tools python3-click xxd

# may need to install libnvidia-gl with the right version for your system

# COPY nvidia_icd.json /etc/vulkan/icd.d

# From https://github.com/j3soon/docker-vulkan-runtime/blob/master/Dockerfile
RUN cat > /etc/vulkan/icd.d/nvidia_icd.json <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194"
    }
}
EOF
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
RUN cat > /etc/vulkan/implicit_layer.d/nvidia_layers.json <<EOF
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
EOF
