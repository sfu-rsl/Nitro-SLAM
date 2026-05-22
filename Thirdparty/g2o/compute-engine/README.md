# Compute Engine

This is a work-in-progress library which implements sparse matrix-matrix and matrix-vector operations to compute the Schur complement for visual and visual-inertial bundle adjustment using Vulkan compute shaders.
Please see the [main project](https://github.com/sfu-rsl/gpu-block-solver) or our [paper](https://ieeexplore.ieee.org/document/10160499) for more information.

Note: Please see [this branch](https://github.com/sfu-rsl/compute-engine/tree/sycl-backend) for the experimental SYCL backend.

## Hardware Requirements

- A GPU which supports Vulkan 1.2 and double-precision floats in shaders

## Building

Dependencies:

- Vulkan 1.2 (may require Vulkan 1.3 SDK/headers)
- Modified Kompute Library (submodule)
- Eigen 3.1
- Google Test
- Google Benchmark
- robin_hood unordered_map (included)
- glslangValidator (needs to support `--target-env vulkan1.1` flag)
  - This is available from the Ubuntu `vulkan-tools` package or can also be [built manually](https://github.com/KhronosGroup/glslang) for a more up-to-date version.
- Python 3


```bash
# Clone
git clone https://github.com/sfu-rsl/compute-engine.git --recursive

# After cloning
mkdir build
cd build

# Build
cmake /path/to/repo
make -j

# Run tests
./tests
```

## Locking the GPU Frequencies

GPU frequency scaling may affect the performance, especially for small workloads. For NVIDIA GPUs on Linux, the following commands can be used to lock the frequencies, but be sure to check for specific instructions for your OS, GPU, and driver version.

```bash
# Setting PowerMizer mode to prefer maximum performance
nvidia-settings -a "[gpu:0]/GpuPowerMizerMode=1"


# Enable persistent mode
sudo nvidia-smi -pm 1 

# Query clocks
nvidia-smi -q -d SUPPORTED_CLOCKS

# Set clocks to max supported
sudo nvidia-smi -lgc <max graphics clock>
sudo nvidia-smi -lmc <max memory clock>


# To reset once done
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

## Examples

Please see the [tests](src/tests.cpp) for example usage.
