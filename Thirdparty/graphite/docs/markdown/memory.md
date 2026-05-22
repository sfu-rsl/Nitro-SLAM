Memory Management                        
============

In Graphite, optimizable variables are *pointers* associated with an identifier. This means that you have a few options for storing these variables on some systems.

For older systems, you may have to explicitly allocate memory through CUDA. For device (GPU) memory, this includes memory allocated using:
- `cudaMalloc`
- `thrust::device_vector`

Alternatively, you can use unified memory allocated from:
- `cudaMallocManaged`
- `thrust::universal_vector`
- `graphite::managed_vector`

\note
We found that `thrust::universal_vector` is slow when pushing back elements even after reserving memory, so we recommend using `graphite::managed_vector` instead.
However, this can change in the future, so you should try both.

Newer GPUs and platforms have more advanced unified memory support. For example, the RTX 5080 on Linux supports heterogeneous memory management (HMM), meaning that you can
use **system-allocated** memory, e.g. memory allocated on the **stack** or **heap**. For example:

```cpp
Eigen::Vector3d points_stack[100];
std::vector<Eigen::Vector3d> points_heap;
```

For more information about unified memory and HMM, check out CUDA's [documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).

