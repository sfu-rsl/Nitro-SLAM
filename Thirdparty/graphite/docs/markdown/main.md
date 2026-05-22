Introduction                         {#mainpage}
============

Graphite is a GPU-accelerated mixed-precision nonlinear least squares graph optimization framework.
It allows you to represent problems as a graph and optimize parameters in-place on the GPU using an
algorithm such as Levenberg-Marquardt.

To get started, you may find it helpful to view the [examples](./examples.html).

\warning
Graphite is a work-in-progress research prototype. It may have several bugs, performance issues, and other limitations. The interface and implementation may change over time.

### Important Links
- [Documentation](https://sfu-rsl.github.io/graphite/)
- [GitHub Repository](https://github.com/sfu-rsl/graphite)
- [Paper](https://arxiv.org/abs/2509.26581)

## Building

You need the following dependencies:
- CMake
- Eigen3
- CUDA Toolkit (e.g. >= 12.0)
- cuDSS 0.7.0
- Doxygen (optional, for documentation)

Other dependencies are automatically fetched by CMake.

We also provide a Dockerfile for development, along with a devcontainer configuration for Visual Studio Code (requires the NVIDIA Container Toolkit). 

You can build Graphite and its examples using the following commands:

```bash
git clone https://github.com/sfu-rsl/graphite.git
cmake -DCMAKE_BUILD_TYPE=Release graphite -B build
cmake --build build
```

You may also need to define `-DCMAKE_CUDA_ARCHITECTURES` if CMake cannot detect your GPU architecture. For example, for an NVIDIA RTX 3090, you can set it to `86`. You can look up the compute capability number for recent GPUs [here](https://developer.nvidia.com/cuda/gpus) and for older GPUs [here](https://developer.nvidia.com/cuda/gpus/legacy).

## Project Integration

To instead use Graphite as a library inside an existing CMake project,
you can add it as a dependency using the [`add_subdirectory(...)`](https://cmake.org/cmake/help/latest/command/add_subdirectory.html) command in your `CMakeLists.txt` file. You can then link `graphite` to your build target. 

You may also have to adjust the following settings near the top of your project's CMakeLists.txt file, before adding the subdirectory:
```cmake
project(YourProject LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda --default-stream per-thread")
set(CMAKE_CUDA_ARCHITECTURES 86) # You may need to look up the correct CC number for your GPU
```

\note
Graphite code should be called inside a .cu file to avoid compile errors.

## Workflow

The overall workflow consists of three steps:

1. Define static properties
2. Construct an optimizable graph
3. Optimize the graph

## Static Properties

Optimizable variables and constraints are described by creating
data structures with static properties.


### Vertex Descriptors

To use a data type as an optimizable variable, you need to define
its properties so that Graphite knows how to interact with it.

Suppose you want to perform bundle adjustment to optimize camera parameters
and 3D points, and you represent a camera as a \f$9\times1\f$ vector.

```cpp
template <typename T> using Camera = Eigen::Matrix<T, 9, 1>;
```
\note
We show these examples using templates. However, it is not strictly required when defining static properties as shown next.

The properties for the camera can be defined as below.

```cpp

template <typename T> struct CameraTraits {
  static constexpr size_t dimension = 9;
  using State = Camera<T>; // State can be optionally defined
  using Vertex = Camera<T>;

  template <typename P>
  d_fn static void parameters(const Vertex &vertex, P *parameters) {
    Eigen::Map<Eigen::Matrix<P, dimension, 1>> params_map(parameters);
    params_map = vertex.template cast<P>();
  }

  d_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }

  // Defining the state requires custom setters and getters
  d_fn static State get_state(const Vertex &vertex) { return vertex; }

  d_fn static void set_state(Vertex &vertex, const State &state) {
    vertex = state;
  }
};
```

\note 
The macro `d_fn` is an alias for [`__device__`](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#device-and-host-functions).

We will discuss each of these properties next.

The value of `dimension` defines the size of the variable's parameter block.

```cpp
static constexpr size_t dimension = 9;
```
You can optionally define a data type for the state.
This is an *optional* definition which is used when backing up and restoring
a variable. It's useful if for some reason, you don't want the entire data structure for the variable to be copied or modified.

```cpp
using State = Camera<T>;
```

Defining `State` requires additional getters and setters to descibe the conversion.

```cpp
d_fn static State get_state(const Vertex &vertex) { return vertex; }

d_fn static void set_state(Vertex &vertex, const State &state) {
vertex = state;
}
```

The underlying data type for the variable is defined by assigning a type to `Vertex`.

```cpp
using Vertex = Camera<T>;
```

The `parameters(...)` function constructs a 1D parameterization, casts it to type `P` (a floating point value or a dual number), and copies it into the location at `parameters`, which is a 9-element array.

```cpp  
template <typename P>
d_fn static void parameters(const Vertex &vertex, P *parameters) {
    Eigen::Map<Eigen::Matrix<P, dimension, 1>> params_map(parameters);
    params_map = vertex.template cast<P>();
}
```
\note The `parameters(...)` function can have an empty definition if you are *not* using automatic differentiation.

The `update(...)` function describes how your variable should be updated,
where `delta` is a 9-element array.

```cpp
d_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
}
```

After defining the properties of your variable, the next step is to
define the corresponding descriptor, which takes the properties in as a template parameter.

```cpp
template <typename T, typename S>
using PointDescriptor = VertexDescriptor<T, S, PointTraits<T>>;
```

This point descriptor can then be initialized and added to a graph.

### Factor (Constraint) Descriptor

Suppose you want to model the reprojection error for bundle adjustment.
As before, you can define a data structure for your constraint properties.


```cpp
template <typename T, typename S> struct ReprojectionErrorTraits {
  static constexpr size_t dimension = 2;
  using VertexDescriptors =
      std::tuple<CameraDescriptor<T, S>, PointDescriptor<T, S>>;
  using Observation = Eigen::Matrix<T, dimension, 1>;
  using Data = Empty;
  using Loss = DefaultLoss<T, dimension>;
  // using Differentiation = DifferentiationMode::Auto;
  using Differentiation = DifferentiationMode::Manual;

  // You can pass in vertex references (class references), parameter blocks
  // (pointer to 1D parameters), or both. The framework will automatically call
  // your function with the correct arguments.
  template <typename D>
  d_fn static void error(const D *camera, const D *point,
                         const Observation &obs, D *error) {
    bal_reprojection_error_simple<D, Observation, T>(camera, point, &obs,
                                                     error);
  }

  template <typename D, size_t I>
  d_fn static void jacobian(const Camera<T> &camera, const Point<T> &point,
                            const Observation &obs, D *jacobian) {
    bal_jacobian_simple<T, D, I>(camera.data(), point.data(), &obs, jacobian);
  }
};

template <typename T, typename S>
using ReprojectionError = FactorDescriptor<T, S, ReprojectionErrorTraits<T, S>>;
```

Again, we define a dimension, this time for the length of the residual vector,
which are 2D coordinates in this case.

```cpp
static constexpr size_t dimension = 2;
```

To tell the factor what types of vertices are involved, we define a tuple for
`VertexDescriptors`.

```cpp
using VertexDescriptors =
      std::tuple<CameraDescriptor<T, S>, PointDescriptor<T, S>>;
```

We use a \f$2\times1\f$ vector to represent the observation.

```cpp
using Observation = Eigen::Matrix<T, dimension, 1>;
```

\note
Some factors may not need an observation data type. For those cases, Graphite provides `graphite::Empty`.


Sometimes you need some constants or non-optimizable data. For this, we can define a type for `Data`.
However, this is not necessary in this particular example, so we set it to `Empty`.

```cpp
using Data = Empty;
```

The loss function is specified as follows.

```cpp
using Loss = DefaultLoss<T, dimension>;
```

We can also set the differentiation mode as below.

```cpp
using Differentiation = DifferentiationMode::Manual;
```
\note
Graphite supports automatic differentiation, manual differentiation, and dynamic (on-the-fly) manual differentiation. 

\note
To compute Jacobians on-the-fly, your factor must use manual differentiation. Then the following setting can be toggled at runtime.
```cpp
factor_desc.set_jacobian_storage(false);
```



Next, the function for computing the residual is defined. In this case, since
`Data` is `Empty`, we don't pass it to `error(...)`.

```cpp
template <typename D>
d_fn static void error(const D *camera, const D *point,
                        const Observation &obs, D *error) {
bal_reprojection_error_simple<D, Observation, T>(camera, point, &obs,
                                                    error);
}
```

\note
Graphite supports various error function signatures. Functions can be constructed with parameters in the following order.
1. `const A& vertex1, const B& vertex2, ....` (required if skipping 2.)
2. `const D* vertex1_params, const D* vertex2_params, ...` (required if using automatic differentiation)
3. `const Observation& obs` (required if not `Empty`)
4. `const Data& data` (required if not `Empty`)
5. `D* error` (required) 

\note In some cases, you may want to pass both vertex references and vertex parameters to the same function, such as when you are using automatic differentiation, but want to reduce the degrees of freedom (e.g. 4DoF pose graph optimization).


Since manual differentiation was chosen, it is necessary to define a function which computes the Jacobians, where `D` is the precision of the Jacobian block and `I` is the index of the corresponding vertex.

```cpp
template <typename D, size_t I>
d_fn static void jacobian(const Camera<T> &camera, const Point<T> &point,
                        const Observation &obs, D *jacobian) {
    bal_jacobian_simple<T, D, I>(camera.data(), point.data(), &obs, jacobian);
}
```
\note
You can handle different values of `I` using `if constexpr`. 
```cpp
d_fn static void jacobian(const Camera<T> &camera, const Point<T> &point,
                        const Observation &obs, D *jacobian) {
   if constexpr (I == 0) {
       // Compute camera Jacobian
   }
   else {
       // Compute point Jacobian
   }
}
```

Lastly, the descriptor can be defined using the properties.

```cpp
template <typename T, typename S>
using ReprojectionError = FactorDescriptor<T, S, ReprojectionErrorTraits<T, S>>;
```
