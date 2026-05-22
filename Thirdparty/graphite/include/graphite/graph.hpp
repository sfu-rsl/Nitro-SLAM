/// @file graph.hpp
#pragma once
#include <graphite/factor.hpp>
#include <graphite/stream.hpp>
#include <graphite/vertex.hpp>
#include <limits>
#include <thrust/execution_policy.h>

namespace graphite {

/**
 * @brief Graph class which stores references to vertex and factor descriptors,
 * and provides methods for optimization.
 *
 * @tparam T Scalar type (e.g., double, float) representing the precision of the
 * vertices and constraints.
 * @tparam S Scalar type (e.g., double, Point2D) representing the precision of
 * the linear system.
 */
template <typename T, typename S> class Graph {

private:
  std::vector<BaseVertexDescriptor<T, S> *> vertex_descriptors;
  std::vector<BaseFactorDescriptor<T, S> *> factor_descriptors;
  thrust::device_vector<T> b;
  thrust::device_vector<T> jacobian_scales;
  size_t hessian_column;
  std::vector<size_t> hessian_offsets;
  bool scale_jacobians;

public:
  Graph() : scale_jacobians(true) {}

  size_t get_hessian_dimension() const { return hessian_column; }
  size_t get_variable_dimension(const size_t block_index) const {
    return hessian_offsets[block_index + 1] - hessian_offsets[block_index];
  }
  size_t get_num_block_columns() const { return hessian_offsets.size() - 1; }

  const std::vector<size_t> &get_offset_vector() const {
    return hessian_offsets;
  }

  thrust::device_vector<T> &get_b() { return b; }

  std::vector<BaseVertexDescriptor<T, S> *> &get_vertex_descriptors() {
    return vertex_descriptors;
  }

  std::vector<BaseFactorDescriptor<T, S> *> &get_factor_descriptors() {
    return factor_descriptors;
  }

  thrust::device_vector<T> &get_jacobian_scales() { return jacobian_scales; }

  template <typename V> void add_vertex_descriptor(V *descriptor) {
    vertex_descriptors.push_back(descriptor);
  }

  template <typename F> void add_factor_descriptor(F *factor) {
    factor_descriptors.push_back(factor);
  }

  template <typename D> void add_descriptor(D *descriptor) {
    if constexpr (std::is_base_of<BaseVertexDescriptor<T, S>, D>::value) {
      add_vertex_descriptor(descriptor);
    } else if constexpr (std::is_base_of<BaseFactorDescriptor<T, S>,
                                         D>::value) {
      add_factor_descriptor(descriptor);
    } else {
      static_assert(std::is_base_of<BaseVertexDescriptor<T, S>, D>::value ||
                        std::is_base_of<BaseFactorDescriptor<T, S>, D>::value,
                    "You tried to add something strange to the graph.");
    }
  }

  bool initialize_optimization(const uint8_t level) {

    // For each vertex descriptor, take global to local id mapping and transform
    // it into a Hessian column to local id mapping.

    std::vector<std::pair<size_t, std::pair<size_t, size_t>>>
        global_to_local_combined;

    for (size_t i = 0; i < vertex_descriptors.size(); ++i) {
      const auto &map = vertex_descriptors[i]->get_global_map();
      for (const auto &entry : map) {
        global_to_local_combined.push_back({entry.first, {i, entry.second}});
      }
    }

    // Sort the combined list by global ID
    std::sort(global_to_local_combined.begin(), global_to_local_combined.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    // Initialize device ids and copy over factor and current vertex state
    for (auto &desc : factor_descriptors) {
      desc->initialize_device_ids(level);
    }
    deactivate_unused_vertices(level);

    // Assign Hessian columns to local indices
    hessian_column = 0;
    size_t hessian_block_index = 0;
    hessian_offsets.clear();
    for (const auto &entry : global_to_local_combined) {
      if (vertex_descriptors[entry.second.first]->is_active(entry.first)) {
        vertex_descriptors[entry.second.first]->set_hessian_column(
            entry.first, hessian_column, hessian_block_index);
        hessian_offsets.push_back(hessian_column);
        hessian_column += vertex_descriptors[entry.second.first]->dimension();
        hessian_block_index++;
      }
    }
    hessian_offsets.push_back(hessian_column);

    // Copy vertex values to device
    for (auto &desc : vertex_descriptors) {
      desc->to_device();
    }

    // Copy factors to device
    for (auto &desc : factor_descriptors) {
      desc->to_device();
    }

    // Initialize Jacobian storage
    for (auto &f : factor_descriptors) {
      f->initialize_jacobian_storage();
    }

    return true;
  }

  // Deactivates vertices of inactive factors
  // Expects that vertices and factor states are finalized
  void deactivate_unused_vertices(const uint8_t level) {

    // Check for empty descriptors
    for (size_t i = 0; i < vertex_descriptors.size(); ++i) {
      if (vertex_descriptors[i]->count() == 0) {
        std::cerr << "Error: Vertex descriptor " << i << " has no entries."
                  << std::endl;
      }
    }

    for (size_t i = 0; i < factor_descriptors.size(); ++i) {
      if (factor_descriptors[i]->active_count() == 0) {
        std::cerr << "Error: Factor descriptor " << i << " has no entries."
                  << std::endl;
      }
    }

    // For each vertex descriptor, set the state MSB to 0
    for (auto &desc : vertex_descriptors) {
      thrust::transform(
          thrust::cuda::par_nosync.on(0), desc->get_active_state(),
          desc->get_active_state() + desc->count(), desc->get_active_state(),
          [] __device__(uint8_t state) { return state & 0x7F; });
    }
    // For each factor descriptor
    // Go through each vertex descriptor and set the state MSB to 1 if the
    // constraint is active
    for (auto &desc : factor_descriptors) {
      desc->flag_active_vertices_async(level); // stream 0
    }
    // For each vertex descriptor, MSB of the active state is XOR'd with 1
    // (0->1, 1->0)
    for (auto &desc : vertex_descriptors) {
      thrust::transform(
          thrust::cuda::par_nosync.on(0), desc->get_active_state(),
          desc->get_active_state() + desc->count(), desc->get_active_state(),
          [] __device__(uint8_t state) { return state ^ 0x80; });
    }
    cudaStreamSynchronize(0);
  }

  bool build_structure() {
    // Allocate storage for solver vectors
    const auto size_x = get_hessian_dimension();
    b.resize(size_x);
    jacobian_scales.resize(size_x);

    return true;
  }

  void compute_error() {
    for (auto &factor : factor_descriptors) {
      factor->compute_error(); // TODO: Make non-autodiff version
    }
    cudaStreamSynchronize(0);
  }

  T chi2() {
    T chi2 = static_cast<T>(0);
    for (auto &factor : factor_descriptors) {
      chi2 += factor->chi2();
    }
    return chi2;
  }

  void linearize(StreamPool &streams) {

    for (auto &factor : factor_descriptors) {
      // compute error
      if (factor->use_autodiff() && (factor->store_jacobians() ||
                                     !factor->supports_dynamic_jacobians())) {
        factor->compute_error_autodiff(streams); // synchronous
      } else {
        factor->compute_error();            // synchronous
        factor->compute_jacobians(streams); // synchronous
      }
    }
    cudaStreamSynchronize(0);

    // Compute chi2
    chi2();

    // Compute Jacobian scale
    if (scale_jacobians) {
      thrust::fill(jacobian_scales.begin(), jacobian_scales.end(), 0);
      for (auto &factor : factor_descriptors) {
        factor->compute_hessian_scalar_diagonal_async(
            jacobian_scales.data().get(), nullptr);
      }
      cudaStreamSynchronize(0);

      thrust::transform(
          thrust::device, jacobian_scales.begin(), jacobian_scales.end(),
          jacobian_scales.begin(), [] __device__(T value) {
            const double denom = std::numeric_limits<double>::epsilon() +
                                 sqrt(static_cast<double>(value));
            // const double denom = 1.0
            //   + sqrt(static_cast<double>(value));
            return static_cast<T>(1.0 / denom);
          });
    } else {
      thrust::fill(jacobian_scales.begin(), jacobian_scales.end(), 1.0);
    }

    // Scale Jacobians
    if (scale_jacobians) {
      for (auto &factor : factor_descriptors) {
        factor->scale_jacobians_async(jacobian_scales.data().get());
      }
      cudaStreamSynchronize(0);
    }

    // Calculate b=J^T * r
    thrust::fill(thrust::cuda::par_nosync.on(0), b.begin(), b.end(), 0);
    for (auto &fd : factor_descriptors) {
      fd->compute_b_async(b.data().get(), jacobian_scales.data().get());
    }

    cudaStreamSynchronize(0);
  }

  void apply_update(const T *delta_x, StreamPool &streams) {
    size_t i = 0;
    for (auto &desc : vertex_descriptors) {
      desc->apply_update_async(delta_x, jacobian_scales.data().get(),
                               streams.select(i));
      i++;
    }
    streams.sync_n(i);
  }

  void backup_parameters() {

    for (const auto &desc : vertex_descriptors) {
      desc->backup_parameters_async();
    }

    cudaStreamSynchronize(0);
  }

  void revert_parameters() {

    for (auto &desc : vertex_descriptors) {
      desc->restore_parameters_async();
    }

    cudaStreamSynchronize(0);
  }

  void clear() {
    vertex_descriptors.clear();
    factor_descriptors.clear();
    b.clear();
    jacobian_scales.clear();
    hessian_column = 0;
    hessian_offsets.clear();
  }

  void scale_system(const bool enable_scaling) {
    scale_jacobians = enable_scaling;
  }
};

/**
 * \example circle.cu
 * This is an example of a simple optimization problem where points are fit to a
 * circle. \example bal.cu This is an example of bundle adjustment on the BAL
 * dataset.
 */

} // namespace graphite