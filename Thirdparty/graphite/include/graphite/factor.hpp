/// @file factor.hpp
#pragma once
#include <graphite/active.hpp>
#include <graphite/block.hpp>
#include <graphite/common.hpp>
#include <graphite/differentiation.hpp>
#include <graphite/loss.hpp>
#include <graphite/ops/active.hpp>
#include <graphite/ops/chi2.hpp>
#include <graphite/ops/error.hpp>
#include <graphite/ops/hessian.hpp>
#include <graphite/ops/linearize.hpp>
#include <graphite/ops/product.hpp>
#include <graphite/utils.hpp>
#include <graphite/vector.hpp>
#include <graphite/vertex.hpp>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/universal_vector.h>

namespace graphite {

/**
 * @brief Storage class for Jacobians of a factor.
 */
template <typename S> class JacobianStorage {
public:
  std::pair<size_t, size_t> dimensions;

  thrust::device_vector<S> data;
};

/**
 * @brief Base class for factor descriptors.
 */
template <typename T, typename S> class BaseFactorDescriptor {
public:
  using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

  virtual ~BaseFactorDescriptor(){};

  virtual bool use_autodiff() = 0;
  virtual void compute_error() = 0;
  virtual void compute_error_autodiff(StreamPool &streams) = 0;
  virtual void compute_b_async(T *b, const T *jacobian_scales) = 0;
  virtual void compute_Jv(T *out, T *in, const T *jacobian_scales,
                          StreamPool &streams) = 0;
  virtual void compute_Jtv(T *out, T *in, const T *jacobian_scales,
                           StreamPool &streams) = 0;

  virtual void flag_active_vertices_async(const uint8_t level) = 0;

  virtual void compute_jacobians(StreamPool &streams) = 0;
  virtual void compute_hessian_block_diagonal_async(
      std::unordered_map<BaseVertexDescriptor<T, S> *,
                         thrust::device_vector<InvP>> &block_diagonals,
      const T *jacobian_scales, cudaStream_t stream) = 0;

  virtual void
  compute_hessian_scalar_diagonal_async(T *diagonal,
                                        const T *jacobian_scales) = 0;

  virtual JacobianStorage<S> *get_jacobians() = 0;
  virtual void initialize_jacobian_storage() = 0;

  virtual size_t active_count() const = 0;
  virtual size_t get_num_descriptors() const = 0;

  virtual size_t get_residual_size() const = 0;
  virtual void scale_jacobians_async(T *jacobian_scales) = 0;

  virtual void initialize_device_ids(const uint8_t optimization_level) = 0;
  virtual void to_device() = 0;

  virtual T chi2() = 0;

  virtual void set_jacobian_storage(const bool store) = 0;
  virtual bool store_jacobians() = 0;
  virtual bool supports_dynamic_jacobians() = 0;

  virtual void get_hessian_block_coordinates(
      thrust::device_vector<BlockCoordinates> &block_coords) = 0;

  virtual size_t setup_hessian_computation(
      std::unordered_map<BlockCoordinates, size_t> &block_indices,
      thrust::device_vector<S> &d_hessian, size_t *h_block_offsets,
      StreamPool &streams) = 0;

  virtual size_t execute_hessian_computation(
      std::unordered_map<BlockCoordinates, size_t> &block_indices,
      thrust::device_vector<S> &d_hessian, size_t *d_block_offsets,
      StreamPool &streams) = 0;
};

template <typename T> struct get_vertex_type {
  using type = typename T::VertexType;
};

template <typename T> struct get_vertex_pointer_type {
  using type = typename T::VertexType *;
};

template <typename T> struct get_vertex_pointer_pointer_type {
  using type = typename T::VertexType **;
};

template <typename Tuple, template <typename> class MetaFunc>
struct transform_tuple;

// Partial specialization for std::tuple
template <typename... Ts, template <typename> class MetaFunc>
struct transform_tuple<std::tuple<Ts...>, MetaFunc> {
  using type = std::tuple<typename MetaFunc<Ts>::type...>;
};

/**
 * @brief Represents a group of factors which will be processed together on the
 * GPU.
 */
template <typename T, typename S, typename FTraits>
class FactorDescriptor : public BaseFactorDescriptor<T, S> {

private:
  std::vector<size_t> global_ids;
  std::unordered_map<size_t, size_t> global_to_local_map;
  std::vector<size_t> local_to_global_map;

  HandleManager<size_t> hm;

  bool _store_jacobians;
  size_t _active_count;

public:
  using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

  using Traits = FTraits;

  static constexpr size_t N =
      std::tuple_size<typename Traits::VertexDescriptors>::value;
  static constexpr size_t error_dim = Traits::dimension;

  std::array<BaseVertexDescriptor<T, S> *, N> vertex_descriptors;

  using VertexDescriptorTuple = typename Traits::VertexDescriptors;
  using VertexTypesTuple =
      typename transform_tuple<VertexDescriptorTuple, get_vertex_type>::type;
  using VertexPointerTuple =
      typename transform_tuple<VertexDescriptorTuple,
                               get_vertex_pointer_type>::type;
  using VertexPointerPointerTuple =
      typename transform_tuple<VertexDescriptorTuple,
                               get_vertex_pointer_pointer_type>::type;

  using ObservationType = typename Traits::Observation;
  using ConstraintDataType = typename Traits::Data;
  using LossType = typename Traits::Loss;

  thrust::host_vector<size_t> host_ids;
  thrust::device_vector<size_t> device_ids;
  managed_vector<ObservationType> device_obs;
  thrust::device_vector<T> residuals;
  managed_vector<S> precision_matrices;
  managed_vector<ConstraintDataType> data;

  managed_vector<T> chi2_vec;
  thrust::device_vector<S> chi2_derivative;
  managed_vector<LossType> loss;

  thrust::host_vector<uint8_t> active;
  thrust::device_vector<uint8_t> device_active;
  thrust::device_vector<size_t> active_indices;

  std::array<JacobianStorage<S>, N> jacobians;
  std::array<S, Traits::dimension * Traits::dimension> default_precision_matrix;

  /**
   * @brief Constructs a FactorDescriptor with the given vertex descriptors.
   * @tparam VertexDescPtrs Types of the vertex descriptor pointers.
   * @param vertex_descriptors Pointers to the vertex descriptors.
   */
  template <typename... VertexDescPtrs,
            typename = std::enable_if_t<sizeof...(VertexDescPtrs) == N>>
  FactorDescriptor(VertexDescPtrs... vertex_descriptors)
      : _store_jacobians(true), _active_count(0) {
    link_factors({vertex_descriptors...});

    default_precision_matrix = get_default_precision_matrix();
  }

  /**
   * @brief Computes the residuals across all factors in the descriptor.
   */
  void compute_error() override { ops::compute_error<T>(this); }

  /**
   * @brief Simultaneously computes the residuals and corresponding Jacobians
   * using automatic differentiation.
   */
  void compute_error_autodiff(StreamPool &streams) override {
    ops::compute_error_autodiff<T, S>(this, streams);
  }

  /**
   * @brief Computes the gradient vector b asynchronously.
   */
  void compute_b_async(T *b, const T *jacobian_scales) override {
    ops::compute_b_async<T, S>(this, b, jacobian_scales);
  }

  /**
   * @brief Computes the product of the Jacobian matrix and a vector.
   */
  void compute_Jv(T *out, T *in, const T *jacobian_scales,
                  StreamPool &streams) override {
    ops::compute_Jv<T, S>(this, out, in, jacobian_scales, streams);
  }

  /**
   * @brief Computes the product of the transposed Jacobian matrix and a vector.
   */
  void compute_Jtv(T *out, T *in, const T *jacobian_scales,
                   StreamPool &streams) override {
    ops::compute_Jtv<T, S>(this, out, in, jacobian_scales, streams);
  }

  void flag_active_vertices_async(const uint8_t level) override {
    ops::flag_active_vertices(this, level);
  }

  /**
   * @brief Computes the Jacobians using manual differentiation.
   */
  void compute_jacobians(StreamPool &streams) override {
    if constexpr (std::is_same_v<typename Traits::Differentiation,
                                 DifferentiationMode::Manual>) {
      ops::compute_jacobians<T, S>(this, streams);
    }
  }

  /**
   * @brief Computes the block diagonal of the Hessian matrix asynchronously.
   */
  void compute_hessian_block_diagonal_async(
      std::unordered_map<BaseVertexDescriptor<T, S> *,
                         thrust::device_vector<InvP>> &block_diagonals,
      const T *jacobian_scales, cudaStream_t stream) override {

    std::array<InvP *, N> diagonal_blocks;
    for (size_t i = 0; i < N; i++) {
      diagonal_blocks[i] = block_diagonals[vertex_descriptors[i]].data().get();
    }

    ops::compute_block_diagonal<T, S>(this, diagonal_blocks, jacobian_scales,
                                      stream);
  }

  /**
   * @brief Computes the scalar diagonal of the Hessian matrix asynchronously.
   */
  void
  compute_hessian_scalar_diagonal_async(T *diagonal,
                                        const T *jacobian_scales) override {
    ops::compute_hessian_scalar_diagonal<T, S>(this, diagonal, jacobian_scales);
  }

  /**
   * @brief Returns the number of vertices connected to each factor.
   */
  static constexpr size_t get_num_vertices() { return N; }

  /**
   * @brief Returns the number of vertex descriptors (same as the number of
   * vertices for each factor).
   */
  size_t get_num_descriptors() const override { return N; }

  /**
   * @brief Returns the Jacobian storage for the factor.
   */
  JacobianStorage<S> *get_jacobians() override { return jacobians.data(); }

  /**
   * @brief Reserves memory for the factor. Generally, you should always reserve
   * the memory you need before adding factors, because reallocating GPU memory
   * is expensive.
   * @param size The number of factors to reserve memory for.
   */
  void reserve(size_t size) {
    global_ids.reserve(N * size);
    host_ids.reserve(N * size);
    device_ids.reserve(N * size);
    device_obs.reserve(size);
    global_to_local_map.reserve(size);
    local_to_global_map.reserve(size);
    precision_matrices.reserve(size * error_dim * error_dim);
    data.reserve(size);
    loss.reserve(size);
    chi2_vec.reserve(size);
    residuals.reserve(size * error_dim);
    active.reserve(size);
  }

  /**
   * @brief Removes a factor by its id.
   * @param id The id of the factor to remove (i.e. the id returned by
   * add_factor).
   */
  void remove_factor(const size_t id) {
    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return;
    }

    auto local_id = global_to_local_map[id];
    auto last_local_id = internal_count() - 1;

    // copy constraint
    for (size_t i = 0; i < N; i++) {
      global_ids[local_id * N + i] = global_ids[last_local_id * N + i];
    }
    global_ids.resize(global_ids.size() - N);

    // observation
    device_obs[local_id] = device_obs[last_local_id];
    device_obs.pop_back();

    // precision matrix
    constexpr size_t precision_matrix_size = error_dim * error_dim;
    for (size_t i = 0; i < precision_matrix_size; i++) {
      precision_matrices[local_id * precision_matrix_size + i] =
          precision_matrices[last_local_id * precision_matrix_size + i];
    }

    precision_matrices.resize(precision_matrices.size() -
                              precision_matrix_size);

    // Constraint, loss and chi2
    data[local_id] = data[last_local_id];
    data.pop_back();

    loss[local_id] = loss[last_local_id];
    loss.pop_back();

    chi2_vec[local_id] = chi2_vec[last_local_id];
    chi2_vec.pop_back();

    active[local_id] = active[last_local_id];
    active.pop_back();

    // Next, fix ids
    const auto last_global_id = local_to_global_map[last_local_id];
    global_to_local_map[last_global_id] = local_id;
    local_to_global_map[local_id] = last_global_id;

    // Remove unused entries
    global_to_local_map.erase(id);
    local_to_global_map.pop_back();

    hm.release(id);
  }

  /**
   * @brief Adds a factor to the descriptor.
   * @param ids The ids of the vertices connected to the factor. Note that later
   * arguments have default values.
   * @param obs The observation associated with the factor.
   * @param precision_matrix The precision matrix for the factor (default
   * nullptr is the identity matrix).
   * @param constraint_data The constraint data for the factor.
   * @param loss_func The loss function for the factor.
   * @return The id of the added factor.
   */
  size_t
  add_factor(const std::array<size_t, N> &ids,
             const ObservationType &obs = ObservationType(),
             const S *precision_matrix = nullptr,
             const ConstraintDataType &constraint_data = ConstraintDataType(),
             const LossType &loss_func = LossType()) {

    const auto id = hm.get();
    const auto local_id = internal_count();

    global_to_local_map.insert({id, local_id});
    local_to_global_map.push_back(id);

    global_ids.insert(global_ids.end(), ids.begin(), ids.end());
    device_obs.push_back(obs);

    constexpr size_t precision_matrix_size = error_dim * error_dim;
    if (precision_matrix) {
      precision_matrices.resize(precision_matrices.size() +
                                precision_matrix_size);
      for (size_t i = 0; i < precision_matrix_size; i++) {
        precision_matrices[local_id * precision_matrix_size + i] =
            precision_matrix[i];
      }
    } else {
      // constexpr auto pmat = get_default_precision_matrix();
      precision_matrices.resize(precision_matrices.size() +
                                precision_matrix_size);

      for (size_t i = 0; i < precision_matrix_size; i++) {
        precision_matrices[local_id * precision_matrix_size + i] =
            default_precision_matrix[i];
      }
    }

    data.push_back(constraint_data);
    loss.push_back(loss_func);
    active.push_back(0);
    return id; // only global within this class (it's just an external id)
  }

  /**
   * @brief Sets the active state of a factor.
   * @param id The id of the factor.
   * @param active_value The active state value to set (e.g 0).
   */
  void set_active(size_t id, const uint8_t active_value) {
    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return;
    }

    constexpr uint8_t NOT_MSB = 0x7F; // 01111111

    auto local_id = global_to_local_map[id];
    active[local_id] =
        NOT_MSB & active_value; // we reserve the MSB for later use
  }

  /**
   * @brief Resets the active state of all factors to 0x1.
   */
  void reset_active() {
    thrust::fill(thrust::device, active.begin(), active.end(), 0x1);
  }

  /**
   * @brief Returns the number of all active and inactive factors.
   * @return The internal count of all factors.
   */
  size_t internal_count() const { return global_ids.size() / N; }

  /**
   * @brief Returns the number of active factors.
   * @return The count of active factors.
   */
  size_t active_count() const override { return _active_count; }

  /**
   * @brief Initializes the device ids and a list of active factors.
   * @param optimization_level The optimization level to use.
   */
  void initialize_device_ids(const uint8_t optimization_level) override {
    // Map constraint ids to local ids
    host_ids.resize(global_ids.size());
    for (size_t i = 0; i < global_ids.size(); i++) {
      host_ids[i] =
          vertex_descriptors[i % N]->get_global_map().at(global_ids[i]);
    }
    device_ids = host_ids;

    device_active = active;
    _active_count = build_active_indices(device_active, active_indices,
                                         internal_count(), optimization_level);
  }

  /**
   * @brief Prepares descriptor for GPU processing.
   */
  void to_device() override {
    chi2_vec.resize(internal_count());
    chi2_derivative.resize(internal_count());

    // Resize and reset residuals
    residuals.resize(error_dim * internal_count());
    thrust::fill(residuals.begin(), residuals.end(), 0);
  }

  /**
   * @brief Links the factor to the given vertex descriptors. This is already
   * called during construction.
   * @param vertex_descriptors The array of vertex descriptors to link.
   */
  void link_factors(
      const std::array<BaseVertexDescriptor<T, S> *, N> &vertex_descriptors) {
    this->vertex_descriptors = vertex_descriptors;
  }

  template <std::size_t... I>
  VertexPointerPointerTuple get_vertices_impl(std::index_sequence<I...>) {
    return std::make_tuple(
        (static_cast<typename std::tuple_element<I, VertexDescriptorTuple>::type
                         *>(vertex_descriptors[I])
             ->vertices())...);
  }

  /**
   * @brief Returns a tuple of vertex pointers.
   * @return A tuple containing pointers to the vertex data for each vertex
   * descriptor.
   */
  VertexPointerPointerTuple get_vertices() {
    return get_vertices_impl(std::make_index_sequence<N>{});
  }

  template <std::size_t... I>
  static constexpr std::array<size_t, N>
  get_vertex_sizes_impl(std::index_sequence<I...>) {
    return std::array<size_t, N>{
        std::tuple_element_t<I, VertexDescriptorTuple>::dim...};
  }

  /**
   * @brief Returns the sizes (dimensions) of the vertices.
   * @return An array containing the sizes of the vertices.
   */
  static constexpr std::array<size_t, N> get_vertex_sizes() {
    return get_vertex_sizes_impl(std::make_index_sequence<N>{});
  }

  /**
   * @brief Allocates memory for the Jacobians.
   */
  void initialize_jacobian_storage() override {
    for (size_t i = 0; i < N; i++) {
      if (store_jacobians() || !std::is_same_v<typename Traits::Differentiation,
                                               DifferentiationMode::Manual>) {
        jacobians[i].dimensions = {error_dim,
                                   vertex_descriptors[i]->dimension()};
        jacobians[i].data.resize(
            error_dim * vertex_descriptors[i]->dimension() * internal_count());
      }
    }
  }

  /**
   * @brief Returns the size of the vector for all active and inactive
   * residuals.
   * @return The size of the residuals.
   */
  virtual size_t get_residual_size() const override {
    return error_dim * internal_count();
  }

  /**
   * @brief Computes the chi-squared value for all active and inactive factors.
   * @return The chi-squared value.
   */
  virtual T chi2() override {
    ops::compute_chi2_async<T, S>(this); // runs on stream 0
    // TODO: Make this consider kernels and active edges
    return thrust::reduce(thrust::cuda::par.on(0), chi2_vec.begin(),
                          chi2_vec.end(), static_cast<T>(0.0),
                          thrust::plus<T>()); // want to sync here on stream 0
  }

  /**
   * @brief Returns the chi-squared value for a specific factor.
   * @param id The id of the factor.
   * @return The chi-squared value for the specified factor.
   */
  T chi2(const size_t id) const {
    auto it = global_to_local_map.find(id);
    if (it == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return static_cast<T>(0.0);
    }
    return chi2_vec[it->second];
  }

  /**
   * @brief Returns the constraint data for a specific factor.
   * @param id The id of the factor.
   * @return A pointer to the constraint data for the specified factor, or
   * nullptr if not found. The memory can be accessed on both the host and the
   * device.
   */
  ConstraintDataType *get_constraint_data(const size_t id) {
    auto it = global_to_local_map.find(id);
    if (it == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return nullptr;
    }
    return &data[it->second];
  }

  /**
   * @brief Returns the vertex ids connected to a specific factor.
   * @param id The id of the factor.
   * @return An array containing the vertex ids connected to the specified
   * factor.
   */
  std::array<size_t, N> get_vertex_ids(const size_t id) const {
    auto it = global_to_local_map.find(id);
    if (it == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return {};
    }
    const auto local_id = it->second;
    std::array<size_t, N> ids;
    for (size_t i = 0; i < N; i++) {
      ids[i] = global_ids[local_id * N + i];
    }
    return ids;
  }

  /**
   * @brief Scales the Jacobians asynchronously.
   * @param jacobian_scales Pointer to the array of scales for the Jacobians.
   */
  virtual void scale_jacobians_async(T *jacobian_scales) override {
    ops::scale_jacobians<T, S>(this, jacobian_scales);
  }

  /**
   * @brief Determines whether to use automatic differentiation based on the
   * traits of the factor.
   * @return True if automatic differentiation should be used, false otherwise.
   */
  virtual bool use_autodiff() override {
    return use_autodiff_impl<typename Traits::Differentiation>();
  }

  /**
   * @brief Sets whether to store the Jacobians. If false, Jacobians will be
   * computed dynamically (on-the-fly), which requires manual differentiation.
   * @param store True to store the Jacobians (default mode), false to compute
   * them dynamically.
   */
  virtual void set_jacobian_storage(const bool store) {
    _store_jacobians = store;
  }

  /**
   * @brief Returns whether the Jacobians are stored.
   * @return True if the Jacobians are stored, false otherwise.
   */
  virtual bool store_jacobians() override { return _store_jacobians; }

  /**
   * @brief Determines whether dynamic Jacobian computation is supported (only
   * supported for manual differentiation).
   * @return True if dynamic Jacobian computation is supported, false otherwise.
   */
  virtual bool supports_dynamic_jacobians() override {
    return std::is_same_v<typename Traits::Differentiation,
                          DifferentiationMode::Manual>;
  }

  /**
   * @brief Determines which Hessian blocks are filled in (upper triangle).
   * @param block_coords A vector to be filled with the coordinates of the
   * Hessian blocks that are filled in by this factor descriptor.
   */
  virtual void get_hessian_block_coordinates(
      thrust::device_vector<BlockCoordinates> &block_coords) override {
    const size_t num_vertices = get_num_vertices();
    const auto &active_factors = active_indices;
    const auto ids = device_ids.data().get();
    for (size_t i = 0; i < num_vertices; i++) {
      const auto vi_active = vertex_descriptors[i]->get_active_state();
      const auto vi_block_ids = vertex_descriptors[i]->get_block_ids();
      for (size_t j = i; j < num_vertices; j++) {
        const auto vj_active = vertex_descriptors[j]->get_active_state();
        const auto vj_block_ids = vertex_descriptors[j]->get_block_ids();
        // Iterate over active factors and generate block coordinates
        auto num_coords = block_coords.size();
        block_coords.resize(block_coords.size() + active_factors.size());
        auto end = thrust::transform_if(
            thrust::device, active_factors.begin(), active_factors.end(),
            block_coords.begin() + num_coords,
            [=] __device__(size_t factor_idx) {
              const size_t vi_id = ids[factor_idx * num_vertices + i];
              const size_t vj_id = ids[factor_idx * num_vertices + j];

              const auto block_i = vi_block_ids[vi_id];
              const auto block_j = vj_block_ids[vj_id];
              if (block_i > block_j) {
                return BlockCoordinates{block_j, block_i};
              }
              return BlockCoordinates{block_i, block_j};
            },
            [=] __device__(const size_t &factor_idx) {
              const auto vi_id = ids[factor_idx * num_vertices + i];
              const auto vj_id = ids[factor_idx * num_vertices + j];
              return (is_vertex_active(vi_active, vi_id) &&
                      is_vertex_active(vj_active, vj_id));
            });
        block_coords.resize(end - block_coords.begin());
      }
    }
  }

  /**
   * @brief Sets up data needed to compute the upper triangle of the Hessian
   * matrix on the GPU.
   * @returns The number of multiplications that will be performed by this
   * descriptor.
   */
  virtual size_t setup_hessian_computation(
      std::unordered_map<BlockCoordinates, size_t> &block_indices,
      thrust::device_vector<S> &d_hessian, size_t *h_block_offsets,
      StreamPool &streams) override {

    const size_t num_vertices = get_num_vertices();
    const auto &d_active_factors = active_indices;
    thrust::host_vector<size_t> h_active_factors = active_indices;
    const auto d_ids = device_ids.data().get();
    const auto h_ids = &host_ids[0];

    size_t mul_count = 0;
    // Determine max number of multiplications based on active factors
    for (size_t i = 0; i < num_vertices; i++) {
      for (size_t j = i; j < num_vertices; j++) {
        mul_count += d_active_factors.size();
      }
    }

    size_t write_idx = 0;

    // Then actually do the multiplications
    for (size_t i = 0; i < num_vertices; i++) {
      const auto vi_active = vertex_descriptors[i]->get_active_state();
      const auto vi_block_ids = vertex_descriptors[i]->get_block_ids();
      for (size_t j = i; j < num_vertices; j++) {
        const auto vj_active = vertex_descriptors[j]->get_active_state();
        const auto vj_block_ids = vertex_descriptors[j]->get_block_ids();
        // Iterate over active factors and generate block coordinates
        for (const auto &factor_idx : h_active_factors) {
          // TODO: Build this in the GPU using a GPU hash map
          const auto vi_id = h_ids[factor_idx * num_vertices + i];
          const auto vj_id = h_ids[factor_idx * num_vertices + j];

          if (is_vertex_active(vi_active, vi_id) &&
              is_vertex_active(vj_active, vj_id)) {
            const auto block_i = vi_block_ids[vi_id];
            const auto block_j = vj_block_ids[vj_id];
            BlockCoordinates coordinates{block_i, block_j};
            if (block_i > block_j) {
              coordinates = BlockCoordinates{block_j, block_i};
            }

            auto it = block_indices.find(coordinates);
            if (it != block_indices.end()) {
              const size_t block_offset = it->second;
              h_block_offsets[write_idx++] = block_offset;
            } else {
              // TODO: this should actually be an error, but also impossible
              h_block_offsets[write_idx++] = 0;
              std::cerr << "Error: Hessian block coordinate not found!"
                        << std::endl;
            }
          } else {
            h_block_offsets[write_idx++] = 0;
          }
        }
      }
    }

    return mul_count;
  }

  /**
   * @brief Executes the Hessian block computations for this descriptor using
   * the data from setup_hessian_computation.
   * @returns The number of multiplications that were performed by this
   * descriptor.
   */
  virtual size_t execute_hessian_computation(
      std::unordered_map<BlockCoordinates, size_t> &block_indices,
      thrust::device_vector<S> &d_hessian, size_t *d_block_offsets,
      StreamPool &streams) override {
    const size_t num_vertices = get_num_vertices();
    const auto &d_active_factors = active_indices;
    thrust::host_vector<size_t> h_active_factors = active_indices;
    const auto d_ids = device_ids.data().get();

    size_t mul_count = 0;
    // Determine max number of multiplications based on active factors
    for (size_t i = 0; i < num_vertices; i++) {
      for (size_t j = i; j < num_vertices; j++) {
        mul_count += d_active_factors.size();
      }
    }

    size_t write_idx = 0;

    size_t stream_idx = 0;
    // Then actually do the multiplications
    for (size_t i = 0; i < num_vertices; i++) {
      const auto vi_active = vertex_descriptors[i]->get_active_state();
      const auto vi_block_ids = vertex_descriptors[i]->get_block_ids();
      for (size_t j = i; j < num_vertices; j++) {
        const auto vj_active = vertex_descriptors[j]->get_active_state();
        // Iterate over active factors and generate block coordinates
        const auto start_idx = write_idx;
        write_idx += d_active_factors.size();
        const auto stream = streams.select(stream_idx++);

        const auto dim_i = vertex_descriptors[i]->dimension();
        const auto dim_j = vertex_descriptors[j]->dimension();
        const size_t block_dim = dim_i * dim_j;
        const size_t num_threads = d_active_factors.size() * block_dim;
        const size_t threads_per_block = 256;
        const size_t num_blocks =
            (num_threads + threads_per_block - 1) / threads_per_block;

        ops::compute_hessian_block_kernel<T, S, N, error_dim>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                i, j, dim_i, dim_j, d_active_factors.data().get(),
                d_active_factors.size(), d_ids, d_block_offsets + start_idx,
                vi_active, vj_active, vertex_descriptors[i]->get_hessian_ids(),
                vertex_descriptors[j]->get_hessian_ids(),
                jacobians[i].data.data().get(), jacobians[j].data.data().get(),
                precision_matrices.data().get(), chi2_derivative.data().get(),
                d_hessian.data().get());
      }
    }

    streams.sync_all();

    return mul_count;
  }

  /**
   * @brief Clears all data associated with this factor descriptor.
   */
  void clear() {
    global_ids.clear();
    global_to_local_map.clear();
    local_to_global_map.clear();

    hm.clear();

    _active_count = 0;

    host_ids.clear();
    device_ids.clear();
    device_obs.clear();
    residuals.clear();
    precision_matrices.clear();
    data.clear();

    chi2_vec.clear();
    chi2_derivative.clear();
    loss.clear();

    active.clear();
    device_active.clear();
    active_indices.clear();

    for (size_t i = 0; i < N; i++) {
      jacobians[i].data.clear();
    }
  }

private:
  /**
   * @brief Gets the default precision matrix (identity matrix).
   * @return An array representing the default precision matrix.
   */
  constexpr static std::array<S, error_dim * error_dim>
  get_default_precision_matrix() {
    constexpr size_t E = error_dim;
    return []() constexpr {
      std::array<S, E *E> pmat = {};
      for (size_t i = 0; i < E; i++) {
        pmat[i * E + i] = static_cast<S>(1.0);
      }
      return pmat;
    }
    ();
  }
};

} // namespace graphite