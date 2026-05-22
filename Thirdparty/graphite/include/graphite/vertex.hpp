/// @file vertex.hpp
#pragma once
#include <graphite/active.hpp>
#include <graphite/common.hpp>
#include <graphite/ops/hessian.hpp>
#include <graphite/ops/state.hpp>
#include <graphite/ops/update.hpp>
#include <graphite/stream.hpp>
#include <graphite/vector.hpp>
#include <type_traits>

namespace graphite {

template <typename T, typename S> class BaseVertexDescriptor {
public:
  using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

  virtual ~BaseVertexDescriptor(){};

  // virtual void update(const T* x, const T* delta) = 0;
  virtual void apply_update_async(const T *delta_x, T *jacobian_scales,
                                  cudaStream_t stream) = 0;
  virtual void augment_block_diagonal_async(InvP *block_diagonal,
                                            InvP *scalar_diagonal, T mu,
                                            cudaStream_t stream) = 0;
  virtual void apply_block_jacobi(T *z, const T *r, InvP *block_diagonal,
                                  cudaStream_t stream) = 0;

  virtual size_t dimension() const = 0;
  virtual size_t count() const = 0;

  virtual void backup_parameters_async() = 0;
  virtual void restore_parameters_async() = 0;
  virtual void to_device() = 0;

  virtual const std::unordered_map<size_t, size_t> &get_global_map() const = 0;
  virtual const size_t *get_hessian_ids() const = 0;
  virtual void set_hessian_column(const size_t global_id,
                                  const size_t hessian_column,
                                  const size_t block_index) = 0;
  virtual bool is_fixed(const size_t id) const = 0;
  virtual bool is_active(const size_t id) const = 0;
  virtual uint8_t *get_active_state() const = 0;
  virtual const size_t *get_block_ids() const = 0;
};

/**
 * @brief Represents a collection of optimizable variables to be processed
 * together on the GPU.
 */
template <typename T, typename S, typename VTraits>
class VertexDescriptor : public BaseVertexDescriptor<T, S> {
public:
  using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

  using Traits = VTraits;

  using VertexType = typename Traits::Vertex;
  using State = get_State_or_t<Traits, VertexType>;

private:
  thrust::device_vector<VertexType *> x_device;
  thrust::host_vector<VertexType *> x_host;
  thrust::device_vector<State> backup_state;

public:
  // Mappings
  std::unordered_map<size_t, size_t> global_to_local_map;
  std::vector<size_t> local_to_global_map;
  thrust::host_vector<size_t> local_to_hessian_offsets;
  thrust::device_vector<size_t> hessian_ids;
  managed_vector<size_t> block_ids;
  managed_vector<uint8_t> active_state;

  static constexpr size_t dim = Traits::dimension;

public:
  virtual ~VertexDescriptor(){};

  /**
   * @brief Applies an update to the vertex parameters asynchronously.
   */
  void apply_update_async(const T *delta_x, T *jacobian_scales,
                          cudaStream_t stream) override {
    ops::apply_update<T, S>(this, delta_x, jacobian_scales, stream);
  }

  /**
   * @brief Adds damping to the Hessian block diagonal asynchronously.
   */
  void augment_block_diagonal_async(InvP *block_diagonal, InvP *scalar_diagonal,
                                    T mu, cudaStream_t stream) override {
    ops::augment_block_diagonal<T, S>(this, block_diagonal, scalar_diagonal, mu,
                                      stream);
  }

  /**
   * @brief Applies the block Jacobi preconditioner asynchronously.
   */
  void apply_block_jacobi(T *z, const T *r, InvP *block_diagonal,
                          cudaStream_t stream) override {
    ops::apply_block_jacobi<T, S>(this, z, r, block_diagonal, stream);
  }

  /**
   * @brief Prepares descriptor for GPU processing.
   */
  virtual void to_device() override {
    x_device = x_host;
    hessian_ids = local_to_hessian_offsets;
  }

  /**
   * @brief Returns a pointer to the array of vertex pointers for the
   * descriptor.
   */
  VertexType **vertices() { return x_device.data().get(); }

  /**
   * @brief Backs up the state of the vertices asynchronously.
   */
  virtual void backup_parameters_async() override {
    VertexType **vertices = x_device.data().get();

    const int num_vertices = static_cast<int>(count());
    const int num_threads = num_vertices;
    const int block_size = 256;
    const auto num_blocks = (num_threads + block_size - 1) / block_size;
    backup_state.resize(num_vertices);

    ops::backup_state_kernel<VertexType, State, Traits, T>
        <<<num_blocks, block_size>>>(vertices, backup_state.data().get(),
                                     active_state.data().get(), num_vertices);
  }

  /**
   * @brief Restores the state of the vertices from the backup asynchronously.
   */
  virtual void restore_parameters_async() override {
    VertexType **vertices = x_device.data().get();

    const int num_vertices = static_cast<int>(count());
    const int num_threads = num_vertices;
    const int block_size = 256;
    const auto num_blocks = (num_threads + block_size - 1) / block_size;
    ops::set_state_kernel<VertexType, State, Traits, T>
        <<<num_blocks, block_size>>>(vertices, backup_state.data().get(),
                                     active_state.data().get(), num_vertices);
  }

  /**
   * @brief Returns the number of vertices in the descriptor (including fixed
   * vertices).
   */
  virtual size_t count() const override { return x_host.size(); }

  /**
   * @brief Reserves memory for the specified number of vertices. You should
   * call this before constructing the graph to avoid excessive reallocations.
   * @param size The number of vertices to reserve space for.
   */
  void reserve(size_t size) {
    x_host.reserve(size);
    global_to_local_map.reserve(size);
    local_to_global_map.reserve(size);
    local_to_hessian_offsets.reserve(size);
    block_ids.reserve(size);
    active_state.reserve(size);
  }

  /**
   * @brief Removes a vertex from the descriptor by its ID.
   * @param id The ID of the vertex to remove.
   */
  void remove_vertex(const size_t id) {
    if (count() == 0) {
      return;
    }

    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Vertex with id " << id << " not found." << std::endl;
      return;
    }

    const auto local_id = global_to_local_map[id];
    const auto last_index = x_host.size() - 1;

    // Swap the vertex to be removed with the last vertex
    std::swap(x_host[local_id], x_host[last_index]);

    if (local_to_hessian_offsets.size() > 0) { // may not be initialized yet
      std::swap(local_to_hessian_offsets[local_id],
                local_to_hessian_offsets[last_index]);
    }

    // Update the global_to_local_map for the swapped vertex
    const auto last_global_id = local_to_global_map[last_index];
    global_to_local_map[last_global_id] = local_id;
    local_to_global_map[local_id] = last_global_id;

    // Only need to update the fixed mask for the swapped vertex
    active_state[local_id] = active_state[last_index];

    // Remove unused entry
    active_state.pop_back();

    // Remove the last vertex
    x_host.pop_back();
    local_to_hessian_offsets.pop_back();
    global_to_local_map.erase(id);
    local_to_global_map.pop_back();
    block_ids.pop_back();
  }

  /**
   * @brief Replaces a vertex pointer in the descriptor for the given ID.
   * @param id The ID of the vertex to replace.
   * @param vertex The new pointer of the vertex.
   */
  void replace_vertex(const size_t id, VertexType *vertex) {
    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Vertex with id " << id << " not found." << std::endl;
      return;
    }

    const auto local_id = global_to_local_map[id];
    x_host[local_id] = vertex;
  }

  /**
   * @brief Adds a vertex to the descriptor.
   * @param id The ID of the vertex to add. Must be unique across all vertex
   * descriptors in the same graph.
   * @param vertex The pointer of the vertex to add.
   * @param fixed Whether the vertex is fixed or not.
   */
  void add_vertex(const size_t id, VertexType *vertex,
                  const bool fixed = false) {
    x_host.push_back(vertex);
    const auto local_id = x_host.size() - 1;
    global_to_local_map.insert({id, local_id});
    local_to_global_map.push_back(id);
    local_to_hessian_offsets.push_back(0); // Initialize to 0
    block_ids.push_back(0);                // Initialize to 0

    // Update fixed mask
    active_state.push_back(static_cast<uint8_t>(fixed));
  }

  /**
   * @brief Sets the fixed state of a vertex.
   * @param id The ID of the vertex to update.
   * @param fixed Whether the vertex is fixed (true) or not (false).
   */
  void set_fixed(const size_t id, const bool fixed) {
    const auto local_id = global_to_local_map.at(id);
    // Don't preserve MSB flag
    active_state[local_id] = static_cast<uint8_t>(fixed);
  }

  /**
   * @brief Checks if a vertex is fixed.
   * @param id The ID of the vertex to check.
   * @return True if the vertex is fixed, false otherwise.
   */
  bool is_fixed(const size_t id) const override {
    const auto local_id = global_to_local_map.at(id);
    return (active_state[local_id] & 0x1) > 0;
  }

  /**
   * @brief Checks if a vertex is active.
   * @param id The ID of the vertex to check.
   * @return True if the vertex is active, false otherwise.
   */
  bool is_active(const size_t id) const override {
    const auto local_id = global_to_local_map.at(id);
    return is_vertex_active(active_state.data().get(), local_id);
  }

  /**
   * @brief Retrieves a pointer to the active state array for the vertices.
   * @return A pointer to the active state array.
   */
  uint8_t *get_active_state() const override {
    return active_state.data().get();
  }

  /**
   * @brief Retrieves a pointer to the block IDs array for the vertices.
   * @return A pointer to the block IDs array.
   */
  const size_t *get_block_ids() const override {
    return block_ids.data().get();
  }

  /**
   * @brief Retrieves a pointer to the vertex associated with the given ID.
   * @param id The ID of the vertex to retrieve.
   * @return A pointer to the vertex if found, or nullptr if the vertex does not
   * exist in the descriptor.
   */
  VertexType *get_vertex(const size_t id) {
    auto it = global_to_local_map.find(id);
    if (it != global_to_local_map.end()) {
      return x_host[it->second];
    } else {
      std::cerr << "Vertex with id " << id << " not found." << std::endl;
      return nullptr;
    }
  }

  /**
   * @brief Checks if a vertex exists in the descriptor.
   * @param id The ID of the vertex to check.
   * @return True if the vertex exists, false otherwise.
   */
  bool exists(const size_t id) const {
    return global_to_local_map.find(id) != global_to_local_map.end();
  }

  const std::unordered_map<size_t, size_t> &get_global_map() const override {
    return global_to_local_map;
  }

  /**
   * @brief Returns the dimension of the vertex parameterization.
   * @return The dimension of the vertex parameterization.
   */
  size_t dimension() const override { return dim; }

  const size_t *get_hessian_ids() const override {
    return hessian_ids.data().get();
  }

  /**
   * @brief Sets the Hessian column and block index for a vertex.
   * @param global_id The global ID of the vertex.
   * @param hessian_column The Hessian column index.
   * @param block_index The block index.
   */
  void set_hessian_column(const size_t global_id, const size_t hessian_column,
                          const size_t block_index) {
    const auto local_id = global_to_local_map.at(global_id);
    local_to_hessian_offsets[local_id] = hessian_column;
    block_ids[local_id] = block_index;
  }

  /**
   * @brief Clears all vertices and resets the descriptor to an empty state.
   */
  void clear() {
    x_device.clear();
    x_host.clear();
    backup_state.clear();

    global_to_local_map.clear();
    local_to_global_map.clear();
    local_to_hessian_offsets.clear();
    hessian_ids.clear();
    block_ids.clear();
    active_state.clear();
  }
};

} // namespace graphite