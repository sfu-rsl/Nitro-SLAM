
#pragma once
#include <mutex>
#include <unordered_set>
#include <type_traits>
#include <atomic>
#include <numeric>
#include <fstream>
#include <string>
#include <iterator>
#include <future>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <rh/robin_hood.h>

#include <solver/matrix_types.hpp>

namespace compute
{
    template <typename T>
    class BlockInfo
    {
    private:
        BlockDim dim;
        BlockIndex index;

    public:
        BlockInfo() : dim({0, 0}), index(std::numeric_limits<BlockIndex>::max())
        {
        }

        BlockInfo(const BlockDim &dim, const BlockIndex &index) : dim(dim), index(index) {}

        bool reserved() const
        {
            return index != std::numeric_limits<BlockIndex>::max();
        }

        size_t num_values() const
        {
            // return rows*cols;
            return dim.first * dim.second;
        }

        uint32_t get_index() const
        {
            return static_cast<uint32_t>(index);
        }

        T *calculate_ptr(T *base_ptr) const
        {
            return base_ptr + index;
        }

        size_t mem_size() const
        {
            return num_values() * sizeof(T);
        }

        BlockDim get_dim() const
        {
            return dim;
        }
    };

    class GPUBlockInfo
    {
    public:
        uint32_t row;
        uint32_t col;
        uint32_t offset;

        GPUBlockInfo(const BlockPos &pos, const uint32_t &offset) : row(pos.first), col(pos.second), offset(offset) {}
    };

    using BlockIndexOffset = std::pair<uint32_t, uint32_t>;

    class MatrixBlockData
    {
    public:
        std::vector<uint32_t> indices;
        std::vector<uint32_t> offsets;
        std::vector<uint32_t> ptr;

        void initialize(size_t n)
        {
            ptr = std::vector<uint32_t>(n + 1, 0);
        }
    };

    class MatrixBufferData
    {
    public:
        BufferPtr<uint32_t> indices;
        BufferPtr<uint32_t> offsets;
        BufferPtr<uint32_t> ptr;
    };

    template <typename DataType>
    class SparseBlockMatrix;

    template <typename DataType>
    using MatPtr = std::shared_ptr<SparseBlockMatrix<DataType>>;

    template <typename DataType>
    using ConstMatPtr = std::shared_ptr<const SparseBlockMatrix<DataType>>;

    template <typename DataType>
    class SparseBlockMatrix
    {
    public:
        SparseBlockMatrix(ComputeEngine &engine) : block_buffer_size(0), block_buffer(nullptr), owner(&engine)
        {
        }

        SparseBlockMatrix(const SparseBlockMatrix &) = delete;
        SparseBlockMatrix &operator=(const SparseBlockMatrix &) = delete;

        void clear()
        {
            indices.clear();
            rows.clear();
            cols.clear();
            std::for_each(row_indices.begin(), row_indices.end(), [](auto &i)
                          { i.clear(); });
            std::for_each(col_indices.begin(), col_indices.end(), [](auto &i)
                          { i.clear(); });
            block_buffer_size = 0;
        }

        // Change the number of block-rows and block-columns in the matrix.
        // Warning: Must be called carefully
        void resize(const std::vector<BlockIndex> &rows,
                    const std::vector<BlockIndex> &cols)
        {
            this->rows = rows;
            this->cols = cols;

            row_indices.resize(num_rows());
            col_indices.resize(num_cols());
            std::for_each(row_indices.begin(), row_indices.end(), [](auto &i)
                          { i.clear(); });
            std::for_each(col_indices.begin(), col_indices.end(), [](auto &i)
                          { i.clear(); });
        }

        // Takes duplicates the rows, columns, and indices. Does not allocate memory or copy data.
        void take_structure_from(MatPtr<DataType> src)
        {
            rows = src->rows;
            cols = src->cols;

            indices = src->indices;
            row_indices = src->row_indices;
            col_indices = src->col_indices;
            block_buffer_size = src->block_buffer_size;
            block_dimensions = src->block_dimensions;
        }

        void take_diagonal_structure_from(const MatPtr<DataType> src)
        {

            if (src->num_scalar_rows() != src->num_scalar_cols())
            {
                throw std::runtime_error("Source matrix is not square!");
            }

            // Copy row, col dimensions
            this->rows = src->rows;
            this->cols = src->cols;

            row_indices.resize(src->num_rows());
            col_indices.resize(src->num_cols());

            auto nrows = src->num_rows();

            for (size_t i = 0; i < nrows; i++)
            {
                this->reserve_block(i, i);
            }
        }

        void reserve_block(BlockIndex row, BlockIndex col)
        {
            if (is_allocated())
            {
                throw std::runtime_error("SBM tried to reserve block after allocation!");
            }

            // look up the dimensions
            auto num_rows = rows[row + 1] - rows[row];
            auto num_cols = cols[col + 1] - cols[col];

            // update block record with offset
            BlockPos pos{row, col};
            const BlockIndex num_mat_values = num_rows * num_cols;
            auto it = indices.find(pos);
            if (it == indices.end())
            {
                const auto block_info = BlockInfo<DataType>({num_rows, num_cols}, {block_buffer_size});
                indices[pos] = block_info;

                if (static_cast<uint64_t>(block_buffer_size) + static_cast<uint64_t>(num_mat_values) > std::numeric_limits<BlockIndex>::max())
                {
                    throw std::runtime_error("SBM tried to reserve block outside the range of BlockIndex!");
                }

                block_buffer_size += num_mat_values;
                // add the block to the row/col lists
                row_indices[row].push_back(col);
                col_indices[col].push_back(row);

                // additional information
                block_dimensions.insert(block_info.get_dim());
            }
            else
            {
                // throw std::runtime_error("SBM tried to reserve block which already exists!");
            }
        }

        BufferPtr<DataType> get_buffer()
        {
            return block_buffer;
        }

        bool has_reserved_block(BlockIndex row, BlockIndex col) const
        {
            BlockPos pos{row, col};
            return indices.find(pos) != indices.end();
        }

        // Allocates memory for the reserved blocks
        void allocate_memory(BufferType buffer_type = BufferType::Host)
        {
            if (is_allocated())
            {
                throw std::runtime_error("SBM already has allocated memory!");
            }

            block_buffer = owner->create_buffer<DataType>(nullptr, block_buffer_size, buffer_type);
        }

        uint32_t get_block_offset(BlockIndex row, BlockIndex col) const
        {
            auto it = indices.find({row, col});
            if (it == indices.end() || !it->second.reserved())
            {
                throw std::runtime_error("SBM does not have block!");
            }
            return it->second.get_index();
        }

        DataType *get_block_ptr(BlockIndex row, BlockIndex col)
        {
            auto it = indices.find({row, col});
            if (it == indices.end() || !it->second.reserved())
            {
                return nullptr;
            }

            return it->second.calculate_ptr(block_buffer->map());
        }

        Eigen::Map<Eigen::MatrixXd> get_block_map(BlockIndex row, BlockIndex col)
        {
            auto &block = indices.at({row, col});
            const auto dim = block.get_dim();
            auto ptr = block.calculate_ptr(block_buffer->map());
            return Eigen::Map<Eigen::MatrixXd>(ptr, dim.first, dim.second);
        }

        const Eigen::Map<Eigen::MatrixXd> get_block_map(BlockIndex row, BlockIndex col) const
        {
            auto &block = indices.at({row, col});
            const auto dim = block.get_dim();
            auto ptr = block.calculate_ptr(block_buffer->map());
            return Eigen::Map<Eigen::MatrixXd>(ptr, dim.first, dim.second);
        }

        BlockDim get_block_dim(BlockIndex row, BlockIndex col)
        {
            return {rows.at(row + 1) - rows.at(row), cols.at(col + 1) - cols.at(col)};
        }

        bool is_allocated() const
        {
            return block_buffer != nullptr;
        }

        size_t num_rows() const
        {
            return rows.size() - 1;
        }

        size_t num_cols() const
        {
            return cols.size() - 1;
        }

        size_t num_blocks() const
        {
            return indices.size();
            // auto block_indices = &row_indices;

            // if (num_cols() < num_rows())
            // {
            //     block_indices = &col_indices;
            // }

            // // count
            // size_t num_blocks = 0;
            // for (const auto &list : *block_indices)
            // {
            //     num_blocks += list.size();
            // }
            // return num_blocks;
        }

        BlockIndex num_scalar_rows() const
        {
            return rows.back();
        }

        BlockIndex num_scalar_cols() const
        {
            return cols.back();
        }

        BlockIndex num_non_zeros() const
        {
            return block_buffer_size;
        }

        const std::vector<BlockIndex> &get_rows()
        {
            return rows;
        }

        const std::vector<BlockIndex> &get_cols()
        {
            return cols;
        }

        const std::vector<std::vector<BlockIndex>> &get_row_indices()
        {
            return row_indices;
        }

        const std::vector<std::vector<BlockIndex>> &get_col_indices()
        {
            return col_indices;
        }

        static void rec_multiply(std::vector<MulPair> &global_mul_pairs,
                                 robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> &mul_lists,
                                 BufferPtr<DataType> left, BufferPtr<DataType> right, BufferPtr<DataType> dest,
                                 BufferPtr<uint32_t> lists, BufferPtr<uint32_t> pairs, std::shared_ptr<SolverSeq> seq,
                                 bool add, bool transpose)
        {

            seq->sync_device<uint32_t>({lists, pairs}, {}, true);

            auto pLists = lists->map();
            auto pPairs = pairs->map();

            auto write_ptr = reinterpret_cast<MulPair *>(pPairs);
            std::copy(global_mul_pairs.begin(), global_mul_pairs.end(), write_ptr);
            std::atomic_uint32_t write_idx(0);
            for (auto &list_it : mul_lists)
            {
                auto start_idx = write_idx.fetch_add(static_cast<uint32_t>(list_it.second.size() * 3));
                auto list_write_ptr = reinterpret_cast<MulList *>(pLists + start_idx);
                std::copy(list_it.second.begin(), list_it.second.end(), list_write_ptr);
                // Dispatch
                seq->multiply(list_it.first.first, list_it.first.second, start_idx / 3, list_it.second.size(),
                              left, right, dest, lists, pairs, add, transpose);
                seq->insert_cc_barrier();
            }
        }

        static void rec_multiply_packed(std::vector<MulPair> &global_mul_pairs,
                                        robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> &mul_lists,
                                        BufferPtr<DataType> left, BufferPtr<DataType> right, BufferPtr<DataType> dest,
                                        BufferPtr<uint32_t> lists, BufferPtr<uint32_t> pairs, std::shared_ptr<SolverSeq> seq,
                                        bool add, bool transpose)
        {

            seq->sync_device<uint32_t>({lists, pairs}, {}, true);

            auto pLists = lists->map();
            auto pPairs = pairs->map();

            auto write_ptr = reinterpret_cast<MulPair *>(pPairs);
            std::copy(global_mul_pairs.begin(), global_mul_pairs.end(), write_ptr);
            std::atomic_uint32_t write_idx(0);
            for (auto &list_it : mul_lists)
            {
                auto start_idx = write_idx.fetch_add(static_cast<uint32_t>(list_it.second.size() * 3));
                auto list_write_ptr = reinterpret_cast<MulList *>(pLists + start_idx);
                std::copy(list_it.second.begin(), list_it.second.end(), list_write_ptr);
                // Dispatch
                seq->multiply_packed(list_it.first.first, list_it.first.second, start_idx / 3, list_it.second.size(),
                                     left, right, dest, lists, pairs, add, transpose);
                seq->insert_cc_barrier();
            }
        }

        static void rec_multiply_abat(std::vector<MulPair> &global_mul_pairs,
                                      robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> &mul_lists,
                                      BufferPtr<DataType> left, BufferPtr<DataType> right, BufferPtr<DataType> dest,
                                      BufferPtr<uint32_t> lists, BufferPtr<uint32_t> pairs, std::shared_ptr<SolverSeq> seq,
                                      bool add)
        {

            seq->sync_device<uint32_t>({lists, pairs}, {}, true);

            auto pLists = lists->map();
            auto pPairs = pairs->map();

            auto write_ptr = reinterpret_cast<MulPair *>(pPairs);
            std::copy(global_mul_pairs.begin(), global_mul_pairs.end(), write_ptr);
            std::atomic_uint32_t write_idx(0);
            for (auto &list_it : mul_lists)
            {
                auto start_idx = write_idx.fetch_add(static_cast<uint32_t>(list_it.second.size() * 3));
                auto list_write_ptr = reinterpret_cast<MulList *>(pLists + start_idx);
                std::copy(list_it.second.begin(), list_it.second.end(), list_write_ptr);
                // Dispatch
                seq->multiply_abat(list_it.first.first, list_it.first.second, start_idx / 3, list_it.second.size(),
                                   left, right, dest, lists, pairs, add);
                seq->insert_cc_barrier();
            }
        }

        /*
            Assumptions:
            - Right matrix is square and block-diagonal
            - Destination matrix has structure taken from Left matrix so that buffer offsets are the same

            Warning:
            - Data in destination matrix will be overwritten (optimization)
        */
        void subgroup_block_diagonal_multiply_add(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest, MatPtr<DataType> right, bool add = true)
        {
            if (!(dest->is_allocated() && right->is_allocated() && is_allocated()))
            {
                throw std::runtime_error("SBM multiply_add using unallocated matrices!");
            }

            if (cols != right->rows && right->num_rows() == right->num_cols())
            {
                throw std::runtime_error("SBM multiply_add dimension mismatch!");
            }

            if (!check_multiply_dest_size(dest, right))
            {
                throw std::runtime_error("SBM multiply_add dest has wrong size!");
            }

            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulBlockDiagList>, BlockDimPairHash> mul_lists;
            std::vector<uint32_t> global_mul_pairs;
            global_mul_pairs.reserve(estimate_mul_ops(right));

            size_t total_num_pairs = 0;

            // Go through all the block-diagonal blocks in the right matrix
            for (BlockIndex block_col = 0; block_col < num_cols(); block_col++)
            {
                // look up the block diagonal block
                const auto &dd = right->indices.at({block_col, block_col});

                // multiply this block with every block in the left's block-column
                std::for_each(col_indices[block_col].begin(), col_indices[block_col].end(),
                              [&](const BlockIndex &left_row)
                              {
                                  const auto &left_block = indices.at({left_row, block_col});
                                  // just assume that destination block has same strucuture
                                  local_mul_pairs[{left_block.get_dim(), dd.get_dim()}].push_back(MulPair(left_block.get_index(), dd.get_index()));
                                  total_num_pairs++;
                              });
            }

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = bufPairs; // dummy buffer
            auto pPairs = bufPairs->map();

            // Copy data and record dispatches
            auto write_ptr = reinterpret_cast<MulPair *>(pPairs);

            uint32_t num_written = 0;

            // sync once
            seq->sync_device<uint32_t>({bufPairs}, {}, true);

            for (auto &list_it : local_mul_pairs)
            {
                std::copy(list_it.second.begin(), list_it.second.end(), write_ptr + num_written);
                // Dispatch

                seq->dispatch_right_block_diagonal(list_it.first.first, list_it.first.second, num_written, list_it.second.size(),
                                                   block_buffer, right->get_buffer(), dest->get_buffer(), bufPairs, add);
                // no barrier needed between batches
                num_written += list_it.second.size();
            }
        }

        // dest = (this block diagonal matrix)*right
        void subgroup_block_diagonal_multiply_vec_add(std::shared_ptr<SolverSeq> seq, BufferPtr<DataType> dest, BufferPtr<DataType> right, bool add = true)
        {

            auto total_rows = rows.back();
            auto total_cols = cols.back();

            if (dest->size() != total_rows || right->size() != total_cols)
            {
                throw std::runtime_error("SBM subgroup_block_diagonal_multiply_vec_add dimension mismatch!");
            }

            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> grouped_mul_pairs;
            std::vector<uint32_t> global_mul_pairs;
            global_mul_pairs.reserve(estimate_mul_ops());

            size_t total_num_pairs = 0;

            // Go through all the block-diagonal blocks in the left matrix
            for (BlockIndex block_col = 0; block_col < num_cols(); block_col++)
            {
                // look up the block diagonal block
                const auto &dd = indices.at({block_col, block_col});

                // multiply this block with the corresponding row block in the right vector
                BlockDim vec_dim{dd.get_dim().second, 1};
                auto vec_offset = rows.at(block_col);
                grouped_mul_pairs[{dd.get_dim(), vec_dim}].push_back(MulPair(dd.get_index(), vec_offset));
                total_num_pairs++;
            }

            // auto t1 = std::chrono::high_resolution_clock::now();

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);

            auto pPairs = bufPairs->map();

            // Copy data and record dispatches
            auto write_ptr = reinterpret_cast<MulPair *>(pPairs);
            uint32_t num_written = 0;

            seq->sync_device<uint32_t>({bufPairs}, {}, true);

            for (auto &list_it : grouped_mul_pairs)
            {
                std::copy(list_it.second.begin(), list_it.second.end(), write_ptr + num_written);
                // Dispatch

                seq->dispatch_left_block_diagonal(list_it.first.first, list_it.first.second, num_written, list_it.second.size(),
                                                  block_buffer, right, dest, bufPairs, add);
                // no barrier needed between batches
                num_written += list_it.second.size();
            }
        }

        void subgroup_multiply_add(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest, MatPtr<DataType> right, bool add = true)
        {
            if (!(dest->is_allocated() && right->is_allocated() && is_allocated()))
            {
                throw std::runtime_error("SBM multiply_add using unallocated matrices!");
            }

            if (cols != right->rows)
            {
                throw std::runtime_error("SBM multiply_add dimension mismatch!");
            }

            if (!check_multiply_dest_size(dest, right))
            {
                throw std::runtime_error("SBM multiply_add dest has wrong size!");
            }

            // auto t0 = std::chrono::high_resolution_clock::now();
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> mul_lists;
            std::vector<MulPair> global_mul_pairs;
            global_mul_pairs.reserve(estimate_mul_ops(right));

            size_t total_num_pairs = 0;
            size_t total_num_lists = 0;

            // Go through all the destination blocks (which are known beforehand) and construct the multiplication lists
            for (const auto &dest_it : dest->indices)
            {
                auto dest_index = dest_it.second.get_index();
                auto left_row = dest_it.first.first;
                auto right_col = dest_it.first.second;
                // for each filled in block in the row
                std::for_each(row_indices[left_row].begin(), row_indices[left_row].end(),
                              [&](const BlockIndex &left_col)
                              {
                                  const BlockPos left_block = {left_row, left_col};
                                  const auto &left_it = indices.at(left_block);

                                  auto left_dim = left_it.get_dim();
                                  auto left_index = left_it.get_index();

                                  // look up the corresponding block in the right's column for each right column
                                  const BlockPos right_block = {left_col, right_col};
                                  auto right_it = right->indices.find(right_block);
                                  if (right_it != right->indices.end()) // may not exist
                                  {
                                      // record the left and right pairs
                                      auto right_dim = right_it->second.get_dim();
                                      auto right_index = right_it->second.get_index();
                                      local_mul_pairs[{left_dim, right_dim}].push_back(MulPair(left_index, right_index));
                                  }
                              });
                // now we have found all the pairs for this row-col for all block combinations
                // so we can build the mul list for each block combination
                for (auto &mp_it : local_mul_pairs)
                {
                    // auto & pairs = global_mul_pairs[mp_it.first];
                    auto &pairs = global_mul_pairs;
                    auto list_start = pairs.size();
                    auto list_size = mp_it.second.size();
                    if (list_size > 0)
                    {
                        // copy pairs into global list
                        pairs.insert(pairs.end(), mp_it.second.begin(), mp_it.second.end());
                        // now create the list info
                        mul_lists[mp_it.first].push_back(MulList(list_start, list_size, dest_index));

                        // can clear the pairs for this row-col + BlockDim pair combo
                        total_num_pairs += mp_it.second.size();
                        total_num_lists++;
                        mp_it.second.clear();
                    }
                }
            }

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = owner->create_buffer<uint32_t>(nullptr, total_num_lists * 3, list_buf_type);

            rec_multiply(global_mul_pairs, mul_lists, block_buffer, right->get_buffer(), dest->get_buffer(), bufLists, bufPairs, seq, add, false);
        }

        // dest += this*transpose(right) (seq version)
        void subgroup_transposed_multiply_add(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest, MatPtr<DataType> right, bool add = true, bool sorted = false, bool upper = false)
        {
            if (!(dest->is_allocated() && right->is_allocated() && is_allocated()))
            {
                throw std::runtime_error("SBM transposed_multiply_add using unallocated matrices!");
            }

            if (cols != right->cols)
            {
                throw std::runtime_error("SBM transposed_multiply_add dimension mismatch!");
            }

            if (!check_transposed_multiply_dest_size(dest, right))
            {
                throw std::runtime_error("SBM transposed_multiply_add dest has wrong size!");
            }

            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> mul_lists;

            std::vector<MulPair> global_mul_pairs;
            global_mul_pairs.reserve(estimate_transposed_mul_ops(right));

            size_t total_num_pairs = 0;
            size_t total_num_lists = 0;

            bool both_sorted = sorted;

            // Go through all the destination blocks (which are known beforehand) and construct the multiplication lists
            for (const auto &dest_it : dest->indices)
            {
                const BlockPos &dest_block = dest_it.first;

                if (upper && dest_block.first > dest_block.second)
                {
                    continue;
                }

                auto dest_index = dest_it.second.get_index();
                auto left_row = dest_it.first.first;
                auto right_row = dest_it.first.second;

                if (both_sorted)
                {
                    if (!row_indices[left_row].empty() && !right->row_indices[right_row].empty())
                    {
                        // for each filled in block in the row
                        auto row_start = std::lower_bound(row_indices[left_row].cbegin(), row_indices[left_row].cend(), right->row_indices[right_row].front());
                        auto row_end = row_indices[left_row].end();

                        auto right_end = right->row_indices[right_row].cend();
                        auto right_idx = right->row_indices[right_row].cbegin();
                        for (auto it = row_start; it < row_end && right_idx < right_end;)
                        {
                            const BlockIndex &left_col = *it;
                            if (left_col > right->row_indices[right_row].back())
                            {
                                break;
                            }

                            if (left_col == *right_idx)
                            {

                                // look up the corresponding block in the right's row for each right row
                                const BlockPos right_pos = {right_row, left_col};
                                const BlockPos left_pos = {left_row, left_col};

                                // auto tf0 = std::chrono::high_resolution_clock::now();
                                const auto &right_block = right->indices.at(right_pos);
                                const auto &left_block = indices.at(left_pos);

                                // auto tf1 = std::chrono::high_resolution_clock::now();
                                // tfind += std::chrono::duration<double>(tf1-tf0).count();

                                // record the left and right pairs
                                auto right_dim = right_block.get_dim();
                                // transpose dimensions of b
                                std::swap(right_dim.first, right_dim.second);

                                // get left block info

                                local_mul_pairs[{left_block.get_dim(), right_dim}].push_back(MulPair(left_block.get_index(), right_block.get_index()));

                                it++;
                                right_idx++;
                            }
                            else if (*it < *right_idx)
                            {
                                it++;
                            }
                            else
                            {
                                right_idx++;
                            }
                        }
                    }
                }
                else
                {
                    // SLOW
                    // for each filled in block in the row
                    std::for_each(row_indices[left_row].begin(), row_indices[left_row].end(),
                                  [&](const BlockIndex &left_col)
                                  {
                                      const BlockPos left_block = {left_row, left_col};
                                      const auto &left_it = indices.at(left_block);

                                      auto left_dim = left_it.get_dim();
                                      auto left_index = left_it.get_index();

                                      // look up the corresponding block in the right's row for each right row
                                      const BlockPos right_block = {right_row, left_col};
                                      auto right_it = right->indices.find(right_block);
                                      if (right_it != right->indices.end()) // may not exist
                                      {
                                          // record the left and right pairs
                                          auto right_dim = right_it->second.get_dim();
                                          // transpose dimensions of b
                                          std::swap(right_dim.first, right_dim.second);
                                          auto right_index = right_it->second.get_index();
                                          local_mul_pairs[{left_dim, right_dim}].push_back(MulPair(left_index, right_index));
                                      }
                                  });
                }
                // now we have found all the pairs for this row-col for all block combinations
                // so we can build the mul list for each block combination
                for (auto &mp_it : local_mul_pairs)
                {
                    auto list_size = mp_it.second.size();

                    if (list_size > 0)
                    {
                        auto list_start = global_mul_pairs.size();
                        // copy pairs into global list
                        global_mul_pairs.insert(global_mul_pairs.end(), mp_it.second.begin(), mp_it.second.end());
                        // now create the list info
                        mul_lists[mp_it.first].push_back(MulList(list_start, list_size, dest_index));

                        // can clear the pairs for this row-col + BlockDim pair combo
                        total_num_pairs += mp_it.second.size();
                        total_num_lists++;
                        mp_it.second.clear();
                    }
                }
            }

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = owner->create_buffer<uint32_t>(nullptr, total_num_lists * 3, list_buf_type);

            rec_multiply(global_mul_pairs, mul_lists, block_buffer, right->get_buffer(), dest->get_buffer(), bufLists, bufPairs, seq, add, true);
        }

        void subgroup_transposed_multiply_add2(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest, MatPtr<DataType> right, bool add = true, bool upper = false)
        {
            if (!(dest->is_allocated() && right->is_allocated() && is_allocated()))
            {
                throw std::runtime_error("SBM transposed_multiply_add using unallocated matrices!");
            }

            if (cols != right->cols)
            {
                throw std::runtime_error("SBM transposed_multiply_add dimension mismatch!");
            }

            if (!check_transposed_multiply_dest_size(dest, right))
            {
                throw std::runtime_error("SBM transposed_multiply_add dest has wrong size!");
            }

            // Construct representation for each block size
            auto task_left_map = std::async(std::launch::async, [&]()
                                            { return this->get_csr_bins(); });
            auto task_right_map = std::async(std::launch::async, [&]()
                                             { return right->get_csr_bins(); });

            auto left_map = task_left_map.get();
            auto right_map = task_right_map.get(); // right is transposed so we iterate on its rows

            std::vector<GPUBlockInfo> gpu_block_info;
            gpu_block_info.reserve(dest->indices.size());

            // Go through all the destination blocks (which are known beforehand) and construct the multiplication lists
            for (const auto &dest_it : dest->indices)
            {
                // auto tp0 = std::chrono::high_resolution_clock::now();

                const BlockPos &dest_block = dest_it.first;

                if (upper && dest_block.first > dest_block.second)
                {
                    continue;
                }

                const GPUBlockInfo info(dest_block, dest_it.second.get_index());
                gpu_block_info.push_back(info);
            }

            // Bufferize everything
            auto list_buf_type = get_list_buffer_type();
            auto buf_info = owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(gpu_block_info.data()), gpu_block_info.size() * 3, list_buf_type);

            auto bufferize_bins = [this, list_buf_type](auto &bins)
            {
                robin_hood::unordered_map<BlockDim, MatrixBufferData, BlockDimHash> buffer_map;
                for (auto &it : bins)
                {
                    if (it.second.indices.size() > 0)
                    {
                        auto buf_indices = this->owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(it.second.indices.data()), it.second.indices.size(), list_buf_type);
                        auto buf_offsets = this->owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(it.second.offsets.data()), it.second.offsets.size(), list_buf_type);
                        auto buf_ptr = this->owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(it.second.ptr.data()), it.second.ptr.size(), list_buf_type);
                        buffer_map[it.first] = {buf_indices, buf_offsets, buf_ptr};
                    }
                }
                return buffer_map;
            };

            auto left_buffers = bufferize_bins(left_map);
            auto right_buffers = bufferize_bins(right_map);

            // Sync buffers once
            std::vector<BufferPtr<uint32_t>> to_sync;
            to_sync.push_back(buf_info);

            for (auto &it : left_buffers)
            {
                to_sync.push_back(it.second.indices);
                to_sync.push_back(it.second.offsets);
                to_sync.push_back(it.second.ptr);
            }

            for (auto &it : right_buffers)
            {
                to_sync.push_back(it.second.indices);
                to_sync.push_back(it.second.offsets);
                to_sync.push_back(it.second.ptr);
            }
            seq->sync_device<uint32_t>(to_sync, {}, true);

            // Record dispatches
            for (auto &lit : left_buffers)
            {
                auto dim1 = lit.first;
                for (auto &rit : right_buffers)
                {
                    // dispatch for all destination blocks
                    auto dim2 = rit.first;
                    BlockDim dim2_transposed{dim2.second, dim2.first};
                    if (dim1.second == dim2_transposed.first)
                    {
                        seq->tmultiply(dim1, dim2_transposed, static_cast<uint32_t>(gpu_block_info.size()),
                                       buf_info, dest->block_buffer,
                                       lit.second.indices, lit.second.offsets, lit.second.ptr, this->block_buffer,
                                       rit.second.indices, rit.second.offsets, rit.second.ptr, right->block_buffer,
                                       add, true);
                        seq->insert_cc_barrier();
                    }
                }
            }
        }

        void subgroup_transposed_multiply_add3(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest, MatPtr<DataType> right, bool add = true, bool upper = false)
        {
            if (!(dest->is_allocated() && right->is_allocated() && is_allocated()))
            {
                throw std::runtime_error("SBM transposed_multiply_add using unallocated matrices!");
            }

            if (cols != right->cols)
            {
                throw std::runtime_error("SBM transposed_multiply_add dimension mismatch!");
            }

            if (!check_transposed_multiply_dest_size(dest, right))
            {
                throw std::runtime_error("SBM transposed_multiply_add dest has wrong size!");
            }

            // Bufferize everything
            auto list_buf_type = get_list_buffer_type();

            auto bufferize_bins = [owner = owner, list_buf_type](auto &&bins)
            {
                robin_hood::unordered_map<BlockDim, MatrixBufferData, BlockDimHash> buffer_map;
                for (auto &it : bins)
                {
                    if (it.second.indices.size() > 0)
                    {
                        auto buf_indices = owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(it.second.indices.data()), it.second.indices.size(), list_buf_type);
                        auto buf_offsets = owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(it.second.offsets.data()), it.second.offsets.size(), list_buf_type);
                        auto buf_ptr = owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(it.second.ptr.data()), it.second.ptr.size(), list_buf_type);
                        buffer_map[it.first] = {buf_indices, buf_offsets, buf_ptr};
                    }
                }
                return buffer_map;
            };

            // Construct representation for each block size
            // Most expensive part?
            auto task_left_map = std::async(std::launch::async, [&]()
                                            { return bufferize_bins(this->get_csr_bins()); });
            auto task_right_map = std::async(std::launch::async, [&]()
                                             { return bufferize_bins(right->get_csr_bins()); });

            std::vector<GPUBlockInfo> gpu_block_info;
            gpu_block_info.reserve(dest->indices.size());

            // Go through all the destination blocks (which are known beforehand) and construct the multiplication lists
            for (const auto &dest_it : dest->indices)
            {
                // auto tp0 = std::chrono::high_resolution_clock::now();

                const BlockPos &dest_block = dest_it.first;

                if (upper && dest_block.first > dest_block.second)
                {
                    continue;
                }

                const GPUBlockInfo info(dest_block, dest_it.second.get_index());
                gpu_block_info.push_back(info);
            }
            auto buf_info = owner->create_buffer<uint32_t>(reinterpret_cast<uint32_t *>(gpu_block_info.data()), gpu_block_info.size() * 3, list_buf_type);

            auto left_buffers = task_left_map.get();
            auto right_buffers = task_right_map.get(); // right is transposed so we iterate on its rows

            // Sync buffers once
            std::vector<BufferPtr<uint32_t>> to_sync;
            to_sync.push_back(buf_info);

            for (auto &it : left_buffers)
            {
                to_sync.push_back(it.second.indices);
                to_sync.push_back(it.second.offsets);
                to_sync.push_back(it.second.ptr);
            }

            for (auto &it : right_buffers)
            {
                to_sync.push_back(it.second.indices);
                to_sync.push_back(it.second.offsets);
                to_sync.push_back(it.second.ptr);
            }
            // seq->sync_device<uint32_t>(to_sync, {}, true);

            auto estimate_list_seq = owner->create_op_sequence();
            // sync index buffers, offsets
            estimate_list_seq->sync_device<uint32_t>(to_sync, {}, true);

            // Record dispatches

            // Record dispatches for estimating list and execute
            auto buf_type = get_list_buffer_type();
            robin_hood::unordered_map<BlockDimPair, BufferPtr<uint32_t>, BlockDimPairHash> alloc_map;
            std::vector<BufferPtr<uint32_t>> alloc_buffers;
            std::vector<uint32_t> alloc_zero{0, 0};
            for (auto &lit : left_buffers)
            {
                auto dim1 = lit.first;
                for (auto &rit : right_buffers)
                {
                    // dispatch for all destination blocks
                    auto buf_alloc_info = owner->create_buffer<uint32_t>(alloc_zero.data(), alloc_zero.size(), buf_type);
                    estimate_list_seq->sync_device<uint32_t>({buf_alloc_info}, {}, true);

                    auto dim2 = rit.first;
                    BlockDim dim2_transposed{dim2.second, dim2.first};
                    if (dim1.second == dim2_transposed.first)
                    {
                        estimate_list_seq->estimate_list<DataType>(static_cast<uint32_t>(gpu_block_info.size()),
                                                                   buf_info,
                                                                   lit.second.indices, lit.second.offsets, lit.second.ptr,
                                                                   rit.second.indices, rit.second.offsets, rit.second.ptr,
                                                                   buf_alloc_info);

                        alloc_map[{dim1, dim2_transposed}] = buf_alloc_info;
                        alloc_buffers.push_back(buf_alloc_info);
                    }
                }
            }
            estimate_list_seq->sync_local(alloc_buffers);
            estimate_list_seq->execute(); // TODO: make thread-safe

            // Allocate lists and buffers
            robin_hood::unordered_map<BlockDimPair, std::pair<BufferPtr<uint32_t>, BufferPtr<uint32_t>>, BlockDimPairHash> mul_map;
            for (auto &it : alloc_map)
            {
                const uint32_t num_lists = it.second->map()[0];
                const uint32_t num_pairs = it.second->map()[1];

                if (num_lists > 0 && num_pairs > 0)
                {
                    auto buf_list = owner->create_buffer<uint32_t>(nullptr, num_lists * 3, buf_type);
                    auto buf_pairs = owner->create_buffer<uint32_t>(nullptr, num_pairs * 2, buf_type);

                    mul_map[it.first] = {buf_list, buf_pairs};
                }
            }

            auto build_list_seq = owner->create_op_sequence();

            // Reset allocator buffers
            for (auto &buf : alloc_buffers)
            {
                // reset to zero
                std::memset(buf->map(), 0, buf->mem_size());
                build_list_seq->sync_device<uint32_t>({buf});
            }

            build_list_seq->insert_tc_barrier();

            // Record commands to populate buffers and dispatch multiplications
            for (auto &lit : left_buffers)
            {
                auto dim1 = lit.first;
                for (auto &rit : right_buffers)
                {
                    // dispatch for all destination blocks
                    auto dim2 = rit.first;
                    BlockDim dim2_transposed{dim2.second, dim2.first};
                    if (dim1.second == dim2_transposed.first)
                    {
                        auto it = mul_map.find({dim1, dim2_transposed});
                        if (it != mul_map.end())
                        {

                            auto mul_lists = it->second.first;
                            auto mul_pairs = it->second.second;

                            auto num_blocks = mul_lists->size() / 3;
                            // std::cout << "Num blocks: " << num_blocks << std::endl;
                            // std::cout << "Num pairs: " << mul_pairs->size()/2 << std::endl;

                            build_list_seq->build_list<DataType>(static_cast<uint32_t>(gpu_block_info.size()),
                                                                 buf_info,
                                                                 lit.second.indices, lit.second.offsets, lit.second.ptr,
                                                                 rit.second.indices, rit.second.offsets, rit.second.ptr,
                                                                 mul_lists, mul_pairs,
                                                                 alloc_map[{dim1, dim2_transposed}]);

                            seq->multiply(
                                dim1, dim2_transposed, 0, num_blocks, block_buffer, right->block_buffer,
                                dest->get_buffer(), mul_lists, mul_pairs, add, true);
                            seq->insert_cc_barrier();
                        }
                    }
                }
            }

            // build lists
            build_list_seq->insert_cc_barrier();
            build_list_seq->execute();
        }

        // Compute only the block diagonal of A*B*(A^T) where B is also block diagonal symmetric (Hpl*Hll^-1*Hpl^T)
        void multiply_block_diagonal_self_transpose(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest, MatPtr<DataType> right, bool add = true)
        {
            if (!(dest->is_allocated() && right->is_allocated() && is_allocated()))
            {
                throw std::runtime_error("SBM transposed_multiply_add using unallocated matrices!");
            }

            if (cols != right->cols && rows != dest->rows)
            {
                throw std::runtime_error("SBM transposed_multiply_add dimension mismatch!");
            }

            // if (!check_transposed_multiply_dest_size(dest, right))
            // {
            //     throw std::runtime_error("SBM transposed_multiply_add dest has wrong size!");
            // }

            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> mul_lists;

            std::vector<MulPair> global_mul_pairs;
            global_mul_pairs.reserve(estimate_transposed_mul_ops(right));

            size_t total_num_pairs = 0;
            size_t total_num_lists = 0;

            // Go through all the destination blocks (which are known beforehand) and construct the multiplication lists
            for (size_t i = 0; i < dest->num_rows(); i++)
            {
                const auto &dest_block = dest->indices.at({i, i});

                const auto dest_index = dest_block.get_index();
                auto left_row = i;
                // auto right_row = i;

                // for each filled in block in the row
                std::for_each(row_indices[left_row].begin(), row_indices[left_row].end(),
                              [&](const BlockIndex &left_col)
                              {
                                  const auto &left_block = indices.at({left_row, left_col});
                                  const auto &right_block = right->indices.at({left_col, left_col});

                                  local_mul_pairs[{left_block.get_dim(), right_block.get_dim()}].push_back(MulPair(left_block.get_index(), right_block.get_index()));
                              });
                // now we have found all the pairs for this row-col for all block combinations
                // so we can build the mul list for each block combination
                for (auto &mp_it : local_mul_pairs)
                {
                    auto list_size = mp_it.second.size();

                    if (list_size > 0)
                    {
                        auto list_start = global_mul_pairs.size();
                        // copy pairs into global list
                        global_mul_pairs.insert(global_mul_pairs.end(), mp_it.second.begin(), mp_it.second.end());
                        // now create the list info
                        mul_lists[mp_it.first].push_back(MulList(list_start, list_size, dest_index));

                        // can clear the pairs for this row-col + BlockDim pair combo
                        total_num_pairs += mp_it.second.size();
                        total_num_lists++;
                        mp_it.second.clear();
                    }
                }
            }

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = owner->create_buffer<uint32_t>(nullptr, total_num_lists * 3, list_buf_type);

            // Record commands to dispatch multiplications
            rec_multiply_abat(
                global_mul_pairs, mul_lists, block_buffer, right->block_buffer, dest->block_buffer, bufLists, bufPairs, seq, add);
        }

        void subgroup_multiply_vec_add(std::shared_ptr<SolverSeq> seq, BufferPtr<DataType> dest, BufferPtr<DataType> right, bool add = true)
        {

            auto total_rows = rows.back();
            auto total_cols = cols.back();

            if (dest->size() != total_rows || right->size() != total_cols)
            {
                // std::cout << "dest size: " << dest->size() << std::endl;
                // std::cout << "total rows: " << total_rows << std::endl;
                // std::cout << "right size: " << right->size() << std::endl;

                throw std::runtime_error("SBM subgroup_multiply_vec_add dimension mismatch!");
            }

            // auto t0 = std::chrono::high_resolution_clock::now();
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> mul_lists;
            std::vector<MulPair> global_mul_pairs;
            global_mul_pairs.reserve(estimate_mul_ops());

            size_t total_num_pairs = 0;
            size_t total_num_lists = 0;

            // Go through all the destination blocks (which are known beforehand) and construct the multiplication lists
            uint32_t offset = 0;
            for (size_t left_row = 0; left_row < num_rows(); left_row++)
            {
                // for each filled in block in the row
                std::for_each(row_indices[left_row].begin(), row_indices[left_row].end(),
                              [&](const BlockIndex &left_col)
                              {
                                  const BlockPos left_block = {left_row, left_col};
                                  const auto &left_it = indices.at(left_block);

                                  auto left_dim = left_it.get_dim();
                                  auto left_index = left_it.get_index();

                                  // look up the corresponding block in the right's column for each right column
                                  // record the left and right pairs
                                  //   auto right_dim = BlockDim(rows[left_row + 1] - rows[left_row], 1);
                                  //   auto right_index = offset;
                                  auto right_dim = BlockDim(left_dim.second, 1);
                                  uint32_t right_index = cols[left_col];
                                  local_mul_pairs[{left_dim, right_dim}].push_back(MulPair(left_index, right_index));
                              });
                // now we have found all the pairs for this row-col for all block combinations
                // so we can build the mul list for each block combination
                for (auto &mp_it : local_mul_pairs)
                {
                    // auto & pairs = global_mul_pairs[mp_it.first];
                    auto &pairs = global_mul_pairs;
                    auto list_start = pairs.size();
                    auto list_size = mp_it.second.size();
                    if (list_size > 0)
                    {
                        // copy pairs into global list
                        pairs.insert(pairs.end(), mp_it.second.begin(), mp_it.second.end());
                        // now create the list info
                        mul_lists[mp_it.first].push_back(MulList(list_start, list_size, offset));

                        // can clear the pairs for this row-col + BlockDim pair combo
                        total_num_pairs += mp_it.second.size();
                        total_num_lists++;
                        mp_it.second.clear();
                    }
                }
                offset += rows.at(left_row + 1) - rows.at(left_row);
            }

            // auto t1 = std::chrono::high_resolution_clock::now();

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = owner->create_buffer<uint32_t>(nullptr, total_num_lists * 3, list_buf_type);

            // auto pPairs = bufPairs->map();
            // auto pLists = bufLists->map();
            // auto mult = owner->sbm_create_group_multiplication<DataType>(block_buffer, right, dest, bufLists, bufPairs);

            // Copy data and record dispatches
            rec_multiply(
                global_mul_pairs, mul_lists, block_buffer, right, dest, bufLists, bufPairs, seq, add, false);
            // auto t2 = std::chrono::high_resolution_clock::now();
            // fmt::print("{}\n", std::chrono::duration<double>(t1-t0).count());
            // fmt::print("{}\n", std::chrono::duration<double>(t2 - t1).count());
            // fmt::print("{}\n", std::chrono::duration<double>(td - tc).count());
            // return mult;
        }

        void multiply_vec_add_upper(std::shared_ptr<SolverSeq> seq, BufferPtr<DataType> dest, BufferPtr<DataType> right, bool add = true)
        {

            auto total_rows = rows.back();
            auto total_cols = cols.back();

            if (dest->size() != total_rows || right->size() != total_cols)
            {
                // std::cout << "dest size: " << dest->size() << std::endl;
                // std::cout << "total rows: " << total_rows << std::endl;
                // std::cout << "right size: " << right->size() << std::endl;

                throw std::runtime_error("SBM subgroup_multiply_vec_add dimension mismatch!");
            }

            // auto t0 = std::chrono::high_resolution_clock::now();
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> mul_lists;
            std::vector<MulPair> global_mul_pairs;
            global_mul_pairs.reserve(estimate_mul_ops());

            size_t total_num_pairs = 0;
            size_t total_num_lists = 0;

            // Go through all the destination blocks (which are known beforehand) and construct the multiplication lists
            uint32_t offset = 0;
            for (size_t left_row = 0; left_row < num_rows(); left_row++)
            {
                // for each filled in block in the row
                std::for_each(row_indices[left_row].begin(), row_indices[left_row].end(),
                              [&](const BlockIndex &left_col)
                              {
                                  if (left_col >= left_row)
                                  {
                                      const BlockPos left_block = {left_row, left_col};
                                      const auto &left_it = indices.at(left_block);

                                      auto left_dim = left_it.get_dim();
                                      auto left_index = left_it.get_index();

                                      // look up the corresponding block in the right's column for each right column
                                      // record the left and right pairs
                                      //   auto right_dim = BlockDim(rows[left_row + 1] - rows[left_row], 1);
                                      //   auto right_index = offset;
                                      auto right_dim = BlockDim(left_dim.second, 1);
                                      uint32_t right_index = cols[left_col];
                                      local_mul_pairs[{left_dim, right_dim}].push_back(MulPair(left_index, right_index));
                                  }
                              });
                // now we have found all the pairs for this row-col for all block combinations
                // so we can build the mul list for each block combination
                for (auto &mp_it : local_mul_pairs)
                {
                    // auto & pairs = global_mul_pairs[mp_it.first];
                    auto &pairs = global_mul_pairs;
                    auto list_start = pairs.size();
                    auto list_size = mp_it.second.size();
                    if (list_size > 0)
                    {
                        // copy pairs into global list
                        pairs.insert(pairs.end(), mp_it.second.begin(), mp_it.second.end());
                        // now create the list info
                        mul_lists[mp_it.first].push_back(MulList(list_start, list_size, offset));

                        // can clear the pairs for this row-col + BlockDim pair combo
                        total_num_pairs += mp_it.second.size();
                        total_num_lists++;
                        mp_it.second.clear();
                    }
                }
                offset += rows.at(left_row + 1) - rows.at(left_row);
            }

            // auto t1 = std::chrono::high_resolution_clock::now();

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = owner->create_buffer<uint32_t>(nullptr, total_num_lists * 3, list_buf_type);

            // Copy data and record dispatches
            rec_multiply(
                global_mul_pairs, mul_lists,
                block_buffer, right, dest,
                bufLists, bufPairs,
                seq, add, false);
        }

        void multiply_vec_lower_no_diagonal(std::shared_ptr<SolverSeq> seq, BufferPtr<DataType> dest, BufferPtr<DataType> vec, bool add = true)
        {

            auto total_rows = rows.back();
            auto total_cols = cols.back();
            if (dest->size() != total_cols || vec->size() != total_rows)
            {
                throw std::runtime_error("SBM multiply_vec_lower_no_diagonal dimension mismatch!");
            }

            // auto t0 = std::chrono::high_resolution_clock::now();
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> mul_lists;
            std::vector<MulPair> global_mul_pairs;
            global_mul_pairs.reserve(estimate_mul_ops());

            size_t total_num_pairs = 0;
            size_t total_num_lists = 0;

            // Vector is 1xn and Matrix is nxm.
            // Go through each matrix column and multiply.
            // Iterating through rows may be better if # block-rows << # block-columns.
            uint32_t offset = 0;

            for (uint32_t block_col = 0; block_col < num_cols(); block_col++)
            {
                std::for_each(col_indices[block_col].begin(), col_indices[block_col].end(),
                              [&](const BlockIndex &block_row)
                              {
                                  if (block_col > block_row)
                                  {
                                      const BlockPos matrix_block = {block_row, block_col};
                                      const auto &matrix_it = indices.at(matrix_block);

                                      auto matrix_dim = matrix_it.get_dim();
                                      auto matrix_index = matrix_it.get_index();

                                      // figure out the write offset in the destination vector
                                      // which should correspond to the column of the block
                                      auto vec_dim = BlockDim(1, matrix_dim.first);
                                      auto vec_index = rows.at(block_row);
                                      local_mul_pairs[{vec_dim, matrix_dim}].push_back(MulPair(vec_index, matrix_index));
                                  }
                              });

                // create lists
                for (auto &mp_it : local_mul_pairs)
                {
                    auto &pairs = global_mul_pairs;
                    auto list_start = pairs.size();
                    auto list_size = mp_it.second.size();

                    if (list_size > 0)
                    {
                        // copy pairs into global list
                        pairs.insert(pairs.end(), mp_it.second.begin(), mp_it.second.end());
                        // now create the list info
                        mul_lists[mp_it.first].push_back(MulList(list_start, list_size, offset));

                        // can clear the pairs for this row-col + BlockDim pair combo
                        total_num_pairs += mp_it.second.size();
                        total_num_lists++;
                        mp_it.second.clear();
                    }
                }
                offset += cols.at(block_col + 1) - cols.at(block_col);
            }

            // auto t1 = std::chrono::high_resolution_clock::now();

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();

            if (total_num_pairs == 0)
            {
                return;
            }

            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = owner->create_buffer<uint32_t>(nullptr, total_num_lists * 3, list_buf_type);

            // matrix and vector buffers need to be swapped
            // Copy data and record dispatches
            rec_multiply(
                global_mul_pairs, mul_lists, vec, this->get_buffer(), dest, bufLists, bufPairs, seq, add, false);
        }

        void subgroup_right_multiply_vec_add(std::shared_ptr<SolverSeq> seq, BufferPtr<DataType> dest, BufferPtr<DataType> vec, bool add = true)
        {

            auto total_rows = rows.back();
            auto total_cols = cols.back();
            if (dest->size() != total_cols || vec->size() != total_rows)
            {
                throw std::runtime_error("SBM subgroup_right_multiply_vec_add dimension mismatch!");
            }

            // auto t0 = std::chrono::high_resolution_clock::now();
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulPair>, BlockDimPairHash> local_mul_pairs;
            robin_hood::unordered_map<std::pair<BlockDim, BlockDim>, std::vector<MulList>, BlockDimPairHash> mul_lists;
            std::vector<MulPair> global_mul_pairs;
            global_mul_pairs.reserve(estimate_mul_ops());

            size_t total_num_pairs = 0;
            size_t total_num_lists = 0;

            // Vector is 1xn and Matrix is nxm.
            // Go through each matrix column and multiply.
            // Iterating through rows may be better if # block-rows << # block-columns.
            uint32_t offset = 0;

            for (uint32_t block_col = 0; block_col < num_cols(); block_col++)
            {
                std::for_each(col_indices[block_col].begin(), col_indices[block_col].end(),
                              [&](const BlockIndex &block_row)
                              {
                                  const BlockPos matrix_block = {block_row, block_col};
                                  const auto &matrix_it = indices.at(matrix_block);

                                  auto matrix_dim = matrix_it.get_dim();
                                  auto matrix_index = matrix_it.get_index();

                                  // figure out the write offset in the destination vector
                                  // which should correspond to the column of the block
                                  auto vec_dim = BlockDim(1, matrix_dim.first);
                                  auto vec_index = rows.at(block_row);
                                  local_mul_pairs[{vec_dim, matrix_dim}].push_back(MulPair(vec_index, matrix_index));
                              });

                // create lists
                for (auto &mp_it : local_mul_pairs)
                {
                    auto &pairs = global_mul_pairs;
                    auto list_start = pairs.size();
                    auto list_size = mp_it.second.size();

                    if (list_size > 0)
                    {
                        // copy pairs into global list
                        pairs.insert(pairs.end(), mp_it.second.begin(), mp_it.second.end());
                        // now create the list info
                        mul_lists[mp_it.first].push_back(MulList(list_start, list_size, offset));

                        // can clear the pairs for this row-col + BlockDim pair combo
                        total_num_pairs += mp_it.second.size();
                        total_num_lists++;
                        mp_it.second.clear();
                    }
                }
                offset += cols.at(block_col + 1) - cols.at(block_col);
            }

            // auto t1 = std::chrono::high_resolution_clock::now();

            // Create the buffers and record commands
            auto list_buf_type = get_list_buffer_type();
            auto bufPairs = owner->create_buffer<uint32_t>(nullptr, total_num_pairs * 2, list_buf_type);
            auto bufLists = owner->create_buffer<uint32_t>(nullptr, total_num_lists * 3, list_buf_type);

            // matrix and vector buffers need to be swapped in constructor

            // Copy data and record dispatches
            rec_multiply_packed(
                global_mul_pairs, mul_lists, vec, this->get_buffer(), dest, bufLists, bufPairs, seq, add, false);
        }

        // Record operations to set lambda for block diagonal of a matrix
        void set_lambda(BufferPtr<DataType> lambda, std::shared_ptr<SolverSeq> set_seq, std::shared_ptr<SolverSeq> restore_seq)
        {

            robin_hood::unordered_map<uint32_t, std::vector<uint32_t>> offsets;
            // std::vector<uint32_t> offsets;
            // offsets.reserve(num_rows());
            size_t total_num_offsets = 0;
            size_t backup_buffer_size = 0;

            for (size_t i = 0; i < num_rows(); i++)
            {
                // offsets.push_back(indices.at({i, i}).get_index());
                const auto &block = indices.at({i, i});
                offsets[block.get_dim().first].push_back(block.get_index());
                total_num_offsets++;
                backup_buffer_size += block.get_dim().first;
            }

            auto list_buf_type = get_list_buffer_type();

            auto offsets_buffer = owner->create_buffer<uint32_t>(nullptr, total_num_offsets, list_buf_type);
            auto backup_buffer = owner->create_buffer<DataType>(nullptr, backup_buffer_size, list_buf_type);

            set_seq->sync_device<uint32_t>({offsets_buffer}, {}, true); // only sync once, assume restore_seq will not run before
            // assume that lambda has already been sync'd to device memory at this point

            uint32_t id_offset = 0;
            uint32_t backup_offset = 0;
            for (const auto &it : offsets)
            {
                uint32_t size = it.second.size();
                std::copy(it.second.begin(), it.second.end(), offsets_buffer->map() + id_offset);

                set_seq->set_lambda(it.first, block_buffer, backup_buffer, offsets_buffer, lambda, size, id_offset, backup_offset);
                restore_seq->restore_lambda(it.first, block_buffer, backup_buffer, offsets_buffer, lambda, size, id_offset, backup_offset);

                id_offset += size;
                backup_offset += size * it.first;
            }
        }

        void copy_blocks_into(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest)
        {

            robin_hood::unordered_map<BlockDim, std::vector<MulPair>, BlockDimHash> offsets;
            // std::vector<uint32_t> offsets;
            // offsets.reserve(num_rows());
            size_t total_num_offsets = 0;

            /*
            for (size_t i = 0; i < num_rows(); i++) {
                const auto & block = indices.at({i, i});
                const auto & dest_block = dest->indices.at({i, i});

                offsets[block.get_dim()].push_back(MulPair(block.get_index(), dest_block.get_index()));
                total_num_offsets++;
            }
            */

            for (auto &it : indices)
            {
                const auto &block = it.second;
                const auto dim = block.get_dim();
                const auto dest_block = dest->indices.at(it.first);
                const auto dest_dim = dest_block.get_dim();
                if (dest_dim != dim)
                {
                    throw std::runtime_error("Destination dimension does not match source block!");
                }
                offsets[dest_dim].push_back(MulPair(block.get_index(), dest_block.get_index()));
                total_num_offsets++;
            }

            auto offsets_buffer = owner->create_buffer<uint32_t>(nullptr, total_num_offsets * 2, get_list_buffer_type());

            seq->sync_device<uint32_t>({offsets_buffer}, {}, true);

            uint32_t id_offset = 0;
            for (const auto &it : offsets)
            {
                uint32_t size = it.second.size();
                std::copy(it.second.begin(), it.second.end(), reinterpret_cast<MulPair *>(offsets_buffer->map()) + id_offset);
                seq->copy_blocks(it.first, block_buffer, dest->get_buffer(), offsets_buffer, size, id_offset);
                id_offset += size;
            }
        }

        void copy_diagonal_blocks_into(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest)
        {

            robin_hood::unordered_map<BlockDim, std::vector<MulPair>, BlockDimHash> offsets;
            size_t total_num_offsets = 0;

            for (size_t i = 0; i < num_rows(); i++)
            {
                const auto &block = indices.at({i, i});
                const auto dim = block.get_dim();
                const auto &dest_block = dest->indices.at({i, i});

                const auto dest_dim = dest_block.get_dim();
                if (dest_dim != dim)
                {
                    throw std::runtime_error("Destination dimension does not match source block!");
                }
                offsets[dest_dim].push_back(MulPair(block.get_index(), dest_block.get_index()));
                total_num_offsets++;
            }

            auto offsets_buffer = owner->create_buffer<uint32_t>(nullptr, total_num_offsets * 2, get_list_buffer_type());

            seq->sync_device<uint32_t>({offsets_buffer}, {}, true);

            uint32_t id_offset = 0;
            for (const auto &it : offsets)
            {
                uint32_t size = it.second.size();
                std::copy(it.second.begin(), it.second.end(), reinterpret_cast<MulPair *>(offsets_buffer->map()) + id_offset);
                seq->copy_blocks(it.first, block_buffer, dest->get_buffer(), offsets_buffer, size, id_offset);
                id_offset += size;
            }
        }

        void create_inversion_op(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> dest)
        {
            // check that destination is allocated
            robin_hood::unordered_map<Dim, std::vector<uint32_t>> offsets;
            // std::vector<MappedPair<DataType>> invert_on_cpu;
            for (auto &it : indices)
            {
                if (it.first.first != it.first.second)
                {
                    throw std::runtime_error("SBM cpu_block_diagonal_inversion: Block is not on diagonal!");
                }
                const auto dim = it.second.get_dim();

                if (dim.first != dim.second)
                {
                    throw std::runtime_error("SBM cpu_block_diagonal_inversion: Diagonal block is not square!");
                }

                if (dim.first < 5)
                {
                    // send to shader
                    offsets[dim.first].push_back(it.second.get_index());
                }
                else
                {
                    throw std::runtime_error("Inverting matrices larger than 4x4 on GPU is not currently supported!\n");
                }

                // auto m = Eigen::Map<Eigen::Matrix3d>(it.second.calculate_ptr(ptr));
                // m = m.inverse().eval();
            }

            // dispatch
            seq->invert_block_diagonal<DataType>(dest->block_buffer, block_buffer, offsets);
        }

        void copy_blocks_into(MatPtr<DataType> dest) const
        {
            auto ptr = block_buffer->map();
            auto dest_ptr = dest->block_buffer->map();
            for (auto &it : indices)
            {
                const auto dim = it.second.get_dim();
                auto dest_block = dest->indices.at(it.first);
                auto dest_dim = dest_block.get_dim();
                auto m1 = Eigen::Map<Eigen::MatrixXd>(it.second.calculate_ptr(ptr), dim.first, dim.second);
                auto m2 = Eigen::Map<Eigen::MatrixXd>(dest_block.calculate_ptr(dest_ptr), dest_dim.first, dest_dim.second);
                m2 = m1;
            }
        }

        // Inversion is done on the CPU
        void copy_inverse_diagonal_blocks_into(MatPtr<DataType> dest) const
        {
            auto ptr = block_buffer->map();
            auto dest_ptr = dest->block_buffer->map();
            for (auto &dest_it : dest->indices)
            {

                auto &dest_block = dest_it.second;
                auto dest_dim = dest_block.get_dim();

                auto src_block = indices.at(dest_it.first);
                const auto dim = src_block.get_dim();

                auto m1 = Eigen::Map<Eigen::MatrixXd>(src_block.calculate_ptr(ptr), dim.first, dim.second);
                auto m2 = Eigen::Map<Eigen::MatrixXd>(dest_block.calculate_ptr(dest_ptr), dest_dim.first, dest_dim.second);
                m2 = m1.inverse();
            }
        }

        bool debug_check_symmetry()
        {
            for (auto &it : indices)
            {
                BlockPos pos2 = it.first;
                std::swap(pos2.first, pos2.second);
                if (indices.find(pos2) == indices.end())
                {
                    return false;
                }
            }
            return true;
        }

        void copy_buffer_into(MatPtr<DataType> dest) const
        {
            if (block_buffer->size() != dest->block_buffer->size())
            {
                throw std::runtime_error("SBM copy_buffer_into: src and dest buffer size mismatch!");
            }

            // TODO: Faster to do this on the GPU?
            auto src = block_buffer->map();
            std::copy(src, src + block_buffer->size(), dest->block_buffer->map());
        }

        // Returns true if destination has correct rows and columns.
        // Does not check reserved block layout
        bool check_multiply_dest_size(ConstMatPtr<DataType> dest, ConstMatPtr<DataType> right) const
        {
            return rows == dest->rows && right->cols == dest->cols;
        }

        bool check_transposed_multiply_dest_size(ConstMatPtr<DataType> dest, ConstMatPtr<DataType> right) const
        {
            return rows == dest->rows && right->rows == dest->cols;
        }

        // Sort only the column indices within each row
        void sort_row_indices()
        {
            std::for_each(row_indices.begin(), row_indices.end(), [](std::vector<BlockIndex> &indices)
                          { std::sort(indices.begin(), indices.end()); });
        }

        // Sort only the row indices within each column
        void sort_col_indices()
        {
            std::for_each(col_indices.begin(), col_indices.end(), [](std::vector<BlockIndex> &indices)
                          { std::sort(indices.begin(), indices.end()); });
        }

        // Sorts both row and column indices
        void sort_by_index()
        {
            sort_row_indices();
            sort_col_indices();
        }

        // cheap overestimation for allocating mul ops buffer
        size_t estimate_mul_ops(ConstMatPtr<DataType> right) const
        {
            std::atomic_size_t count(0);
            //    std::vector<BlockIndex> i(col_indices.size());
            //    std::iota(i.begin(), i.end(), 0);
            // std::for_each(i.begin(), i.end(), [&](BlockIndex & index) {
            for (size_t index = 0; index < col_indices.size(); index++)
            {
                count += col_indices.at(index).size() * right->row_indices.at(index).size();
                // });
            }
            return count;
        }

        size_t estimate_transposed_mul_ops(ConstMatPtr<DataType> right) const
        {
            std::atomic_size_t count(0);
            for (size_t index = 0; index < col_indices.size(); index++)
            {
                count += col_indices.at(index).size() * right->col_indices.at(index).size();
            }
            return count;
        }

        size_t estimate_mul_ops() const
        {
            std::atomic_size_t count(0);
            for (size_t index = 0; index < col_indices.size(); index++)
            {
                count += col_indices.at(index).size();
            }
            return count;
        }

        // Convert this block matrix into CSC format
        // Warning: Column indices must be sorted before calling this function!
        // TODO: Investigate if this needs to be done by the GPU instead
        Eigen::SparseMatrix<DataType, Eigen::ColMajor> to_csc(const bool upper_triangular = false) const
        {

            // Precompute row offsets

            // std::vector<BlockIndex> row_offsets = rows;
            // BlockIndex ro = 0;
            // for (int row = 0; row < row_offsets.size(); row++)
            // {
            //     row_offsets[row] = ro;
            //     ro += rows[row+1] - rows[row];
            // }

            // Eigen::SparseMatrix<DataType, Eigen::ColMajor> csc;
            // look up blocks and copy into new matrix
            std::vector<int> outerIndices = {0};
            std::vector<int> innerIndices;
            std::vector<DataType> values;
            values.reserve(block_buffer_size);

            int num_rows = rows.back();
            int num_cols = 0;

            // fill in the indices and values
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {

                const auto cols_in_block = cols[col + 1] - cols[col];
                // fill in the column
                for (size_t c = 0; c < cols_in_block; c++)
                {

                    int num_values_in_col = 0;
                    std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                                  {
                                  // for each actual column in the block column
                                  const auto map = get_block_map(row, col);

                                  // store each value in the block column
                                  for (auto r = 0; r < map.rows(); r++)
                                  {
                                      auto actual_row = rows[row] + r;
                                      if (upper_triangular && actual_row > outerIndices.size() - 1) {
                                          break;
                                      }
                                      num_values_in_col++;
                                      values.push_back(map(r, c));
                                      // also need to push back the actual row
                                      innerIndices.push_back(actual_row);
                                  } });
                    outerIndices.push_back(outerIndices[outerIndices.size() - 1] + num_values_in_col);
                }
                num_cols += cols_in_block;
            }

            // return the mapped matrix which should be copied
            return Eigen::Map<Eigen::SparseMatrix<DataType, Eigen::ColMajor>>(
                num_rows,
                num_cols,
                values.size(),
                outerIndices.data(),
                innerIndices.data(),
                values.data());
        }

        Eigen::SparseMatrix<DataType, Eigen::ColMajor> to_csc2(const bool upper_triangular = false) const
        {
            Eigen::SparseMatrix<DataType, Eigen::ColMajor> matrix;
            fill_csc2(matrix, upper_triangular);
            return matrix;
        }

        // Fill
        void fill_csc(Eigen::SparseMatrix<DataType, Eigen::ColMajor> &dest, bool upper_triangular = false) const
        {
            // Reserve triplets  (row, col, value)
            std::vector<Eigen::Triplet<DataType>> tripletList;
            tripletList.reserve(block_buffer_size); // upper bound

            // for each block column
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {
                // determine how many real columns wide this is
                const auto cols_in_block = cols[col + 1] - cols[col];

                // for each block in the block column
                std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                              {
                    const auto map = get_block_map(row, col);
                    // fill in all columns
                    for (size_t c = 0; c < cols_in_block; c++)
                    {
                        auto actual_col = c + cols[col];
                        auto actual_row = rows[row];
                        // store each value in the block column
                        for (auto r = 0; r < map.rows(); r++)
                        {
                            if (upper_triangular && actual_row > actual_col) {
                                break;
                            }
                            tripletList.push_back(Eigen::Triplet<DataType>(actual_row, actual_col, map(r, c)));
                            actual_row++;
                        } 
                    } });
            }

            // end of collection
            dest.setFromTriplets(tripletList.begin(), tripletList.end());
        }

        // Fill
        void fill_csc_in_order(Eigen::SparseMatrix<DataType, Eigen::ColMajor> &dest, bool upper_triangular = false) const
        {

            std::vector<Eigen::Triplet<DataType>> tripletList;
            tripletList.reserve(block_buffer_size);

            // collect triplets

            int32_t num_rows = rows.back();
            int32_t num_cols = 0;

            // fill in the indices and values
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {

                const auto cols_in_block = cols[col + 1] - cols[col];
                // fill in the column
                for (size_t c = 0; c < cols_in_block; c++)
                {

                    int num_values_in_col = 0;
                    std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                                  {
                                  // for each actual column in the block column
                                  const auto map = get_block_map(row, col);

                                  // store each value in the block column
                                  for (auto r = 0; r < map.rows(); r++)
                                  {
                                      auto actual_row = rows[row] + r;
                                      if (upper_triangular && actual_row > num_cols) {
                                          break;
                                      }
                                      num_values_in_col++;
                                      // push back the triplet
                                      tripletList.push_back(Eigen::Triplet<DataType>(actual_row, num_cols, map(r, c)));
                                  } });
                    num_cols++;
                }
                // num_cols += cols_in_block;
            }

            // end of collection

            dest.setFromTriplets(tripletList.begin(), tripletList.end());
        }

        // Assumes this matrix already has sorted column indices
        void fill_csc2(Eigen::SparseMatrix<DataType, Eigen::ColMajor> &dest, const bool upper_triangular = false) const
        {

            // resize dest
            dest.resize(this->num_scalar_rows(), this->num_scalar_cols());
            dest.resizeNonZeros(this->num_non_zeros());

            // int num_cols = 0;
            size_t nnz = 0;
            DataType *val_ptr = dest.valuePtr();
            int32_t *col_ptr = dest.outerIndexPtr();
            int32_t *row_indices = dest.innerIndexPtr();
            *col_ptr = 0;

            // fill in the indices and values for each block col
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {
                auto actual_col = cols[col];
                const auto cols_in_block = cols[col + 1] - actual_col;
                // fill in the column
                for (size_t c = 0; c < cols_in_block; c++)
                {
                    int32_t num_values_in_col = 0;
                    std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                                  {
                                  // for each actual column in the block column
                                  const auto map = get_block_map(row, col);

                                  // store each value in the block column
                                  auto row_start = rows[row];
                                  for (auto r = 0; r < map.rows(); r++)
                                  {
                                      auto actual_row = row_start + r;
                                      if (upper_triangular && actual_row > actual_col) {
                                          break;
                                      }
                                      num_values_in_col++;
                                      *val_ptr = map(r, c);
                                      val_ptr++;
                                      *row_indices = actual_row;
                                      row_indices++;
                                      nnz++;
                                  } });

                    col_ptr[actual_col + 1] = col_ptr[actual_col] + num_values_in_col;
                    actual_col++;
                }
            }
            // std::cout << "Total nnz: " << nnz << std::endl;
        }

        void fill_csc_values2(DataType *val_ptr, const bool upper_triangular = false) const
        {

            // resize dest

            // fill in the indices and values for each block col
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {
                auto actual_col = cols[col];
                const auto cols_in_block = cols[col + 1] - actual_col;
                // fill in the column
                for (size_t c = 0; c < cols_in_block; c++)
                {
                    std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                                  {
                                  // for each actual column in the block column
                                  const auto map = get_block_map(row, col);

                                  // store each value in the block column
                                  auto row_start = rows[row];
                                  for (auto r = 0; r < map.rows(); r++)
                                  {
                                      auto actual_row = row_start + r;
                                      if (upper_triangular && actual_row > actual_col) {
                                          break;
                                      }
                                      *val_ptr = map(r, c);
                                      val_ptr++;;
                                  } });
                    actual_col++;
                }
            }
        }

        void fill_block_pattern(Eigen::SparseMatrix<DataType, Eigen::ColMajor> &dest, const bool upper_triangular = false) const
        {

            // resize dest
            dest.resize(this->num_rows(), this->num_cols());
            dest.resizeNonZeros(this->num_blocks());

            // int num_cols = 0;
            int32_t *col_ptr = dest.outerIndexPtr();
            int32_t *row_indices = dest.innerIndexPtr();
            *col_ptr = 0;

            // fill in the indices and values for each block col
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {
                // auto actual_col = cols[col];
                // const auto cols_in_block = cols[col + 1] - actual_col;
                int32_t num_blocks_in_col = 0;
                // fill in the column
                std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                              {
                    if ((upper_triangular && row <= col) || !upper_triangular) {
                        *row_indices = row;
                        row_indices++;  
                        num_blocks_in_col++;
                    } });
                col_ptr[col + 1] = col_ptr[col] + num_blocks_in_col;
            }
        }

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t> get_ordered_permutation(const bool upper_triangular = false) const
        {
            // Order blocks
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t> perm;
            Eigen::SparseMatrix<double, Eigen::ColMajor> pattern;
            fill_block_pattern(pattern, upper_triangular);
            Eigen::AMDOrdering<int32_t> amd_ordering;
            amd_ordering(pattern, perm);

            // Construct scalar permutation matrix from ordered block permutation then analyze
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t> perm2(num_scalar_rows());
            // Eigen::VectorXi indices(num_scalar_rows());
            size_t idx = 0;
            for (size_t i = 0; i < num_cols(); i++)
            {
                auto perm_block_idx = perm.indices()[i];
                const auto col_start = cols[perm_block_idx];
                const auto col_end = cols[perm_block_idx + 1];
                for (uint32_t j = col_start; j < col_end; j++)
                {
                    perm2.indices()[idx++] = j;
                    // perm2(idx)
                    // indices(idx++) = j;
                }
            }
            // perm2.indices() = indices;
            return perm2;
        }

        // Fill values only
        void fill_csc_values(DataType *values, bool upper_triangular = false) const
        {

            // collect triplets
            size_t num_values = 0;

            // fill in the indices and values
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {

                const auto cols_in_block = cols[col + 1] - cols[col];
                // fill in the column
                for (size_t c = 0; c < cols_in_block; c++)
                {

                    std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                                  {
                                      // for each actual column in the block column
                                      const auto map = get_block_map(row, col);
                                      //   const auto block_ptr = get_block_ptr(row, col);

                                      // copy each value in each row of the column to the values array
                                      auto num_copy = map.rows();
                                      if (upper_triangular && rows.at(row) == cols.at(col))
                                      {
                                          num_copy = c + 1;
                                      }
                                      auto src = map.data() + c * map.rows();
                                      std::copy(src, src + num_copy, values + num_values);

                                      num_values += num_copy; });
                }
            }
        }

        void init_dense(Eigen::MatrixXd &dest, const bool upper_triangular = false) const
        {
            // resize dest
            dest.resize(this->num_scalar_rows(), this->num_scalar_cols());
            dest.setZero();
            fill_dense(dest, upper_triangular);
        }

        void fill_dense(Eigen::MatrixXd &dest, const bool upper_triangular = false) const
        {

            // fill in the indices and values for each block col
            for (BlockIndex col = 0; col < col_indices.size(); col++)
            {
                auto actual_col = cols[col];
                const auto cols_in_block = cols[col + 1] - actual_col;
                // fill in the column
                for (size_t c = 0; c < cols_in_block; c++)
                {
                    std::for_each(col_indices[col].begin(), col_indices[col].end(), [&](const BlockIndex &row)
                                  {
                                  // for each actual column in the block column
                                  const auto map = get_block_map(row, col);

                                  // store each value in the block column
                                  auto row_start = rows[row];
                                  for (auto r = 0; r < map.rows(); r++)
                                  {
                                      auto actual_row = row_start + r;
                                      if (upper_triangular && actual_row > actual_col) {
                                          break;
                                      }
                                      dest(actual_row, actual_col) = map(r, c);                    
                                  } });

                    actual_col++;
                }
            }
            // std::cout << "Total nnz: " << nnz << std::endl;
        }

        // TODO: Investigate if clearing the buffer via GPU API is faster
        void zero_memory()
        {
            DataType *data = block_buffer->map();
            std::fill(data, data + block_buffer->size(), 0.0);
        }

        void loadFromFile(const char *path, BufferType buffer_type = BufferType::Host)
        {
            std::ifstream file(path);

            clear();
            std::string line;
            // read row sizes
            {
                std::getline(file, line);
                std::istringstream is(line);
                rows = std::vector<uint32_t>(std::istream_iterator<uint32_t>(is), std::istream_iterator<uint32_t>());
            }

            // read col sizes
            {
                std::getline(file, line);
                std::istringstream is(line);
                cols = std::vector<uint32_t>(std::istream_iterator<uint32_t>(is), std::istream_iterator<uint32_t>());
            }

            resize(rows, cols);

            // reserve blocks
            while (std::getline(file, line))
            {
                // read position
                uint32_t row, col;
                std::stringstream ss(line);
                ss >> row;
                ss >> col;
                // reserve block
                reserve_block(row, col);
                // skip line of values
                std::getline(file, line);
            }

            // allocate
            allocate_memory(buffer_type);

            // read values and write
            file.clear();
            file.seekg(0);
            // skip first 2 lines
            std::getline(file, line);
            std::getline(file, line);

            while (std::getline(file, line))
            {
                // read position
                uint32_t row, col;
                std::stringstream ss(line);
                ss >> row;
                ss >> col;

                std::getline(file, line);
                std::istringstream is(line);
                auto start = std::istream_iterator<double>(is);
                auto end = std::istream_iterator<double>();
                double *ptr = get_block_ptr(row, col);
                for (auto it = start; it != end; it++)
                {
                    *ptr = *it;
                    ptr++;
                }
            }
        }

    private:
        robin_hood::unordered_map<BlockPos, BlockInfo<DataType>, BlockPosHash> indices;
        std::vector<BlockIndex> rows;
        std::vector<BlockIndex> cols;
        std::vector<std::vector<BlockIndex>> row_indices;
        std::vector<std::vector<BlockIndex>> col_indices;
        robin_hood::unordered_set<BlockDim, BlockDimHash> block_dimensions;

        BlockIndex block_buffer_size;
        BufferPtr<DataType> block_buffer;
        ComputeEngine *owner;

        // private functions

        robin_hood::unordered_map<BlockDim, MatrixBlockData, BlockDimHash> get_csr_bins() const
        {
            // this->sort_row_indices();
            std::vector<std::vector<uint32_t>> sorted_indices = this->row_indices;

            std::for_each(sorted_indices.begin(), sorted_indices.end(), [](std::vector<BlockIndex> &indices)
                          { std::sort(indices.begin(), indices.end()); });
            // Initialize bmap
            robin_hood::unordered_map<BlockDim, MatrixBlockData, BlockDimHash> bmap;
            for (const auto &dim : block_dimensions)
            {
                bmap[dim].initialize(num_rows());
            }

            for (size_t i = 0; i < sorted_indices.size(); i++)
            {
                for (const auto &dim : block_dimensions)
                {
                    auto &bdata = bmap.at(dim);
                    bdata.ptr[i + 1] = bdata.ptr[i];
                }
                for (const auto &j : sorted_indices[i])
                {
                    const auto &block = indices.at({i, j});
                    auto &bdata = bmap.at(block.get_dim());
                    bdata.indices.push_back(j);
                    bdata.offsets.push_back(block.get_index());
                    bdata.ptr[i + 1]++;
                }
            }

            // debug: check
            /*
            for (auto & it: bmap) {
                std::cout << "BlockDim: " << it.first.first << ", " << it.first.second << std::endl;
                std::cout << "ptrs: ";
                for (const auto& p: it.second.ptr) {
                    std::cout << p << ", ";
                }
                std::cout << std::endl;
                std::cout << "indices: ";
                for (const auto & idx: it.second.indices) {
                    std::cout << "(" << idx.first << "," << idx.second << "), ";
                }
                std::cout << std::endl;
            }
            */
            return bmap;
        }

        /*
        robin_hood::unordered_map<BlockDim, MatrixBlockData, BlockDimHash> get_csc_bins() {
            this->sort_col_indices();
            // Initialize bmap
            robin_hood::unordered_map<BlockDim, MatrixBlockData, BlockDimHash> bmap;
            for (const auto & dim: block_dimensions) {
                bmap[dim].initialize(num_cols());
            }

            for (size_t j = 0; j < col_indices.size(); j++) {
                for (const auto & dim: block_dimensions) {
                    auto & bdata = bmap.at(block.get_dim());
                    bdata.ptr[j+1] = bdata.ptr[j];
                }
                for (const auto & i: col_indices[j]) {
                    const auto & block = indices.at({i, j});
                    auto & bdata = bmap.at(block.get_dim());
                    bdata.indices.push_back({i, block.get_index()});
                    bdata.ptr[j+1]++;
                }
            }
            return bmap;
        }
        */

        BufferType get_list_buffer_type() const
        {
            if (!is_allocated())
            {
                throw std::runtime_error("Cannot get list buffer type for unallocated matrix!");
            }
            if (block_buffer->get_buffer_type() == BufferType::Storage)
            {
                return BufferType::DeviceCached; // safest type to return
            }
            return block_buffer->get_buffer_type();
        }
    };

}
