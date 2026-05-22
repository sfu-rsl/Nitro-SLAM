#pragma once
#include <kompute/Manager.hpp>
#include <kompute/Algorithm.hpp>
#include <kompute/operations/OpTensorCopy.hpp>
#include <numeric>
#include <algorithm>
#include <kompute/operations/OpTensorSyncDevice.hpp>
#include <kompute/operations/OpTensorSyncLocal.hpp>
#include <solver/kp/CustomOp.hpp>
#include <rh/robin_hood.h>

namespace compute
{
    template <typename T>
    T ceil_div(const T &n, const T &d)
    {
        return (n + d - 1) / d;
    }

    struct ArrayInfo
    {
        uint32_t n;
    };
    // Provides functionality for recording and executing solver operations on the GPU
    class VulkanSolverSeq
    {
    private:
        std::shared_ptr<kp::Sequence> seq;
        std::shared_ptr<kp::Sequence> init_seq;
        VulkanComputeEngine *engine;
        bool init;
        bool init_ops; // indicates if init sequence recorded operations

    public:
        VulkanSolverSeq(std::shared_ptr<kp::Sequence> sequence, std::shared_ptr<kp::Sequence> init_seq, VulkanComputeEngine *engine) : seq(sequence), init_seq(init_seq), engine(engine), init(true), init_ops(false)
        {
        }

        void clear() {
            seq->clear();
            init_seq->clear();
            init = true;
            init_ops = false;
        }

        template <typename DataType>
        void inner_product(VCBPtr<DataType> v1, VCBPtr<DataType> v2, VCBPtr<DataType> out1, VCBPtr<DataType> out2)
        {
            // Create algorithm for these buffers
            // shouldn't need lock for manager because algorithm is unmanaged
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 256;
            const uint32_t wgx = ceil_div<uint32_t>(v1->size(), local_size_x);
            kp::Workgroup wg({wgx, 1, 1});

            std::vector<int32_t> spec({local_size_x});

            std::vector<std::shared_ptr<kp::Tensor>> tensors({v1->get_tensor(), v2->get_tensor(), out1->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, ArrayInfo>(tensors,
                                                                                    engine->spv_multiply_vec,
                                                                                    wg,
                                                                                    spec,
                                                                                    {ArrayInfo{static_cast<uint32_t>(v1->size())}});

            seq->record<kp::OpAlgoDispatch>(algo);
            insert_cc_barrier();
            reduction(out1, out2);
        }

        template <typename DataType>
        void self_inner_product(VCBPtr<DataType> in, VCBPtr<DataType> out1, VCBPtr<DataType> out2)
        {
            // Create algorithm for these buffers
            // shouldn't need lock for manager because algorithm is unmanaged
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 256;
            const uint32_t wgx = ceil_div<uint32_t>(in->size(), local_size_x);
            kp::Workgroup wg({wgx, 1, 1});

            std::vector<int32_t> spec({local_size_x});

            std::vector<std::shared_ptr<kp::Tensor>> tensors({in->get_tensor(), out1->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, ArrayInfo>(tensors,
                                                                                    engine->spv_square_array,
                                                                                    wg,
                                                                                    spec,
                                                                                    {ArrayInfo{static_cast<uint32_t>(in->size())}});

            seq->record<kp::OpAlgoDispatch>(algo);
            insert_cc_barrier();
            reduction(out1, out2);
        }

        // Input buffer will be modified, final result is stored in output buffer
        template <typename DataType>
        void reduction(VCBPtr<DataType> in, VCBPtr<DataType> out)
        {
            // Create algorithm for these buffers
            // shouldn't need lock for manager because algorithm is unmanaged
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 256;

            // Provide number of subgroups as a compile-time constant for shared mem array size

            const int32_t subgroups_per_workgroup = ceil_div<int32_t>(local_size_x, engine->get_subgroup_size());

            std::vector<int32_t> spec({local_size_x, subgroups_per_workgroup});

            std::vector<std::shared_ptr<kp::Tensor>> tensors({in->get_tensor(), out->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, ArrayInfo>(tensors,
                                                                                    engine->spv_reduction,
                                                                                    {},
                                                                                    spec,
                                                                                    {ArrayInfo{static_cast<uint32_t>(in->size())}});

            std::shared_ptr<kp::Algorithm> rev_algo = mgr.algorithm<int32_t, ArrayInfo>({out->get_tensor(), in->get_tensor()},
                                                                                        engine->spv_reduction,
                                                                                        {},
                                                                                        spec,
                                                                                        {ArrayInfo{static_cast<uint32_t>(in->size())}});

            // record command for each level of reduction
            uint32_t n = in->size(); // number of elements at each level of reduction
            uint32_t max_iter = 4;
            bool last_dest_out = false;
            while (n > 1)
            {
                // Update number of workgroups
                const uint32_t wgx = ceil_div<uint32_t>(n, local_size_x * max_iter);

                // Record dispatch
                ArrayInfo pc{n};
                algo->setPushConstants(&pc, 1, sizeof(pc));
                algo->setWorkgroup(kp::Workgroup{wgx, 1, 1});
                seq->record<kp::OpAlgoDispatch>(algo);
                insert_cc_barrier();
                last_dest_out = !last_dest_out;
                // update n
                n = wgx;

                if (n > 1)
                {
                    // reduce back into input buffer
                    const uint32_t wgx2 = ceil_div<uint32_t>(n, local_size_x * max_iter);
                    ArrayInfo pc2{n};
                    rev_algo->setPushConstants(&pc2, 1, sizeof(pc2));
                    rev_algo->setWorkgroup(kp::Workgroup{wgx2, 1, 1});
                    seq->record<kp::OpAlgoDispatch>(rev_algo);

                    // Pipeline barrier (before next compute dispatch)
                    insert_cc_barrier();
                    n = wgx2;
                    last_dest_out = !last_dest_out;
                }
            }

            if (!last_dest_out)
            {
                const uint32_t wgx = ceil_div<uint32_t>(n, local_size_x * max_iter);
                // Record dispatch
                ArrayInfo pc{n};
                algo->setPushConstants(&pc, 1, sizeof(pc));
                algo->setWorkgroup(kp::Workgroup{wgx, 1, 1});
                seq->record<kp::OpAlgoDispatch>(algo);
                // insert_cc_barrier();
                last_dest_out = !last_dest_out;
            }
        }

        template <typename DataType>
        void copy_vec(VCBPtr<DataType> src, VCBPtr<DataType> dest, uint32_t size = std::numeric_limits<uint32_t>::max())
        {
            // Create algorithm for these buffers
            const uint32_t buf_size = std::min(static_cast<uint32_t>(src->size()), size);
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 1024;

            // Provide number of subgroups as a compile-time constant for shared mem array size
            int32_t subgroups_per_workgroup = local_size_x / engine->get_subgroup_size();
            if (subgroups_per_workgroup * engine->get_subgroup_size() < static_cast<uint32_t>(local_size_x))
            {
                subgroups_per_workgroup++;
            }

            std::vector<int32_t> spec({local_size_x, subgroups_per_workgroup});
            std::vector<std::shared_ptr<kp::Tensor>> tensors({src->get_tensor(), dest->get_tensor()});

            std::shared_ptr<kp::Algorithm> copy_algo = mgr.algorithm<int32_t, ArrayInfo>(tensors,
                                                                                         engine->spv_copy_vec,
                                                                                         {},
                                                                                         spec,
                                                                                         {ArrayInfo{buf_size}});

            // record command for each level of reduction
            copy_algo->setWorkgroup(kp::Workgroup{buf_size / local_size_x + 1, 1, 1});
            seq->record<kp::OpAlgoDispatch>(copy_algo);
        }

        template <typename DataType>
        void fill_vec(VCBPtr<DataType> dest, const DataType value)
        {
            // Create algorithm for these buffers
            // shouldn't need lock for manager because algorithm is unmanaged
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 1024;

            // Provide number of subgroups as a compile-time constant for shared mem array size
            int32_t subgroups_per_workgroup = local_size_x / engine->get_subgroup_size();
            if (subgroups_per_workgroup * engine->get_subgroup_size() < static_cast<uint32_t>(local_size_x))
            {
                subgroups_per_workgroup++;
            }

            std::vector<int32_t> spec({local_size_x, subgroups_per_workgroup});
            std::vector<std::shared_ptr<kp::Tensor>> tensors({dest->get_tensor()});

            struct FillInfo
            {
                DataType a;
                uint32_t size;
            };

            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, FillInfo>(tensors,
                                                                                   engine->spv_fill_vec,
                                                                                   {},
                                                                                   spec,
                                                                                   {FillInfo{value, static_cast<uint32_t>(dest->size())}});

            // record command for each level of reduction
            uint32_t n = dest->size(); // number of elements at each level of reduction
            // ArrayInfo pc2{n};
            // copy_algo->setPushConstants(&pc2, 1, sizeof(pc2));
            algo->setWorkgroup(kp::Workgroup{n / local_size_x + 1, 1, 1});
            seq->record<kp::OpAlgoDispatch>(algo);
        }

        // w = u+a*v
        template <typename DataType>
        void add_vec(VCBPtr<DataType> u, VCBPtr<DataType> v, VCBPtr<DataType> w, DataType a)
        {
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 256;
            uint32_t wgx = u->size() / (local_size_x);
            if (wgx * local_size_x < u->size())
            {
                wgx++;
            }
            kp::Workgroup wg({wgx, 1, 1});

            std::vector<int32_t> spec({local_size_x});

            struct AddVecInfo
            {
                DataType a;
                uint32_t size;
            };

            std::vector<std::shared_ptr<kp::Tensor>> tensors({u->get_tensor(), v->get_tensor(), w->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, AddVecInfo>(tensors,
                                                                                     engine->spv_add_vec,
                                                                                     wg,
                                                                                     spec,
                                                                                     {AddVecInfo{a, static_cast<uint32_t>(u->size())}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

        // w = u+a*v
        template <typename DataType>
        void add_vec(VCBPtr<DataType> u, VCBPtr<DataType> v, VCBPtr<DataType> w, VCBPtr<DataType> a, bool add = true)
        {
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 256;
            uint32_t wgx = u->size() / (local_size_x);
            if (wgx * local_size_x < u->size())
            {
                wgx++;
            }
            kp::Workgroup wg({wgx, 1, 1});

            std::vector<int32_t> spec({local_size_x, add ? 1 : 0});

            struct AddVecInfo
            {
                uint32_t size;
            };

            std::vector<std::shared_ptr<kp::Tensor>> tensors({u->get_tensor(), v->get_tensor(), w->get_tensor(), a->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, AddVecInfo>(tensors,
                                                                                     engine->spv_add_vec_const_vec,
                                                                                     wg,
                                                                                     spec,
                                                                                     {AddVecInfo{static_cast<uint32_t>(u->size())}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void div_vec(VCBPtr<DataType> u, VCBPtr<DataType> v,
                     VCBPtr<DataType> w, uint32_t size = std::numeric_limits<uint32_t>::max())
        {
            kp::Manager &mgr = *(engine->get_manager());
            uint32_t buf_size = std::min(static_cast<uint32_t>(u->size()), size);

            int32_t local_size_x = 256;
            uint32_t wgx = buf_size / (local_size_x);
            if (wgx * local_size_x < buf_size)
            {
                wgx++;
            }
            kp::Workgroup wg({wgx, 1, 1});

            std::vector<int32_t> spec({local_size_x});

            struct AddVecInfo
            {
                uint32_t size;
            };

            std::vector<std::shared_ptr<kp::Tensor>> tensors({u->get_tensor(), v->get_tensor(), w->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, AddVecInfo>(tensors,
                                                                                     engine->spv_div_vec,
                                                                                     wg,
                                                                                     spec,
                                                                                     {AddVecInfo{buf_size}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

        // Inserts a compute-compute barrier
        void insert_cc_barrier()
        {
            seq->record<kp::OpGlobalMemoryBarrier>(std::vector<std::shared_ptr<kp::Tensor>>(),
                                                   vk::AccessFlagBits::eShaderWrite,
                                                   vk::AccessFlagBits::eShaderRead,
                                                   vk::PipelineStageFlagBits::eComputeShader,
                                                   vk::PipelineStageFlagBits::eComputeShader);
        }

        // Inserts a transfer-compute barrier
        void insert_tc_barrier()
        {
            seq->record<kp::OpGlobalMemoryBarrier>(std::vector<std::shared_ptr<kp::Tensor>>(),
                                                   vk::AccessFlagBits::eTransferWrite,
                                                   vk::AccessFlagBits::eShaderRead,
                                                   vk::PipelineStageFlagBits::eTransfer,
                                                   vk::PipelineStageFlagBits::eComputeShader);
        }

        // Inserts a compute-transfer barrier
        void insert_ct_barrier()
        {
            seq->record<kp::OpGlobalMemoryBarrier>(std::vector<std::shared_ptr<kp::Tensor>>(),
                                                   vk::AccessFlagBits::eShaderWrite,
                                                   vk::AccessFlagBits::eTransferRead,
                                                   vk::PipelineStageFlagBits::eComputeShader,
                                                   vk::PipelineStageFlagBits::eTransfer);
        }

        struct MultiplyInfo
        {
            uint32_t start;
            uint32_t n;
        };

        template <typename DataType>
        void multiply_large(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                            VCBPtr<DataType> left, VCBPtr<DataType> right, VCBPtr<DataType> dest,
                            VCBPtr<uint32_t> lists, VCBPtr<uint32_t> pairs,
                            bool add = true, bool transpose_right = false)
        {
            // get or create algorithm for these specific tensors

            uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            const auto subgroup_size = engine->get_subgroup_size();

            struct MultiplyInfo
            {
                uint32_t start;
                uint32_t n;
            };

            // assume max workgroup size of 1024
            constexpr uint32_t max_size = 512;

            if (szc > max_size)
            {
                throw std::runtime_error("Output matrix size is greater than max workgroup size!");
            }
            // num subgroups per output matrix
            int32_t sg_per_mat = static_cast<int32_t>(ceil_div(szc, subgroup_size));

            int32_t mat_per_wg = static_cast<int32_t>(max_size / (subgroup_size * sg_per_mat)); // workload division factor

            int32_t num_subgroups = mat_per_wg * sg_per_mat;
            int32_t local_size_x = num_subgroups * subgroup_size;
            std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0, transpose_right ? 1 : 0, sg_per_mat, mat_per_wg});
            const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), lists->get_tensor(), pairs->get_tensor()};

            kp::Manager &mgr = *(engine->get_manager());
            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
                                                             engine->spv_sbm_multiply_dist, {}, spec, {MultiplyInfo{start, n}});

            algo->setWorkgroup(kp::Workgroup{n, 1, 1});

            // Dispatch until all output matrix elements have been handled
            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void multiply(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                      VCBPtr<DataType> left, VCBPtr<DataType> right, VCBPtr<DataType> dest,
                      VCBPtr<uint32_t> lists, VCBPtr<uint32_t> pairs,
                      bool add = true, bool transpose_right = false)
        {
            multiply_large(dim1, dim2, start, n, left, right, dest, lists, pairs, add, transpose_right);
        }

        template <typename DataType>
        void multiply_small(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                            VCBPtr<DataType> left, VCBPtr<DataType> right, VCBPtr<DataType> dest,
                            VCBPtr<uint32_t> lists, VCBPtr<uint32_t> pairs,
                            bool add = true, bool transpose_right = false)
        {
            uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            const auto subgroup_size = engine->get_subgroup_size();

            // subgroups per C
            auto subgroups_per_c = szc / subgroup_size;
            if (subgroup_size * subgroups_per_c < szc)
            {
                subgroups_per_c++;
            }
            int32_t local_size_x = subgroups_per_c * subgroup_size;
            // auto tc0 = std::chrono::high_resolution_clock::now();
            std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0, transpose_right ? 1 : 0});

            const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), lists->get_tensor(), pairs->get_tensor()};

            kp::Manager &mgr = *(engine->get_manager());

            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(
                tensors,
                engine->spv_sbm_multiply_subgroup, {n, 1, 1}, spec, {MultiplyInfo{start, n}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void multiply_packed(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                             VCBPtr<DataType> left, VCBPtr<DataType> right, VCBPtr<DataType> dest,
                             VCBPtr<uint32_t> lists, VCBPtr<uint32_t> pairs,
                             bool add = true, bool transpose_right = false)
        {
            const uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            const int32_t local_size_x = 256;
            const uint32_t wgx = ceil_div(n * szc, static_cast<uint32_t>(local_size_x));
            // std::cout << "wgx = " << wgx << "\n";
            // auto tc0 = std::chrono::high_resolution_clock::now();
            std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0, transpose_right ? 1 : 0});

            const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), lists->get_tensor(), pairs->get_tensor()};

            kp::Manager &mgr = *(engine->get_manager());

            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(
                tensors,
                engine->spv_sbm_multiply_packed, {wgx, 1, 1}, spec, {MultiplyInfo{start, n}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void multiply_abat(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                           VCBPtr<DataType> left, VCBPtr<DataType> right, VCBPtr<DataType> dest,
                           VCBPtr<uint32_t> lists, VCBPtr<uint32_t> pairs,
                           bool add = true)
        {

            // uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            const auto subgroup_size = engine->get_subgroup_size();
            // const auto num_subgroups = local_size_x / subgroup_size;

            // subgroups per C
            int32_t req_threads = std::max(dim1.first * dim2.second, dim1.first * dim1.first);
            int32_t subgroups_per_c = (req_threads - 1 + subgroup_size) / subgroup_size;

            // if (subgroup_size * subgroups_per_c < szc)
            // {
            //     subgroups_per_c++;
            // }
            const int32_t local_size_x = subgroups_per_c * subgroup_size;
            // std::cout << "local_size_x: " << local_size_x << std::endl;
            // auto tc0 = std::chrono::high_resolution_clock::now();
            std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0});

            const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), lists->get_tensor(), pairs->get_tensor()};

            kp::Manager &mgr = *(engine->get_manager());

            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(
                tensors,
                engine->spv_sbm_multiply_abat, {n, 1, 1}, spec, {MultiplyInfo{start, n}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void tmultiply(
            BlockDim dim1, BlockDim dim2, uint32_t num_blocks,
            VCBPtr<uint32_t> dest_idx,
            VCBPtr<DataType> dest_data,
            VCBPtr<uint32_t> left_idx,
            VCBPtr<uint32_t> left_offsets,
            VCBPtr<uint32_t> left_ptr,
            VCBPtr<DataType> left_data,
            VCBPtr<uint32_t> right_idx,
            VCBPtr<uint32_t> right_offsets,
            VCBPtr<uint32_t> right_ptr,
            VCBPtr<DataType> right_data,
            bool add = true,
            bool transpose_right = false)
        {

            // get or create algorithm for these specific tensors

            uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            const auto subgroup_size = engine->get_subgroup_size();

            struct MultiplyInfo
            {
                uint32_t start;
                uint32_t n;
            };

            // assume some max workgroup / matrix size
            constexpr uint32_t max_size = 256;

            if (szc > max_size)
            {
                throw std::runtime_error("Output matrix size is greater than max workgroup size!");
            }
            // num subgroups per output matrix
            int32_t sg_per_mat = static_cast<int32_t>(ceil_div(szc, subgroup_size));
            int32_t mat_per_wg = static_cast<int32_t>(max_size / (subgroup_size * sg_per_mat)); // workload division factor

            int32_t num_subgroups = mat_per_wg * sg_per_mat;
            int32_t local_size_x = num_subgroups * subgroup_size;
            std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0, transpose_right ? 1 : 0, sg_per_mat, mat_per_wg});
            const std::vector<std::shared_ptr<kp::Tensor>> tensors{
                dest_idx->get_tensor(),
                dest_data->get_tensor(),
                left_idx->get_tensor(),
                left_offsets->get_tensor(),
                left_ptr->get_tensor(),
                left_data->get_tensor(),
                right_idx->get_tensor(),
                right_offsets->get_tensor(),
                right_ptr->get_tensor(),
                right_data->get_tensor(),
            };

            kp::Manager &mgr = *(engine->get_manager());
            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
                                                             engine->spv_sbm_multiply_dynamic, {}, spec, {MultiplyInfo{0, num_blocks}});

            // algo->setWorkgroup(kp::Workgroup{n, 1, 1});
            algo->setWorkgroup(kp::Workgroup{num_blocks, 1, 1});

            // Dispatch until all output matrix elements have been handled
            // MultiplyInfo push_constant{start, n};
            // algo->setPushConstants(&push_constant, 1, sizeof(push_constant));
            seq->record<kp::OpAlgoDispatch>(algo);
            // std::cout << "Algo dispatched!\n";
            // std::cout << "num_blocks: " << num_blocks << std::endl;
        }

        template <typename DataType>
        void estimate_list(
            uint32_t num_blocks,
            VCBPtr<uint32_t> dest_idx,
            VCBPtr<uint32_t> left_idx,
            VCBPtr<uint32_t> left_offsets,
            VCBPtr<uint32_t> left_ptr,
            VCBPtr<uint32_t> right_idx,
            VCBPtr<uint32_t> right_offsets,
            VCBPtr<uint32_t> right_ptr,
            VCBPtr<uint32_t> allocator_info)
        {

            // get or create algorithm for these specific tensors
            struct MultiplyInfo
            {
                uint32_t n;
            };

            const int32_t local_size = static_cast<int32_t>(engine->get_subgroup_size());
            std::vector<int32_t> spec({local_size});

            const std::vector<std::shared_ptr<kp::Tensor>> tensors{
                dest_idx->get_tensor(),
                left_idx->get_tensor(), left_offsets->get_tensor(), left_ptr->get_tensor(),
                right_idx->get_tensor(), right_offsets->get_tensor(), right_ptr->get_tensor(),
                allocator_info->get_tensor()};

            kp::Manager &mgr = engine->get_manager_ref();
            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
                                                             engine->spv_sbm_multiply_estimate_list, {}, spec, {MultiplyInfo{num_blocks}});

            const uint32_t wgx = (num_blocks + local_size - 1) / local_size;
            algo->setWorkgroup(kp::Workgroup{wgx, 1, 1});

            // Dispatch until all output matrix elements have been handled
            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void build_list(
            uint32_t num_blocks,
            VCBPtr<uint32_t> dest_idx,
            VCBPtr<uint32_t> left_idx,
            VCBPtr<uint32_t> left_offsets,
            VCBPtr<uint32_t> left_ptr,
            VCBPtr<uint32_t> right_idx,
            VCBPtr<uint32_t> right_offsets,
            VCBPtr<uint32_t> right_ptr,
            VCBPtr<uint32_t> mul_lists,
            VCBPtr<uint32_t> mul_pairs,
            VCBPtr<uint32_t> allocator_info)
        {

            // get or create algorithm for these specific tensors

            struct MultiplyInfo
            {
                uint32_t num_blocks;
            };

            const int32_t local_size = static_cast<int32_t>(engine->get_subgroup_size());

            std::vector<int32_t> spec({local_size});

            const std::vector<std::shared_ptr<kp::Tensor>> tensors{
                dest_idx->get_tensor(),
                left_idx->get_tensor(), left_offsets->get_tensor(), left_ptr->get_tensor(),
                right_idx->get_tensor(), right_offsets->get_tensor(), right_ptr->get_tensor(),
                mul_lists->get_tensor(), mul_pairs->get_tensor(),
                allocator_info->get_tensor()};

            kp::Manager &mgr = *(engine->get_manager());
            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
                                                             engine->spv_sbm_multiply_build_list, {}, spec, {MultiplyInfo{num_blocks}});

            const uint32_t wgx = (num_blocks + local_size - 1) / local_size;
            algo->setWorkgroup(kp::Workgroup{wgx, 1, 1});

            // Dispatch until all output matrix elements have been handled
            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void dispatch_right_block_diagonal(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                                           VCBPtr<DataType> left, VCBPtr<DataType> right, VCBPtr<DataType> dest, VCBPtr<uint32_t> pairs, bool add)
        {

            int32_t local_size_x = static_cast<int32_t>(engine->get_subgroup_size());
            int32_t szc = dim1.first * dim2.second;
            uint32_t wgx = (szc * n + local_size_x - 1) / local_size_x;

            std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0});

            const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), pairs->get_tensor()};

            kp::Manager &mgr = *(engine->get_manager());

            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
                                                             engine->spv_sbm_multiply_block_diagonal, kp::Workgroup{wgx, 1, 1}, spec, {MultiplyInfo{start, n}});

            // update push constants
            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void dispatch_left_block_diagonal(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                                          VCBPtr<DataType> left, VCBPtr<DataType> right, VCBPtr<DataType> dest, VCBPtr<uint32_t> pairs, bool add)
        {
            // get or create algorithm for these specific tensors
            auto szc = dim1.first * dim2.second;
            const int32_t local_size_x = 128;
            // auto num_workgroups = static_cast<uint32_t>(std::ceil(static_cast<double>(n * szc) / (local_size_x) + 1));
            const uint32_t num_workgroups = ceil_div<uint32_t>(static_cast<uint32_t>(n * szc), static_cast<uint32_t>(local_size_x));

            kp::Manager &mgr = *(engine->get_manager());

            // TODO: Fix shader so it does not take pairs twice
            const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), pairs->get_tensor()};

            const std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0});
            auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
                                                             engine->spv_sbm_left_multiply_block_diagonal, {num_workgroups, 1, 1}, spec, {MultiplyInfo{start, n}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void invert_block_diagonal(VCBPtr<DataType> dest, VCBPtr<DataType> src, const robin_hood::unordered_map<Dim, std::vector<uint32_t>> &offsets)
        {

            // create a buffer to hold all offsets
            const int32_t local_size_x = static_cast<int32_t>(engine->get_subgroup_size() * 8);
            size_t buf_size = 0;
            for (auto &it : offsets)
            {
                buf_size += it.second.size();
            }

            auto tensor_type = src->get_tensor()->tensorType() == kp::Tensor::TensorTypes::eHost ? BufferType::Host : BufferType::DeviceCached;
            // auto tensor_type = BufferType::Host;
            auto buf_offsets = (*engine).template create_buffer<uint32_t>(nullptr, buf_size, tensor_type);
            sync_device<uint32_t>({buf_offsets}, {}, true);

            auto offset_ptr = buf_offsets->map();

            std::atomic_uint32_t write_idx(0);
            for (auto &it : offsets)
            {
                auto start_idx = write_idx.fetch_add(static_cast<uint32_t>(it.second.size()));
                auto write_ptr = reinterpret_cast<uint32_t *>(offset_ptr + start_idx);
                std::copy(it.second.begin(), it.second.end(), write_ptr);

                // create algorithm
                auto &mgr = engine->get_manager_ref();
                auto inv_algo = mgr.template algorithm<int32_t, MultiplyInfo>({src->get_tensor(), dest->get_tensor(), buf_offsets->get_tensor()},
                                                                              engine->spv_sbm_bdi, {}, {it.first, local_size_x}, {MultiplyInfo{0, 0}});

                auto n = static_cast<uint32_t>(it.second.size());
                // set push constants and workgroup
                MultiplyInfo push_constant{static_cast<uint32_t>(start_idx), n};
                inv_algo->setPushConstants(&push_constant, 1, sizeof(push_constant));
                auto num_workgroups = static_cast<uint32_t>(std::ceil(static_cast<double>(n) / (local_size_x)));
                inv_algo->setWorkgroup(kp::Workgroup{num_workgroups, 1, 1});
                // std::cout << "num inv workgroups: " << num_workgroups << std::endl;
                // record
                seq->record<kp::OpAlgoDispatch>(inv_algo);
            }

            // add final barrier
            seq->record<kp::OpGlobalMemoryBarrier>({dest->get_tensor()},
                                                   vk::AccessFlagBits::eShaderWrite,
                                                   vk::AccessFlagBits::eShaderRead,
                                                   vk::PipelineStageFlagBits::eComputeShader,
                                                   vk::PipelineStageFlagBits::eComputeShader);
            // seq->record<kp::OpTensorSyncLocal>(tensors);
        }

        // sync device
        template <typename DataType>
        void sync_device(const std::vector<VCBPtr<DataType>> &buffers, const std::vector<std::pair<uint32_t, uint32_t>> &ranges = {}, bool once = false)
        {
            std::vector<std::shared_ptr<kp::Tensor>> added;
            std::vector<std::pair<uint32_t, uint32_t>> added_ranges;
            for (size_t i = 0; i < buffers.size(); i++)
            {
                VCBPtr<DataType> b = buffers[i];
                if (b->get_buffer_type() == BufferType::Device || b->get_buffer_type() == BufferType::DeviceCached)
                {
                    added.push_back(b->get_tensor());
                    if (ranges.size() > 0)
                    {
                        added_ranges.push_back(ranges.at(i));
                    }
                }
            }

            // to_local_sync = buffers;
            if (added.size() > 0)
            {
                if (added_ranges.size() > 0)
                {
                    if (once)
                    {
                        init_ops = true;
                        init_seq->record<kp::OpTensorSyncDevice>(added, added_ranges);
                    }
                    else
                    {
                        seq->record<kp::OpTensorSyncDevice>(added, added_ranges);
                    }
                }
                else
                {
                    if (once)
                    {
                        init_ops = true;
                        init_seq->record<kp::OpTensorSyncDevice>(added);
                    }
                    else
                    {
                        seq->record<kp::OpTensorSyncDevice>(added);
                    }
                }
            }
        }

        template <typename DataType>
        void sync_local(const std::vector<VCBPtr<DataType>> &buffers, const std::vector<std::pair<uint32_t, uint32_t>> &ranges = {}, bool once = false)
        {
            std::vector<std::shared_ptr<kp::Tensor>> added;
            std::vector<std::pair<uint32_t, uint32_t>> added_ranges;
            for (size_t i = 0; i < buffers.size(); i++)
            {
                VCBPtr<DataType> b = buffers[i];
                if (b->get_buffer_type() == BufferType::Device || b->get_buffer_type() == BufferType::DeviceCached)
                {
                    added.push_back(b->get_tensor());
                    if (ranges.size() > 0)
                    {
                        added_ranges.push_back(ranges.at(i));
                    }
                }
            }

            // to_local_sync = buffers;
            if (added.size() > 0)
            {
                if (added_ranges.size() > 0)
                {
                    if (once)
                    {
                        init_ops = true;
                        init_seq->record<kp::OpTensorSyncLocal>(added, added_ranges);
                    }
                    else
                    {
                        seq->record<kp::OpTensorSyncLocal>(added, added_ranges);
                    }
                }
                else
                {
                    if (once)
                    {
                        init_ops = true;
                        init_seq->record<kp::OpTensorSyncLocal>(added);
                    }
                    else
                    {
                        seq->record<kp::OpTensorSyncLocal>(added);
                    }
                }
            }
        }

        template <typename DataType>
        void set_lambda(uint32_t dim, VCBPtr<DataType> blocks, VCBPtr<DataType> backups,
                        VCBPtr<uint32_t> offsets, VCBPtr<DataType> lambda, uint32_t num_mat, uint32_t id_offset, uint32_t backup_offset)
        {
            set_restore_lambda<DataType>(dim, blocks, backups, offsets, lambda, num_mat, id_offset, backup_offset, true);
        }

        template <typename DataType>
        void restore_lambda(uint32_t dim, VCBPtr<DataType> blocks, VCBPtr<DataType> backups,
                            VCBPtr<uint32_t> offsets, VCBPtr<DataType> lambda, uint32_t num_mat, uint32_t id_offset, uint32_t backup_offset)
        {
            set_restore_lambda<DataType>(dim, blocks, backups, offsets, lambda, num_mat, id_offset, backup_offset, false);
        }

        template <typename DataType>
        void copy_blocks(BlockDim dim, VCBPtr<DataType> src, VCBPtr<DataType> dest,
                         VCBPtr<uint32_t> offsets, uint32_t num_mat, uint start)
        {
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 256;
            const uint32_t items = num_mat * dim.first * dim.second;
            uint32_t wgx = items / local_size_x;
            if (wgx * local_size_x < items)
                wgx++;
            kp::Workgroup wg({wgx, 1, 1});

            std::vector<int32_t> spec({local_size_x, dim.first, dim.second});

            struct PushConstant
            {
                uint32_t num_blocks;
                uint32_t start;
            };

            std::vector<std::shared_ptr<kp::Tensor>> tensors({src->get_tensor(), dest->get_tensor(),
                                                              offsets->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, PushConstant>(tensors,
                                                                                       engine->spv_copy_blocks,
                                                                                       wg,
                                                                                       spec,
                                                                                       {PushConstant{num_mat, start}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

    private:
        template <typename DataType>
        void set_restore_lambda(uint32_t dim, VCBPtr<DataType> blocks, VCBPtr<DataType> backups,
                                VCBPtr<uint32_t> offsets, VCBPtr<DataType> lambda, uint32_t num_mat, uint32_t id_offset, uint32_t backup_offset, bool set)
        {
            kp::Manager &mgr = *(engine->get_manager());

            int32_t local_size_x = 256;
            uint32_t items = num_mat * dim;
            uint32_t wgx = items / local_size_x;
            if (wgx * local_size_x < items)
                wgx++;
            kp::Workgroup wg({wgx, 1, 1});

            std::vector<int32_t> spec({local_size_x, static_cast<int32_t>(dim), set, set});

            struct PushConstant
            {
                uint32_t items;
                uint32_t id_offset;
                uint32_t backup_offset;
            };

            std::vector<std::shared_ptr<kp::Tensor>> tensors({blocks->get_tensor(), backups->get_tensor(),
                                                              offsets->get_tensor(), lambda->get_tensor()});
            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int32_t, PushConstant>(tensors,
                                                                                       engine->spv_set_diagonal,
                                                                                       wg,
                                                                                       spec,
                                                                                       {PushConstant{items, id_offset, backup_offset}});

            seq->record<kp::OpAlgoDispatch>(algo);
        }

    public:
        void execute()
        {
            if (init && init_ops)
            {
                init_seq->eval();
                init = false;
            }
            seq->eval();
        }
    };

}