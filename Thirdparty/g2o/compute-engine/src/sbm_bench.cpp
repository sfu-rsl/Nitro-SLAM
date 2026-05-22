
#include <iostream>
#include <solver/backend.hpp>
#include <solver/sparse_block_matrix.hpp>
#include <solver/utility.hpp>
#include <benchmark/benchmark.h>
#include <solver/linear_solver.hpp>
#include <solver/utility.hpp>

compute::ComputeEngine engine;

std::shared_ptr<compute::SyncObject> create_sync_object()
{
    return std::make_shared<compute::SyncObject>(engine.create_op_sequence(), engine.create_op_sequence());
}

static void BM_INNER_PRODUCT_EIGEN(benchmark::State &state)
{
    using namespace compute;

    int vlen = 1e6;

    Eigen::VectorXd vec = Eigen::VectorXd::Random(vlen);
    Eigen::VectorXd vec2 = Eigen::VectorXd::Random(vlen);

    double value = 0.0;
    for (auto _ : state)
    {
        value = vec.dot(vec2);
    }
    std::cout << value << "\n";
}

static void BM_INNER_PRODUCT_DEVICE(benchmark::State &state)
{
    using namespace compute;

    int vlen = 1e6;

    Eigen::VectorXd vec = Eigen::VectorXd::Random(vlen);
    Eigen::VectorXd vec2 = Eigen::VectorXd::Random(vlen);
    auto x = engine.create_buffer<double>(vec.data(), vlen, BufferType::DeviceCached);
    auto x2 = engine.create_buffer<double>(vec2.data(), vlen, BufferType::DeviceCached);

    auto out = engine.create_buffer<double>(nullptr, vlen, BufferType::Device);
    auto out2 = engine.create_buffer<double>(nullptr, vlen, BufferType::DeviceCached);

    auto sync_in = create_sync_object();
    sync_in->rec<double>({x, x2});

    auto s = engine.create_op_sequence();
    s->inner_product(x, x2, out, out2);
    sync_in->sync_device();
    auto sync_out = create_sync_object();
    sync_out->rec<double>({out2}, {{0, 1}});
    for (auto _ : state)
    {
        // sync_in->sync_device();
        s->execute();
        sync_out->sync_local();
    }
}

static void BM_SELF_INNER_PRODUCT(benchmark::State &state)
{
    using namespace compute;

    int vlen = 1e6;

    Eigen::VectorXd vec = Eigen::VectorXd::Random(vlen);
    auto x = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
    auto out = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
    auto out2 = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);

    auto s = engine.create_op_sequence();
    s->self_inner_product(x, out, out2);
    // double val;
    for (auto _ : state)
    {
        s->execute();
        // val = vec.dot(vec);
    }
    // std::cout << "val: " << val << "\n";
}

static void BM_INNER_PRODUCT(benchmark::State &state)
{
    using namespace compute;

    int vlen = 1e6;

    Eigen::VectorXd vec = Eigen::VectorXd::Random(vlen);
    Eigen::VectorXd vec2 = Eigen::VectorXd::Random(vlen);
    auto x = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
    auto x2 = engine.create_buffer<double>(vec2.data(), vlen, BufferType::Host);

    auto out = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);
    auto out2 = engine.create_buffer<double>(vec.data(), vlen, BufferType::Host);

    auto s = engine.create_op_sequence();
    s->inner_product(x, x2, out, out2);

    for (auto _ : state)
    {
        s->execute();
    }
}

static void BM_INVERSION_OP(benchmark::State &state)
{
    // setup matrices
    using namespace compute;

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt");

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory();
    hll_inv->zero_memory();
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    for (auto _ : state)
    {
        inversion_op->execute();
    }
}

static void BM_RIGHT_BLOCK_DIAGONAL_MULTIPLICATION(benchmark::State &state)
{
    // int32_t d = state.range(0);
    // setup matrices
    using namespace compute;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt");

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt");

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt");
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt");

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory();
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory();
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory();
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto calc_Hschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false, true, true);

    // do calculations

    inversion_op->execute();
    hpp->copy_blocks_into(hschur);

    for (auto _ : state)
    {
        calc_HplinvHll->execute();
    }
}

static void BM_TRANSPOSE_MULTIPLY(benchmark::State &state)
{
    // int32_t d = state.range(0);
    // setup matrices
    using namespace compute;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt");

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt");

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt");
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt");

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory();
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory();
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory();
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    // auto calc_HplinvHll = hpl->subgroup_block_diagonal_multiply_add(hpl_inv_hll, hll_inv);

    // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur, hpl, false, 0, true, true);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto calc_Hschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false, true, true);

    // do calculations

    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    // warm up
    // auto t0 = std::chrono::high_resolution_clock::now();
    // calc_Hschur->execute();
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "First took: " << std::chrono::duration<double>(t1-t0).count() << std::endl;

    for (auto _ : state)
    {
        // auto t0 = std::chrono::high_resolution_clock::now();
        // hschur->zero_memory();
        // hpp->copy_blocks_into(hschur);
        // inversion_op->execute();
        // calc_HplinvHll->execute();
        calc_Hschur->execute();
        // auto t1 = std::chrono::high_resolution_clock::now();
        // std::cout << "First took: " << std::chrono::duration<double>(t1-t0).count() << std::endl;
    }
}

static void BM_COPY_BLOCKS(benchmark::State &state)
{
    // int32_t d = state.range(0);
    // setup matrices
    using namespace compute;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt");

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt");

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt");
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt");

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory();
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory();
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory();
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto calc_Hschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false, true, true);

    // do calculations

    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    for (auto _ : state)
    {
        hpp->copy_blocks_into(hschur);
    }
}

static void BM_ZERO_MEMORY(benchmark::State &state)
{
    // int32_t d = state.range(0);
    // setup matrices
    using namespace compute;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt");

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt");

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt");
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt");

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory();
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory();
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory();
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto calc_Hschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false, true, true);

    // do calculations

    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    for (auto _ : state)
    {
        hschur->zero_memory();
    }
}

static void BM_BLOCK_DIAGONAL_VEC_MULTIPLICATION(benchmark::State &state)
{
    // setup matrices
    using namespace compute;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt");

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt");

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt");
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt");

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory();
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory();
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory();
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto calc_Hschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false, true, true);

    // do calculations

    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    // load bschur data
    auto bp = loadFromFile<double>(engine, "../data/bp.txt");
    auto bl = loadFromFile<double>(engine, "../data/bl.txt");
    auto bschur_expected = loadFromFile<double>(engine, "../data/bschur.txt");
    // auto bschur = engine.create_buffer<double>(nullptr, bschur_expected->size(), compute::BufferType::Host);
    auto bschur = bp;

    // calculate bschur
    auto calc_bschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_multiply_vec_add(calc_bschur, bschur, bl, false);
    calc_bschur->execute();
    // check
    auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur->map(), bschur->size(), 1);
    auto bschur_expected_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(), bschur_expected->size(), 1);

    // std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);
    double tol = 1e-4;
    for (size_t i = 0; i < bschur_expected_map.rows(); i++)
    {

        if (fabs(bschur_map(i) - bschur_expected_map(i)) > tol)
        {
            std::cerr << "Expected " << bschur_expected_map(i) << " at " << i << ", got " << bschur_map(i) << std::endl;
        }
    }

    // load update data
    auto xp = loadFromFile<double>(engine, "../data/xp.txt");
    auto xl_expected = loadFromFile<double>(engine, "../data/xl.txt");

    auto cl = bl;
    auto xl = engine.create_buffer<double>(nullptr, xl_expected->size(), compute::BufferType::Host);
    memset(xl->map(), 0, xl->size() * sizeof(double));

    // calculate
    auto calc_cl = engine.create_op_sequence();
    auto calc_xl = engine.create_op_sequence();
    hpl->subgroup_right_multiply_vec_add(calc_cl, cl, xp, false);
    hll_inv->subgroup_block_diagonal_multiply_vec_add(calc_xl, xl, cl, true);

    calc_cl->execute();
    calc_xl->execute();

    double *xl_cpu = new double[xl->size()];
    for (auto _ : state)
    {
        // memset(xl->map(), 0, xl->size()*sizeof(double));
        // calc_cl->execute();
        calc_xl->execute();
        // copy
        // memcpy(xl_cpu, xl->map(), sizeof(double)*xl->size());
    }
    delete[] xl_cpu;
}

static void BM_RIGHT_MULTIPLY_VEC_ADD(benchmark::State &state)
{
    // setup matrices
    using namespace compute;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt");

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt");

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt");
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt");

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory();
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory();
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory();
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto calc_Hschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false, true, true);
    // do calculations

    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    // load bschur data
    auto bp = loadFromFile<double>(engine, "../data/bp.txt");
    auto bl = loadFromFile<double>(engine, "../data/bl.txt");
    auto bschur_expected = loadFromFile<double>(engine, "../data/bschur.txt");
    // auto bschur = engine.create_buffer<double>(nullptr, bschur_expected->size(), compute::BufferType::Host);
    auto bschur = bp;

    // calculate bschur
    auto calc_bschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_multiply_vec_add(calc_bschur, bschur, bl, false);
    calc_bschur->execute();
    // check
    auto bschur_map = Eigen::Map<Eigen::MatrixXd>(bschur->map(), bschur->size(), 1);
    auto bschur_expected_map = Eigen::Map<Eigen::MatrixXd>(bschur_expected->map(), bschur_expected->size(), 1);

    // std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);
    double tol = 1e-4;
    for (Eigen::Index i = 0; i < bschur_expected_map.rows(); i++)
    {

        if (fabs(bschur_map(i) - bschur_expected_map(i)) > tol)
        {
            std::cerr << "Expected " << bschur_expected_map(i) << " at " << i << ", got " << bschur_map(i) << std::endl;
        }
    }

    // load update data
    auto xp = loadFromFile<double>(engine, "../data/xp.txt");
    auto xl_expected = loadFromFile<double>(engine, "../data/xl.txt");

    auto cl = bl;
    auto xl = engine.create_buffer<double>(nullptr, xl_expected->size(), compute::BufferType::Host);
    memset(xl->map(), 0, xl->size() * sizeof(double));

    // calculate
    auto calc_cl = engine.create_op_sequence();
    // auto calc_xl = engine.create_op_sequence();
    hpl->subgroup_right_multiply_vec_add(calc_cl, cl, xp, false);
    // hll_inv->subgroup_block_diagonal_multiply_vec_add2(calc_xl, xl, cl, true);

    // calc_cl->execute();
    // calc_xl->execute();

    double *xl_cpu = new double[xl->size()];
    for (auto _ : state)
    {
        // memset(xl->map(), 0, xl->size()*sizeof(double));
        calc_cl->execute();
        // calc_xl->execute();
        // copy
        // memcpy(xl_cpu, xl->map(), sizeof(double)*xl->size());
    }
    delete[] xl_cpu;
}

static void BM_HSCHUR_DEVICE(benchmark::State &state)
{

    using namespace compute;
    auto btd = BufferType::DeviceCached;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt", btd);

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt", btd);

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt", btd);
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt", btd);

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory(btd);
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory(btd);
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory(btd);
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto calc_Hschur = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add(calc_Hschur, hschur, hpl, false, true, true);
    // calc_Hschur->insert_host_sync_barrier();

    // Sync
    auto sync_Hll_Hpl = create_sync_object();
    sync_Hll_Hpl->rec<double>({hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
    sync_Hll_Hpl->sync_device();

    auto sync_Hschur = create_sync_object();
    sync_Hschur->rec<double>({hschur->get_buffer()});

    // do calculations
    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    sync_Hschur->sync_device();
    calc_Hschur->execute();
    // sync_Hschur->sync_local();
    // calc_Hschur->debug_print();

    for (auto _ : state)
    {
        calc_Hschur->execute();
    }
}

static void BM_HSCHUR_DEVICE2(benchmark::State &state)
{

    using namespace compute;
    auto btd = BufferType::DeviceCached;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt", btd);

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt", btd);

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt", btd);
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt", btd);

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory(btd);
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory(btd);
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory(btd);
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto seq = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add2(seq, hschur, hpl, false, true);
    // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur, hpl, false, 0, true, true);
    // calc_Hschur->insert_host_sync_barrier();

    // Sync
    auto sync_Hll_Hpl = create_sync_object();
    sync_Hll_Hpl->rec<double>({hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
    sync_Hll_Hpl->sync_device();

    auto sync_Hschur = create_sync_object();
    sync_Hschur->rec<double>({hschur->get_buffer()});

    // do calculations
    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    sync_Hschur->sync_device();
    // calc_Hschur->execute();

    // sync_Hschur->sync_local();
    // calc_Hschur->debug_print();

    for (auto _ : state)
    {
        // calc_Hschur->execute();
        seq->execute();
    }
}

static void BM_HSCHUR_DEVICE3(benchmark::State &state)
{

    using namespace compute;
    auto btd = BufferType::DeviceCached;
    auto hpp = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpp->loadFromFile("../data/hpp.txt", btd);

    auto hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll->loadFromFile("../data/hll.txt", btd);

    auto hpl = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl->loadFromFile("../data/hpl.txt", btd);
    hpl->sort_by_index();

    auto hschur_expected = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur_expected->loadFromFile("../data/hschur.txt", btd);

    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->take_structure_from(hschur_expected);
    hschur->sort_by_index();
    hschur->allocate_memory(btd);
    hschur->zero_memory();

    auto hll_inv = std::make_shared<SparseBlockMatrix<double>>(engine);
    hll_inv->take_structure_from(hll);
    hll_inv->allocate_memory(btd);
    hll_inv->zero_memory();

    auto hpl_inv_hll = std::make_shared<SparseBlockMatrix<double>>(engine);
    hpl_inv_hll->take_structure_from(hpl);
    hpl_inv_hll->allocate_memory(btd);
    hpl_inv_hll->zero_memory();

    // prepare calculations
    auto inversion_op = engine.create_op_sequence();
    hll->create_inversion_op(inversion_op, hll_inv);

    auto calc_HplinvHll = engine.create_op_sequence();
    hpl->subgroup_block_diagonal_multiply_add(calc_HplinvHll, hpl_inv_hll, hll_inv);

    auto seq = engine.create_op_sequence();
    hpl_inv_hll->subgroup_transposed_multiply_add3(seq, hschur, hpl, false, true);
    // auto calc_Hschur = hpl_inv_hll->subgroup_transposed_multiply_add(hschur, hpl, false, 0, true, true);
    // calc_Hschur->insert_host_sync_barrier();

    // Sync
    auto sync_Hll_Hpl = create_sync_object();
    sync_Hll_Hpl->rec<double>({hpp->get_buffer(), hll->get_buffer(), hpl->get_buffer()});
    sync_Hll_Hpl->sync_device();

    auto sync_Hschur = create_sync_object();
    sync_Hschur->rec<double>({hschur->get_buffer()});

    // do calculations
    inversion_op->execute();
    hpp->copy_blocks_into(hschur);
    calc_HplinvHll->execute();

    sync_Hschur->sync_device();
    // calc_Hschur->execute();

    // sync_Hschur->sync_local();
    // calc_Hschur->debug_print();

    for (auto _ : state)
    {
        // calc_Hschur->execute();
        seq->execute();
    }
}

static void BM_LDLT(benchmark::State &state)
{
    using namespace compute;

    auto btd = BufferType::DeviceCached;

    // Load Data
    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->loadFromFile("data/hschur.txt", btd);
    auto sync_Hschur = create_sync_object();
    sync_Hschur->rec<double>({hschur->get_buffer()});
    sync_Hschur->sync_device();
    auto hschur_csc = hschur->to_csc();

    LinearSolver<double> *solver = new LDLTSolver<double>();

    auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
    auto xp = engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

    // sync bschur
    auto sync_b = create_sync_object();
    sync_b->rec<double>({bschur_expected});
    sync_b->sync_device();

    // Solver Setup
    solver->setup(hschur, xp, bschur_expected);

    // Run Solver
    for (auto _ : state)
    {
        solver->solve(hschur, xp, bschur_expected);
    }

    delete solver;
}

static void BM_DENSE_LDLT(benchmark::State &state)
{
    using namespace compute;

    auto btd = BufferType::DeviceCached;

    // Load Data
    auto hschur = std::make_shared<SparseBlockMatrix<double>>(engine);
    hschur->loadFromFile("data/hschur.txt", btd);
    auto sync_Hschur = create_sync_object();
    sync_Hschur->rec<double>({hschur->get_buffer()});
    sync_Hschur->sync_device();
    auto hschur_csc = hschur->to_csc();

    LinearSolver<double> *solver = new DenseLDLTSolver<double>();

    auto bschur_expected = loadFromFile<double>(engine, "data/bschur.txt", btd);
    auto xp = engine.create_buffer<double>(nullptr, hschur->num_scalar_rows(), btd);

    // sync bschur
    auto sync_b = create_sync_object();
    sync_b->rec<double>({bschur_expected});
    sync_b->sync_device();

    // Solver Setup
    solver->setup(hschur, xp, bschur_expected);

    // Run Solver
    for (auto _ : state)
    {
        solver->solve(hschur, xp, bschur_expected);
    }

    delete solver;
}

BENCHMARK(BM_LDLT)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_DENSE_LDLT)->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_LDLT2)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_INNER_PRODUCT_DEVICE)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_INNER_PRODUCT_EIGEN)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SELF_INNER_PRODUCT)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_INNER_PRODUCT)->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_TRANSPOSE_MULTIPLY)->Iterations(1)->Repetitions(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HSCHUR_DEVICE)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HSCHUR_DEVICE2)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HSCHUR_DEVICE3)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_TRANSPOSE_MULTIPLY)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RIGHT_BLOCK_DIAGONAL_MULTIPLICATION)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RIGHT_MULTIPLY_VEC_ADD)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BLOCK_DIAGONAL_VEC_MULTIPLICATION)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ZERO_MEMORY)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_COPY_BLOCKS)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_INVERSION_OP)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
