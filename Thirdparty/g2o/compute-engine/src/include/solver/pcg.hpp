#pragma once
#include <solver/linear_solver.hpp>

namespace compute
{

    enum class Preconditioner
    {
        Hpp,
        Schur
    };

    template <typename DataType>
    class PCGSolver : public LinearSolver<DataType>, public ImplicitSchurSolver<DataType>
    {
    private:
        bool init;
        double tol;
        double residual;
        bool absolute_tol;
        Preconditioner preconditioner;
        size_t max_iterations;

        // Engine
        VulkanComputeEngine &engine;

        // Buffers
        VCBPtr<DataType> l0, l1, vk, ak_scratch, akn, akd, rk, zk, pk, rk1;

        // Matrices
        MatPtr<DataType> Minv; // preconditioner;

        // Sequences
        std::shared_ptr<SolverSeq> init_seq, reset_seq, seq1, pc_seq;

    public:
        PCGSolver(VulkanComputeEngine &engine) : init(false), tol(1e-6), residual(-1.0), absolute_tol(true),
                                                 preconditioner(Preconditioner::Schur), max_iterations(0), engine(engine)
        {
            Minv = std::make_shared<SparseBlockMatrix<DataType>>(engine);
        }

        void setup(MatPtr<DataType> A, BufferPtr<DataType> x, BufferPtr<DataType> b) override
        {
            setup(A, nullptr, nullptr, nullptr, x, b);
        }

        void setup(MatPtr<DataType> Hpp, MatPtr<DataType> Hpl, MatPtr<DataType> Hll,
                   BufferPtr<DataType> x, BufferPtr<DataType> b) override
        {
            setup(nullptr, Hpp, Hpl, Hll, x, b);
        }

        bool solve(MatPtr<DataType> A, BufferPtr<DataType> x, BufferPtr<DataType> b) override
        {
            // Pre-loop initialization
            A->copy_inverse_diagonal_blocks_into(Minv);
            size_t max_iter = A->num_scalar_rows();
            if (max_iterations > 0)
            {
                max_iter = max_iterations;
            }
            (void)(x);
            (void)(b);
            return solve(max_iter);
        }

        bool solve(MatPtr<DataType> Hpp, MatPtr<DataType> Hpl, MatPtr<DataType> Hllinv, BufferPtr<DataType> x, BufferPtr<DataType> b)
        {
            switch (preconditioner)
            {
            case Preconditioner::Schur:
                pc_seq->execute();
                for (size_t i = 0; i < Minv->num_rows(); i++)
                {
                    auto map = Minv->get_block_map(i, i);
                    map = map.inverse();
                }
                break;
            case Preconditioner::Hpp:
                Hpp->copy_inverse_diagonal_blocks_into(Minv);
                break;
            default:
                break;
            }
            size_t max_iter = Hpp->num_scalar_rows();
            if (max_iterations > 0)
            {
                max_iter = max_iterations;
            }

            (void)(Hpl);
            (void)(Hllinv);
            (void)(x);
            (void)(b);

            return solve(max_iter);
        }

        bool result_gpu() override
        {
            return true;
        }

        static void compute_Ax(std::shared_ptr<SolverSeq> seq, MatPtr<DataType> A,
                               MatPtr<DataType> Hpp, MatPtr<DataType> Hpl, MatPtr<DataType> Hllinv,
                               BufferPtr<DataType> in, BufferPtr<DataType> out, BufferPtr<DataType> l0, BufferPtr<DataType> l1, bool add)
        {
            if (A)
            {
                A->multiply_vec_add_upper(seq, out, in, add);
                A->multiply_vec_lower_no_diagonal(seq, out, in, add);
            }
            else
            {
                // implicit Schur
                seq->fill_vec(l0, 0.0);
                seq->fill_vec(l1, 0.0);
                seq->insert_cc_barrier();

                Hpl->subgroup_right_multiply_vec_add(seq, l0, in); // Hpl^T*x
                seq->insert_cc_barrier();

                Hllinv->subgroup_block_diagonal_multiply_vec_add(seq, l1, l0); // Hll^{-1}*x'
                seq->insert_cc_barrier();

                Hpl->subgroup_multiply_vec_add(seq, out, l1, !add); //  - Hpl*x''
                seq->insert_cc_barrier();

                // Hpp only stores upper blocks like Hschur
                Hpp->multiply_vec_add_upper(seq, out, in, add);
                Hpp->multiply_vec_lower_no_diagonal(seq, out, in, add);
            }
        }

        void set_preconditioner(const Preconditioner preconditioner)
        {
            this->preconditioner = preconditioner;
        }

        void set_tolerance(const double tolerance)
        {
            tol = tolerance;
        }

        void set_max_iterations(const size_t iterations)
        {
            max_iterations = iterations;
        }

        void use_absolute_tol(const bool absolute_tol)
        {
            this->absolute_tol = absolute_tol;
        }

    private:
        void setup(MatPtr<DataType> A, MatPtr<DataType> Hpp, MatPtr<DataType> Hpl, MatPtr<DataType> Hllinv,
                   BufferPtr<DataType> x, BufferPtr<DataType> b)
        {

            if (init)
            {
                throw std::runtime_error("PCG Solver is already initialized! Reinitialization not supported!");
            }

            // allocate
            auto btd = A ? A->get_buffer()->get_buffer_type() : Hpp->get_buffer()->get_buffer_type();
            size_t n = A ? A->num_scalar_rows() : Hpp->num_scalar_rows();

            vk = engine.create_buffer<DataType>(nullptr, n, compute::BufferType::Storage);
            ak_scratch = engine.create_buffer<DataType>(nullptr, n, compute::BufferType::Storage);
            akn = engine.create_buffer<DataType>(nullptr, n, btd);
            akd = engine.create_buffer<DataType>(nullptr, n, btd);

            rk = engine.create_buffer<DataType>(nullptr, n, compute::BufferType::Storage);
            zk = engine.create_buffer<DataType>(nullptr, n, compute::BufferType::Storage);
            pk = engine.create_buffer<DataType>(nullptr, n, compute::BufferType::Storage);

            if (Hllinv)
            {
                l0 = engine.create_buffer<DataType>(nullptr, Hllinv->num_scalar_rows(), compute::BufferType::Storage);
                l1 = engine.create_buffer<DataType>(nullptr, Hllinv->num_scalar_rows(), compute::BufferType::Storage);
            }

            // execute pre-loop initialization
            // Prepare M
            A ? Minv->take_diagonal_structure_from(A) : Minv->take_diagonal_structure_from(Hpp);
            Minv->allocate_memory(btd);
            if (preconditioner == Preconditioner::Schur && !A)
            {
                pc_seq = engine.create_op_sequence();
                Hpp->copy_diagonal_blocks_into(pc_seq, Minv);
                pc_seq->insert_cc_barrier();
                Hpl->multiply_block_diagonal_self_transpose(pc_seq, Minv, Hllinv, false);
                pc_seq->insert_cc_barrier();
                pc_seq->sync_local<DataType>({Minv->get_buffer()});
            }

            // r0 = b
            init_seq = engine.create_op_sequence();
            {
                init_seq->sync_device<DataType>({Minv->get_buffer()});
                init_seq->copy_vec(b, rk);
                // also init x
                init_seq->fill_vec(x, 0.0);

                // z0 = M^{-1}r0
                init_seq->fill_vec(zk, 0.0);
                init_seq->insert_cc_barrier();
                Minv->subgroup_block_diagonal_multiply_vec_add(init_seq, zk, rk);

                // p0 = z0
                init_seq->insert_cc_barrier();
                init_seq->copy_vec(zk, pk);
                init_seq->insert_cc_barrier();
                // compute a0n = dot(r0, z0)
                init_seq->inner_product(rk, zk, ak_scratch, akn);
                // init_seq->self_inner_product(rk, ak_scratch, akd);
                // init_seq->insert_cc_barrier();
                init_seq->sync_local<DataType>({akn}, {{0, 1}});
            }

            reset_seq = engine.create_op_sequence();
            {
                reset_seq->copy_vec(b, rk);
                reset_seq->fill_vec(zk, 0.0);
                reset_seq->insert_cc_barrier();
                compute_Ax(reset_seq, A, Hpp, Hpl, Hllinv, x, rk, l0, l1, false);
                reset_seq->insert_cc_barrier();
            }

            // vk = 0, vk = Apk
            seq1 = engine.create_op_sequence();
            {
                seq1->fill_vec(vk, 0.0); // TODO: Use a better way to clear vk
                seq1->insert_cc_barrier();
                compute_Ax(seq1, A, Hpp, Hpl, Hllinv, pk, vk, l0, l1, true);

                seq1->insert_cc_barrier();
                seq1->inner_product(pk, vk, ak_scratch, akd);
                seq1->insert_cc_barrier();
                seq1->div_vec(akn, akd, akd, 1); // akd[0] = akn[0]/akd[0], need akn[0] for later
                seq1->insert_cc_barrier();

                // compute xk1 = xk + ak*pk in same seq
                seq1->add_vec(x, pk, x, akd);
                // compute rk1 = rk - ak*vk
                seq1->add_vec(rk, vk, rk, akd, false);

                // zk1 = M^{-1}rk1
                seq1->fill_vec(zk, 0.0); // TODO: Use better way to initialize zk1=0
                seq1->insert_cc_barrier();
                Minv->subgroup_block_diagonal_multiply_vec_add(seq1, zk, rk);
                seq1->insert_cc_barrier();
                // bkn = dot(rk1, zk1) and bkd = dot(rk, zk)
                // we already have akn = dot(rk, zk)
                seq1->inner_product(rk, zk, ak_scratch, akd);
                seq1->insert_cc_barrier(); // need this
                // compute bkn
                seq1->div_vec(akd, akn, akn, 1);
                seq1->insert_cc_barrier();

                // seq 6: pk1 = zk1 + bk*pk
                auto zk1 = zk; // OK, since we no longer need zk given akn
                auto bk = akn;
                seq1->add_vec(zk1, pk, pk, bk);
                seq1->insert_cc_barrier(); // need this
                seq1->copy_vec(akd, akn, 1); // akn = a(k+1)n for next iteration
                // seq1->insert_cc_barrier();
                // compute dot(rk1, rk1)
                // seq1->self_inner_product(rk, ak_scratch, akd);
                // seq1->insert_cc_barrier();
                seq1->sync_local<DataType>({akd}, {{0, 1}});
            }
            // end of pre-loop initialization
            init = true;
        }

        bool solve(const size_t max_iter)
        {
            // Run transfers and initialization sequences
            init_seq->execute(); // r0, x0, p0
            const DataType rel_thresh = tol * (akn->map()[0]);

            const int reset_iter = 50;
            size_t k;
            // double exec_time = 0.0;

            // double e0 = 0.0;
            // double e1 = 0.0;
            DataType res = std::numeric_limits<DataType>::infinity();
            // std::cerr << "rel_thresh: " << rel_thresh << std::endl;
            // auto tb = std::chrono::high_resolution_clock::now();
            // std::cout << "Setup took: " << std::chrono::duration<double>(tb - ta).count() << std::endl;

            const DataType thresh = (absolute_tol && residual > 0.0 && residual > rel_thresh) ? residual : rel_thresh;

            for (k = 0; k < max_iter; k++)
            {
                // std::cout << "res: " << res << std::endl;
                if (res <= thresh)
                    break;
                // auto t0 = std::chrono::high_resolution_clock::now();
                if (k > 0 && k % reset_iter == 0)
                    reset_seq->execute();
                // e0 += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

                // auto t1 = std::chrono::high_resolution_clock::now();
                seq1->execute();
                // res = std::sqrt(akd->map()[0]);
                res = akd->map()[0];
                // e1 += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();
            }

            // std::cerr << "residual[" << k << "]: " << res << std::endl;
            // std::cerr << "thresh: " << thresh << std::endl;
            // std::cerr << "abs_res: " << residual << std::endl;
            residual = res * 0.5;
            // std::cout << "k = " << k << std::endl;
            // std::cout << "reset_seq: " << e0 << std::endl;
            // std::cout << "seq1: " << e1 << std::endl;
            return !(std::isnan(res) || std::isinf(res));
        }
    };
}