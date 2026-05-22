#pragma once
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
// #include <chrono>

namespace g2o::gpu::BAStats
{

    void initStats();

    double getInitTime();

    void saveStats(const char *path);


    class BlockSolverStatistics
    {
    public:
        BlockSolverStatistics()
        {
            memset(this, 0, sizeof(BlockSolverStatistics));
        }

        // Block Solver
        size_t hpp_rows;
        size_t hpp_cols;
        size_t hpp_nnz;

        size_t hpl_rows;
        size_t hpl_cols;
        size_t hpl_nnz;

        size_t hll_rows;
        size_t hll_cols;
        size_t hll_nnz;

        size_t hschur_rows;
        size_t hschur_cols;
        size_t hschur_nnz;

        double time;
        double time_linear_solver;
        double time_schur_complement;
        double time_landmark_delta;
        double time_total;

        // Optimizer loop
        size_t iteration;
        size_t lm_iterations;

        // Optimization algorithm
        double time_algo_compute_active_errors;
        double time_algo_active_robust_chi2;
        double time_algo_solve;
        double time_build_system;

        double time_compute_lambda;
        
        double time_solver_push;
        double time_solver_set_lambda;
        double time_solver_solve; // should be same as time_total, but from outside function
        double time_opt_update;
        double time_solver_restore;
        double time_algo_err_chi2;
        double time_algo_compute_scale;
        double time_opt_discard;
        double time_opt_pop;
        double time_algo_inner_loop;
        double time_iteration;

        double chi2;

    };

    class BAStatistics
    {
    public:
        BAStatistics(size_t id) : 
                                          id(id),
                                    num_keyframes(0),
                                  num_landmarks(0),
                                  num_observations(0),
                                  num_vertices(0),
                                  num_edges(0),
                                  
                                  time_collect(0.0),
                                  time_solver_setup(0.0),
                                  time_build_graph(0.0),
                                  
                                  time_opt_init(0.0),
                                  time_opt_compute_active_errors(0.0),
                                  time_opt_robust_chi2(0.0),
                                  time_opt_optimize(0.0),

                                  time_erase_outliers(0.0),
                                  time_recover(0.0),

                                  time_ba_total(0.0),
                                  time_init_vset(0.0),
                                  time_init_workspace(0.0),
                                  time_init_clear(0.0),
                                  time_init_aux_edges_vertices(0.0),
                                  time_init_active_edges(0.0),
                                  time_init_sort_graph(0.0),
                                  
                                  time_algo_init(0.0),
                                  time_init_build_index_mapping(0.0),
                                  time_solver_init(0.0),
                                  time_build_structure(0.0),
                                  mem_block_bytes(0),
                                  mem_allocation_bytes(0),
                                  initial_chi2(0.0)

        {}

        size_t id;
        size_t num_keyframes;
        size_t num_landmarks;
        size_t num_observations;
        size_t num_vertices;
        size_t num_edges;

        double time_ba_total;


        double time_collect;
        double time_solver_setup;
        double time_build_graph;

        double time_opt_init;
        double time_opt_compute_active_errors;
        double time_opt_robust_chi2;
        double time_opt_optimize;

        double time_erase_outliers;
        double time_recover;

        // Sparse Optimizer Initialization
        double time_init_vset;
        double time_init_workspace;
        double time_init_clear;
        double time_init_aux_edges_vertices;
        double time_init_active_edges;
        double time_init_sort_graph;
        double time_init_build_index_mapping;

        double time_algo_init;
        double time_solver_init;
        double time_build_structure;

        // memory
        size_t mem_block_bytes;
        size_t mem_allocation_bytes;

        double initial_chi2;

        std::vector<BlockSolverStatistics> solver_stats;

        std::string label;
    };

    BAStatistics &
    pushStats(const std::string& label);

}

std::ostream &operator<<(std::ostream &, const g2o::gpu::BAStats::BAStatistics &);
