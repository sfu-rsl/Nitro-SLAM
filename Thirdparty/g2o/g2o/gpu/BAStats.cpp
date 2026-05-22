#include "BAStats.hpp"
#include <vector>
#include "../stuff/timeutil.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <iomanip>

namespace g2o::gpu::BAStats
{

    static std::mutex mtx;
    static std::vector<std::unique_ptr<BAStatistics>> stats;
    double init_time;

    void initStats()
    {
        // init_time = std::chrono::high_resolution_clock::now();
        init_time = get_monotonic_time();
    }

    double getInitTime()
    {
        return init_time;
    }

    BAStatistics &pushStats(const std::string& label)
    {
        auto s = std::make_unique<BAStatistics>(stats.size());
        s->label = label;
        std::lock_guard<std::mutex> guard(mtx);
        stats.push_back(std::move(s));
        // std::cout << "stats size: " << stats.size() << "\n";
        return *stats.back().get();
    }

    void saveStats(const char *path)
    {
        auto file = std::ofstream(path);
        file << "id ";
        file << "iteration ";
        file << "num_keyframes ";
        file << "num_landmarks ";
        file << "num_observations ";
        file << "num_vertices ";
        file << "num_edges ";

        file << "time_ba_total ";
        // file << "time_optimize_total ";

        file << "time_collect ";
        file << "time_solver_setup ";
        file << "time_build_graph ";

        file << "time_opt_init ";
        file << "time_opt_compute_active_errors ";
        file << "time_opt_robust_chi2 ";
        file << "time_opt_optimize ";

        file << "time_erase_outliers ";
        file << "time_recover ";

        // new optimizer statistics

        file << "time_init_vset ";
        file << "time_init_workspace ";
        file << "time_init_clear ";
        file << "time_init_aux_edges_vertices ";
        file << "time_init_active_edges ";
        file << "time_init_sort_graph ";
        file << "time_init_build_index_mapping ";
        file << "time_build_structure ";

        // BlockSolver/Optimization Statistics
        file << "hpp_rows ";
        file << "hpp_cols ";
        file << "hpp_nnz ";

        file << "hpl_rows ";
        file << "hpl_cols ";
        file << "hpl_nnz ";

        file << "hll_rows ";
        file << "hll_cols ";
        file << "hll_nnz ";

        file << "hschur_rows ";
        file << "hschur_cols ";
        file << "hschur_nnz ";

        file << "time ";
        file << "time_solver_init ";
        file << "time_linear_solver ";
        file << "time_schur_complement ";
        file << "time_landmark_delta ";
        file << "time_total ";

        // new stats
        file << "time_algo_compute_active_errors ";
        file << "time_algo_active_robust_chi2 ";
        file << "time_build_system ";

        file << "time_compute_lambda ";
        
        file << "time_algo_solve ";
        file << "time_algo_init ";
        file << "time_solver_push ";
        file << "time_solver_set_lambda ";
        file << "time_solver_solve ";
        file << "time_opt_update ";
        file << "time_solver_restore ";
        file << "time_algo_err_chi2 ";

        file << "time_algo_compute_scale ";
        file << "time_opt_discard ";
        file << "time_opt_pop ";
        file << "time_algo_inner_loop ";
        file << "time_iteration ";

        // memory statistics
        file << "mem_block_bytes ";
        file << "mem_allocation_bytes ";
        file << "initial_chi2 ";
        file << "chi2 ";
        file << "lm_iterations ";
        

        file << "label\n";


        for (auto &s : stats)
        {
            file << *s;
        }
    }

}

std::ostream &operator<<(std::ostream &os, const g2o::gpu::BAStats::BAStatistics &s)
{

    os << std::setprecision(17);
    for (auto & t: s.solver_stats) {

        os << s.id << " ";
        os << t.iteration << " ";
        os << s.num_keyframes << " ";
        os << s.num_landmarks << " ";
        os << s.num_observations << " ";

        os << s.num_vertices << " ";
        os << s.num_edges << " ";

        os << s.time_ba_total << " ";
        // os << s.time_optimize_total << " ";
        os << s.time_collect << " ";
        os << s.time_solver_setup << " ";
        os << s.time_build_graph << " ";
        
        os << s.time_opt_init << " ";
        os << s.time_opt_compute_active_errors << " ";
        os << s.time_opt_robust_chi2 << " ";
        os << s.time_opt_optimize << " ";

        os << s.time_erase_outliers << " ";
        os << s.time_recover << " ";

        // new optimizer statistics

        os << s.time_init_vset << " ";
        os << s.time_init_workspace << " ";
        os << s.time_init_clear << " ";
        os << s.time_init_aux_edges_vertices << " ";
        os << s.time_init_active_edges << " ";
        os << s.time_init_sort_graph << " ";
        os << s.time_init_build_index_mapping << " ";
        os << s.time_build_structure << " ";

        // BlockSolver/Optimization Statistics

        os << t.hpp_rows << " ";
        os << t.hpp_cols << " ";
        os << t.hpp_nnz << " ";

        os << t.hpl_rows << " ";
        os << t.hpl_cols << " ";
        os << t.hpl_nnz << " ";

        os << t.hll_rows << " ";
        os << t.hll_cols << " ";
        os << t.hll_nnz << " ";

        os << t.hschur_rows << " ";
        os << t.hschur_cols << " ";
        os << t.hschur_nnz << " ";

        os << t.time << " ";
        os << s.time_solver_init << " ";
        os << t.time_linear_solver << " ";
        os << t.time_schur_complement << " ";
        os << t.time_landmark_delta << " ";
        os << t.time_total << " ";

        // new stats
        os << t.time_algo_compute_active_errors << " ";
        os << t.time_algo_active_robust_chi2  << " ";
        os << t.time_build_system  << " ";

        os << t.time_compute_lambda  << " ";
        
        os << t.time_algo_solve << " ";
        os << s.time_algo_init  << " ";
        os << t.time_solver_push  << " ";
        os << t.time_solver_set_lambda  << " ";
        os << t.time_solver_solve  << " ";
        os << t.time_opt_update  << " ";
        os << t.time_solver_restore  << " ";
        os << t.time_algo_err_chi2  << " ";

        os << t.time_algo_compute_scale  << " ";
        os << t.time_opt_discard  << " ";
        os << t.time_opt_pop  << " ";
        os << t.time_algo_inner_loop << " ";
        os << t.time_iteration << " ";

        // memory statistics
        os << s.mem_block_bytes << " ";
        os << s.mem_allocation_bytes << " ";

        // other ba stats
        os << s.initial_chi2 << " ";
        os << t.chi2 << " ";
        os << t.lm_iterations << " ";

        os << "\"" << s.label << "\"" << "\n";

    }


    return os;
};