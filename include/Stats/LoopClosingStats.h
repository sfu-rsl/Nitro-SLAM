#ifndef LOOP_CLOSING_STATS_H
#define LOOP_CLOSING_STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "Stats/StatsInterface.h"
#include "Kernels/CudaUtils.h"
#include "Kernels/LoopClosingKernelController.h"

using namespace std;

class LoopClosingStats: public StatsInterface {
    public:
        static LoopClosingStats& getInstance() {
            static LoopClosingStats instance;
            return instance;
        }
        void saveStats(const string &file_path) override;

    public:
        std::vector<double> loopClosing_time, loopCorrection_time, searchAndFuse_time, searchByProjection_time, graphOptimization_time;

    private:
        LoopClosingStats() = default; // Private constructor
};

#endif 