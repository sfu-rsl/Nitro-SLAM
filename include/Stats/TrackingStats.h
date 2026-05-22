#ifndef TRACKING_STATS_H
#define TRACKING_STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "Stats/StatsInterface.h"
#include "Kernels/CudaUtils.h"
#include "Kernels/TrackingKernelController.h"

using namespace std;

class TrackingStats: public StatsInterface {
    public:
        static TrackingStats& getInstance() {
            static TrackingStats instance;
            return instance;
        }
        void saveStats(const string &file_path) override;

    public:
        std::vector<std::pair<long unsigned int, double>> tracking_time;
        std::vector<std::pair<long unsigned int, double>> orbExtraction_time;
        std::vector<std::pair<long unsigned int, double>> stereoMatch_time;
        std::vector<std::pair<long unsigned int, double>> trackWithMotionModel_time;
        std::vector<std::pair<long unsigned int, double>> TWM_poseEstimation_time;
        std::vector<std::pair<long unsigned int, double>> TWM_poseOptimization_time;
        std::vector<std::pair<long unsigned int, double>> relocalization_time;
        std::vector<std::pair<long unsigned int, double>> trackLocalMap_time;
        std::vector<std::pair<long unsigned int, double>> updateLocalMap_time;
        std::vector<std::pair<long unsigned int, double>> updateLocalKF_time;
        std::vector<std::pair<long unsigned int, double>> updateLocalPoints_time;
        std::vector<std::pair<long unsigned int, double>> searchLocalPoints_time;
        std::vector<std::pair<long unsigned int, double>> SLP_frameMapPointsItr_time;
        std::vector<std::pair<long unsigned int, double>> SLP_localMapPointsItr_time;
        std::vector<std::pair<long unsigned int, double>> SLP_searchByProjection_time;
        std::vector<std::pair<long unsigned int, double>> TLM_poseOptimization_time;
        std::vector<std::pair<long unsigned int, int>> num_local_mappoints;

        double orbExtraction_init_time;
        double stereoMatch_init_time;
        double searchLocalPoints_init_time;
        double poseEstimation_init_time;
        int num_frames_lost;

    private:
        TrackingStats() = default; // Private constructor
};

#endif 