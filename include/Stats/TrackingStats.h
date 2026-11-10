#ifndef TRACKING_STATS_H
#define TRACKING_STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
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
        // NeedNewKeyFrame + CreateNewKeyFrame, recorded on every frame.
        std::vector<std::pair<long unsigned int, double>> createKF_time;
        std::vector<std::pair<long unsigned int, double>> updateLocalMap_time;
        std::vector<std::pair<long unsigned int, double>> updateLocalKF_time;
        std::vector<std::pair<long unsigned int, double>> updateLocalPoints_time;
        std::vector<std::pair<long unsigned int, double>> searchLocalPoints_time;
        std::vector<std::pair<long unsigned int, double>> SLP_frameMapPointsItr_time;
        std::vector<std::pair<long unsigned int, double>> SLP_localMapPointsItr_time;
        std::vector<std::pair<long unsigned int, double>> SLP_searchByProjection_time;
        std::vector<std::pair<long unsigned int, double>> TLM_poseOptimization_time;
        std::vector<std::pair<long unsigned int, int>> num_local_mappoints;
        // Diagnostics for tracking-loss investigation. num_local_mappoints alone cannot
        // distinguish "matched too few of a healthy local map" from "matched plenty but
        // the optimiser rejected them", and neither shows whether local mapping has
        // fallen behind tracking.
        std::vector<std::pair<long unsigned int, int>> num_slp_to_match;      // candidates in view
        std::vector<std::pair<long unsigned int, int>> num_slp_matches;       // SearchByProjection result
        std::vector<std::pair<long unsigned int, int>> num_matches_inliers;   // survivors of PoseOptimization
        std::vector<std::pair<long unsigned int, int>> localmapper_queue;     // KFs waiting in LocalMapping

        // (frame id, start ns, end ns) for the whole per-frame tracking call, on the
        // same steady_clock LoopClosing records its windows with. A frame counts as
        // overlapped by a loop closure when the two windows intersect.
        std::vector<std::tuple<long unsigned int, long long, long long>> frame_window;
        // (frame id, ns) for every frame on which NeedNewKeyFrame() declined to insert
        // a keyframe solely because a loop correction had Local Mapping stopped.
        std::vector<std::pair<long unsigned int, long long>> kfRefusedLMStop;

        double orbExtraction_init_time;
        double stereoMatch_init_time;
        double searchLocalPoints_init_time;
        double poseEstimation_init_time;
        int num_frames_lost;

    private:
        TrackingStats() = default; // Private constructor
};

#endif 