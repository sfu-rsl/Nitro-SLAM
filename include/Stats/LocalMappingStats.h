#ifndef LOCAL_MAPPING_STATS_H
#define LOCAL_MAPPING_STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <mutex>
#include <sys/stat.h>
#include <sys/types.h>
#include "Stats/StatsInterface.h"
#include "Kernels/CudaUtils.h"
#include "Kernels/MappingKernelController.h"

using namespace std;

class LocalMappingStats: public StatsInterface {
    public:
        static LocalMappingStats& getInstance() {
            static LocalMappingStats instance;
            return instance;
        }
        void saveStats(const string &file_path) override;

    public:
        // Every series is keyed by mpCurrentKeyFrame->mnId so a consumer can join them
        // per Local Mapping iteration. This matters because LBA and keyframe culling
        // are skipped whenever a new keyframe arrives mid-iteration, so those series
        // are shorter than the rest and cannot be joined positionally.
        typedef std::vector<std::pair<unsigned long, double>> Series;
        typedef std::vector<std::pair<unsigned long, int>> CountSeries;

        // Whole-iteration total; the denominator for "other".
        Series localMapping_time;

        // The five phases of an iteration, in order.
        Series processKF_time;
        Series MPCulling_time;
        Series MPCreation_time;        // keypoint search + triangulation
        Series searchInNeighbors_time; // map point fusion
        Series LBA_time;
        Series KFCulling_time;

        // Sub-part of MPCreation: the SearchForTriangulation matching only.
        Series searchForTriangulation_time;
        CountSeries createdMappoints_num;

        double searchForTriangulation_init_time;

        // GPU bookkeeping. Unlike everything above, these two are NOT written by the
        // Local Mapping thread alone: CudaKeyFrameAllocator::create runs from the
        // KeyFrame constructor (Tracking thread) as well as ProcessNewKeyFrame, and
        // ::destroy runs from KeyFrame::SetBadFlag on both Local Mapping and Loop
        // Closing. Go through the recorders below so the push_backs stay serialised.
        Series addCudaKeyFrame_time;
        Series eraseCudaKeyFrame_time;
        Series addFeatureVector_time;

        void recordAddCudaKeyFrame(unsigned long id, double ms) {
            std::lock_guard<std::mutex> lock(mCudaKeyFrameMutex);
            addCudaKeyFrame_time.emplace_back(id, ms);
        }
        void recordEraseCudaKeyFrame(unsigned long id, double ms) {
            std::lock_guard<std::mutex> lock(mCudaKeyFrameMutex);
            eraseCudaKeyFrame_time.emplace_back(id, ms);
        }

    private:
        LocalMappingStats() = default; // Private constructor
        std::mutex mCudaKeyFrameMutex;
};

#endif
