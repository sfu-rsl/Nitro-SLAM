#ifndef LOOP_CLOSING_STATS_H
#define LOOP_CLOSING_STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <mutex>
#include <algorithm>
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
        // Every series is keyed by mpCurrentKF->mnId so a consumer can join them per
        // Loop Closing iteration. Series that fire more than once per iteration
        // (searchByProjection, one entry per place-recognition candidate) repeat the key.
        typedef std::vector<std::pair<unsigned long, double>> Series;
        typedef std::vector<std::pair<unsigned long, long unsigned int>> CountSeries;

        typedef std::vector<std::pair<unsigned long, int>> FlagSeries;

        // Whole-iteration total, recorded on EVERY iteration that dequeued a keyframe --
        // not just the ones that corrected a loop. This is the denominator for "other".
        Series loopClosing_time;

        // Per-iteration 0/1 outcome flags, so the interesting iterations can be
        // selected by filtering rather than by keeping a parallel copy of the timings.
        //   loopDetected  -- region detection returned a loop candidate
        //   loopClosed    -- CorrectLoop() actually ran; the rows that matter
        //   loopRejected  -- detected but thrown out by the roll/pitch gate
        //   mergeDetected -- the iteration ran a map merge instead of a loop closure
        // loopDetected == loopClosed + loopRejected.
        FlagSeries loopDetected;
        FlagSeries loopClosed;
        FlagSeries loopRejected;
        FlagSeries mergeDetected;

        // Map size as the iteration began, before any correction rewrote it. Recorded
        // every iteration, so filtering on loopClosed gives the size at closure while
        // the full series still shows how the map grew.
        CountSeries numKFs;
        CountSeries numMPs;

        // Region detection: the full NewDetectCommonRegions() call, every iteration.
        Series placeRecognition_time;
        // Sub-part of region detection, one entry per candidate keyframe evaluated.
        Series searchByProjection_time;

        // Loop correction: the full CorrectLoop() call. Only on corrected loops.
        Series loopCorrection_time;
        // Sub-parts of loop correction. loopFusion covers pose propagation + map point
        // fusion up to (not including) the essential-graph optimization.
        Series loopFusion_time;
        Series searchAndFuse_time;
        Series graphOptimization_time;

        // Global BA. Runs on mpThreadGBA, so it is NOT contained in loopClosing_time --
        // do not subtract it when computing "other". Keyed by the loop keyframe id.
        // Written from that thread while the Loop Closing thread writes everything
        // else, and Shutdown() does not join it, so it can still be in flight when
        // saveStats runs: go through the recorder, and read it under the same lock.
        Series globalBA_time;

        void recordGlobalBA(unsigned long id, double ms) {
            std::lock_guard<std::mutex> lock(mGlobalBAMutex);
            globalBA_time.emplace_back(id, ms);
        }

    private:
        LoopClosingStats() = default; // Private constructor
        std::mutex mGlobalBAMutex;
};

#endif
