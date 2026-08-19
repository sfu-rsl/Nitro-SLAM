#ifndef LOCAL_MAPPING_STATS_H
#define LOCAL_MAPPING_STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <mutex>
#include <chrono>
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

        // IMU initialization, and the VIBA1/VIBA2 re-initializations after it. Each
        // runs FullInertialBA with 100 iterations synchronously on this thread. The
        // 100 iterations sound alarming but the map is still small at init: measured
        // on room3 these are 25/57/72 ms against a 43 ms iteration mean, i.e. outlier
        // iterations but not pathological ones. Recorded inside InitializeIMU(), which
        // can be called more than once per iteration -- entries repeat the key and are
        // summed by the consumer.
        Series imuInit_time;
        // The FullInertialBA portion of the above, broken out so the CPU and GPU
        // implementations can be compared directly at this workload size.
        Series imuInitFIBA_time;
        // ScaleRefinement(), monocular only, same thread, same story.
        Series scaleRefinement_time;

        double searchForTriangulation_init_time;

        // GPU bookkeeping. Unlike everything above, these two are NOT written by the
        // Local Mapping thread alone: CudaKeyFrameAllocator::create runs from the
        // KeyFrame constructor (Tracking thread) as well as ProcessNewKeyFrame, and
        // ::destroy runs from KeyFrame::SetBadFlag on both Local Mapping and Loop
        // Closing. Go through the recorders below so the push_backs stay serialised.
        Series addCudaKeyFrame_time;
        Series eraseCudaKeyFrame_time;
        Series addFeatureVector_time;

        // InitializeIMU() and ScaleRefinement() have several early returns each, so
        // the timing is scoped rather than bracketed by hand.
        class ScopedTimer {
        public:
            ScopedTimer(Series &dst, unsigned long id)
                : mDst(dst), mId(id), mStart(std::chrono::steady_clock::now()) {}
            ~ScopedTimer() {
                double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                        std::chrono::steady_clock::now() - mStart).count();
                mDst.emplace_back(mId, ms);
            }
        private:
            Series &mDst;
            unsigned long mId;
            std::chrono::steady_clock::time_point mStart;
        };

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
