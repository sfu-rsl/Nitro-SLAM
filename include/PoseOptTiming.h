#pragma once

// Timing for the three tracking pose-optimization variants.
//
// Timings are accumulated per variant and dumped once at process exit, which
// keeps per-frame printing (and its I/O jitter) out of the measured region.
//
// Which implementation runs is not decided here: FastTrack
// (TrackingKernelController::is_active) routes the pose optimizations to the
// fused single-kernel GPU solver, and everything else runs g2o exactly as stock
// ORB-SLAM3 does. See Tracking::TrackLocalMap.

#include <atomic>
#include <iosfwd>

namespace ORB_SLAM3 {

class PoseOptTiming {
public:
    enum Variant { VISUAL = 0, INERTIAL_KF = 1, INERTIAL_F = 2, NUM_VARIANTS = 3 };

    static void record(Variant v, double ms, int inliers, bool gpu) {
        Slot& s = slots()[v];
        s.count.fetch_add(1, std::memory_order_relaxed);
        s.total_inliers.fetch_add(inliers, std::memory_order_relaxed);
        // std::atomic<double> only grows fetch_add in C++20; CAS loops keep
        // this usable on the C++17 toolchain the rest of the tree builds with.
        double cur = s.total_ms.load(std::memory_order_relaxed);
        while (!s.total_ms.compare_exchange_weak(cur, cur + ms)) {}
        double prev = s.max_ms.load(std::memory_order_relaxed);
        while (ms > prev && !s.max_ms.compare_exchange_weak(prev, ms)) {}
        if (gpu) gpu_backend().store(true, std::memory_order_relaxed);
    }

    static void report(std::ostream& os);

private:
    struct Slot {
        std::atomic<long>   count{0};
        std::atomic<double> total_ms{0.0};
        std::atomic<double> max_ms{0.0};
        std::atomic<long>   total_inliers{0};
    };

    static Slot* slots() {
        static Slot s[NUM_VARIANTS];
        return s;
    }

    // Constant for a run: TrackingKernelController::is_active is set once at
    // startup, so this only records which backend served the calls.
    static std::atomic<bool>& gpu_backend() {
        static std::atomic<bool> b{false};
        return b;
    }

    struct Reporter { ~Reporter(); };
    static Reporter reporter_;
};

} // namespace ORB_SLAM3
