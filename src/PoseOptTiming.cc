#include "PoseOptTiming.h"

#include <iomanip>
#include <iostream>

namespace ORB_SLAM3 {

PoseOptTiming::Reporter PoseOptTiming::reporter_;

static const char* variant_name(int v) {
    switch (v) {
        case PoseOptTiming::VISUAL:      return "PoseOptimization";
        case PoseOptTiming::INERTIAL_KF: return "PoseInertialLastKF";
        case PoseOptTiming::INERTIAL_F:  return "PoseInertialLastFrame";
        default:                         return "?";
    }
}

void PoseOptTiming::report(std::ostream& os) {
    Slot* s = slots();
    long total_calls = 0;
    double total_ms = 0.0;
    for (int v = 0; v < NUM_VARIANTS; v++) {
        total_calls += s[v].count.load();
        total_ms    += s[v].total_ms.load();
    }
    if (total_calls == 0) return;

    os << "\n[PO_TIMING] backend="
       << (gpu_backend().load() ? "gpu-fused" : "cpu-g2o") << "\n";
    os << std::fixed << std::setprecision(4);
    for (int v = 0; v < NUM_VARIANTS; v++) {
        const long n = s[v].count.load();
        if (n == 0) continue;
        const double t = s[v].total_ms.load();
        os << "[PO_TIMING] " << std::setw(22) << std::left << variant_name(v)
           << std::right
           << " calls=" << std::setw(6) << n
           << " mean_ms=" << std::setw(9) << (t / n)
           << " max_ms=" << std::setw(9) << s[v].max_ms.load()
           << " total_ms=" << std::setw(11) << t
           << " mean_inliers=" << std::setw(8)
           << (double(s[v].total_inliers.load()) / n) << "\n";
    }
    os << "[PO_TIMING] ALL calls=" << total_calls
       << " mean_ms=" << (total_ms / total_calls)
       << " total_ms=" << total_ms << std::endl;
}

PoseOptTiming::Reporter::~Reporter() {
    PoseOptTiming::report(std::cout);
}

} // namespace ORB_SLAM3
