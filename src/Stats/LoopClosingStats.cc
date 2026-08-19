#include "Stats/LoopClosingStats.h"
#include <sstream>

using namespace std;

#ifdef REGISTER_LOOP_CLOSING_STATS
namespace {

template <typename S>
void dumpSeries(const string &path, const S &s) {
    std::ofstream myfile(path);
    for (const auto &p : s) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
}

} // namespace
#endif

void LoopClosingStats::saveStats(const string &file_path) {
#ifdef REGISTER_LOOP_CLOSING_STATS
    string data_path = file_path + "/LoopClosing";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[LoopClosingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }

    data_path = data_path + "/data/";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[LoopClosingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }
    cout << "Writing stats data into file: " << data_path << '\n';

    LoopClosingKernelController::saveKernelsStats(data_path);

    // Global BA may still be running on mpThreadGBA -- Shutdown() does not join it.
    Series globalBA_snapshot;
    {
        std::lock_guard<std::mutex> lock(mGlobalBAMutex);
        globalBA_snapshot = globalBA_time;
    }

    dumpSeries(data_path + "/loopClosing_time.txt",        loopClosing_time);
    dumpSeries(data_path + "/placeRecognition_time.txt",   placeRecognition_time);
    dumpSeries(data_path + "/searchByProjection_time.txt", searchByProjection_time);
    dumpSeries(data_path + "/loopCorrection_time.txt",     loopCorrection_time);
    dumpSeries(data_path + "/loopFusion_time.txt",         loopFusion_time);
    dumpSeries(data_path + "/searchAndFuse_time.txt",      searchAndFuse_time);
    dumpSeries(data_path + "/graphOptimization_time.txt",  graphOptimization_time);
    dumpSeries(data_path + "/merge_time.txt",              merge_time);
    dumpSeries(data_path + "/globalBA_time.txt",           globalBA_snapshot);

    // Per-iteration outcome flags and map size: filter on these to get the closures.
    dumpSeries(data_path + "/loopDetected.txt",            loopDetected);
    dumpSeries(data_path + "/loopClosed.txt",              loopClosed);
    dumpSeries(data_path + "/loopRejected.txt",            loopRejected);
    dumpSeries(data_path + "/mergeDetected.txt",           mergeDetected);
    dumpSeries(data_path + "/numKFs.txt",                  numKFs);
    dumpSeries(data_path + "/numMPs.txt",                  numMPs);

    int nClosed = 0;
    for (const auto &f : loopClosed) nClosed += f.second;
    cout << "[LoopClosingStats:] " << nClosed << " closure(s) recorded out of "
         << loopClosing_time.size() << " iteration(s)\n";
#endif
}
