#include "Stats/LocalMappingStats.h"
#include <sstream>

using namespace std;

#ifdef REGISTER_LOCAL_MAPPING_STATS
namespace {
template <typename S>
void dumpSeries(const string &path, const S &s) {
    std::ofstream myfile(path);
    for (const auto &p : s) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
}
}
#endif

void LocalMappingStats::saveStats(const string &file_path) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    string data_path = file_path + "/LocalMapping";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[LocalMappingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }

    data_path = data_path + "/data/";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[LocalMappingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }
    cout << "Writing stats data into file: " << data_path << '\n';

    MappingKernelController::saveKernelsStats(data_path);

    dumpSeries(data_path + "/localMapping_time.txt",           localMapping_time);
    dumpSeries(data_path + "/processKF_time.txt",              processKF_time);
    dumpSeries(data_path + "/MPCulling_time.txt",              MPCulling_time);
    dumpSeries(data_path + "/MPCreation_time.txt",             MPCreation_time);
    dumpSeries(data_path + "/searchInNeighbors_time.txt",      searchInNeighbors_time);
    dumpSeries(data_path + "/LBA_time.txt",                    LBA_time);
    dumpSeries(data_path + "/KFCulling_time.txt",              KFCulling_time);
    dumpSeries(data_path + "/searchForTriangulation_time.txt", searchForTriangulation_time);
    dumpSeries(data_path + "/createdMappoints_num.txt",        createdMappoints_num);

    // GPU functions times
    dumpSeries(data_path + "/addCudaKeyFrame_time.txt",  addCudaKeyFrame_time);
    dumpSeries(data_path + "/eraseCudaKeyFrame_time.txt", eraseCudaKeyFrame_time);
    dumpSeries(data_path + "/addFeatureVector_time.txt", addFeatureVector_time);
#endif
}
