#include "Stats/LoopClosingStats.h"
#include <sstream>  

using namespace std;

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

    std::ofstream myfile;

    LoopClosingKernelController::saveKernelsStats(data_path);

    myfile.open(data_path + "/loopClosing_time.txt");
    for (size_t i = 0; i < loopClosing_time.size(); ++i) {
        myfile << i << ": " << loopClosing_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/loopCorrection_time.txt");
    for (size_t i = 0; i < loopCorrection_time.size(); ++i) {
        myfile << i << ": " << loopCorrection_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/searchAndFuse_time.txt");
    for (size_t i = 0; i < searchAndFuse_time.size(); ++i) {
        myfile << i << ": " << searchAndFuse_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/searchByProjection_time.txt");
    for (size_t i = 0; i < searchByProjection_time.size(); ++i) {
        myfile << i << ": " << searchByProjection_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/graphOptimization_time.txt");
    for (size_t i = 0; i < graphOptimization_time.size(); ++i) {
        myfile << i << ": " << graphOptimization_time[i] << std::endl;
    }
    myfile.close();

#endif
}