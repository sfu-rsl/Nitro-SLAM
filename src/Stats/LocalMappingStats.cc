#include "Stats/LocalMappingStats.h"
#include <sstream>  

using namespace std;

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

    std::ofstream myfile;

    MappingKernelController::saveKernelsStats(data_path);

    myfile.open(data_path + "/localMapping_time.txt");
    for (size_t i = 0; i < localMapping_time.size(); ++i) {
        myfile << i << ": " << localMapping_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/processKF_time.txt");
    for (size_t i = 0; i < processKF_time.size(); ++i) {
        myfile << i << ": " << processKF_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/MPCulling_time.txt");
    for (size_t i = 0; i < MPCulling_time.size(); ++i) {
        myfile << i << ": " << MPCulling_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/MPCreation_time.txt");
    for (size_t i = 0; i < MPCreation_time.size(); ++i) {
        myfile << i << ": " << MPCreation_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/searchInNeighbors_time.txt");
    for (size_t i = 0; i < searchInNeighbors_time.size(); ++i) {
        myfile << i << ": " << searchInNeighbors_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/LBA_time.txt");
    for (size_t i = 0; i < LBA_time.size(); ++i) {
        myfile << i << ": " << LBA_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/KFCulling_time.txt");
    for (size_t i = 0; i < KFCulling_time.size(); ++i) {
        myfile << i << ": " << KFCulling_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/searchForTriangulation_time.txt");
    for (size_t i = 0; i < searchForTriangulation_time.size(); ++i) {
        myfile << i << ": " << searchForTriangulation_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/fuse_time.txt");
    for (size_t i = 0; i < fuse_time.size(); ++i) {
        myfile << i << ": " << fuse_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/createdMappoints_num.txt");
    for (size_t i = 0; i < createdMappoints_num.size(); ++i) {
        myfile << i << ": " << createdMappoints_num[i] << std::endl;
    }
    myfile.close();

    // GPU functions times

    myfile.open(data_path + "/addCudaKeyFrame_time.txt");
    for (size_t i = 0; i < addCudaKeyFrame_time.size(); ++i) {
        myfile << i << ": " << addCudaKeyFrame_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/eraseCudaKeyFrame_time.txt");
    for (size_t i = 0; i < eraseCudaKeyFrame_time.size(); ++i) {
        myfile << i << ": " << eraseCudaKeyFrame_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/addFeatureVector_time.txt");
    for (size_t i = 0; i < addFeatureVector_time.size(); ++i) {
        myfile << i << ": " << addFeatureVector_time[i] << std::endl;
    }
    myfile.close();

#endif
}