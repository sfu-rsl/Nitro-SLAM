#include "Stats/TrackingStats.h"
#include <sstream>  

using namespace std;

void TrackingStats::saveStats(const string &file_path) {
#ifdef REGISTER_TRACKING_STATS
    string data_path = file_path + "/Tracking";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[TrackingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }

    data_path = data_path + "/data/";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[TrackingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }
    cout << "Writing stats data into file: " << data_path << '\n';

    std::ofstream myfile;

    TrackingKernelController::saveKernelsStats(data_path);

    myfile.open(data_path + "/tracking_time.txt");
    for (const auto& p : tracking_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/orbExtraction_time.txt");
    for (const auto& p : orbExtraction_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/stereoMatch_time.txt");
    for (const auto& p : stereoMatch_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/trackWithMotionModel_time.txt");
    for (const auto& p : trackWithMotionModel_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/TWM_poseEstimation_time.txt");
    for (const auto& p : TWM_poseEstimation_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/TWM_poseOptimization_time.txt");
    for (const auto& p : TWM_poseOptimization_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/relocalization_time.txt");
    for (const auto& p : relocalization_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/trackLocalMap_time.txt");
    for (const auto& p : trackLocalMap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/createKF_time.txt");
    for (const auto& p : createKF_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/updateLocalMap_time.txt");
    for (const auto& p : updateLocalMap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/updateLocalKF_time.txt");
    for (const auto& p : updateLocalKF_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/updateLocalPoints_time.txt");
    for (const auto& p : updateLocalPoints_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/searchLocalPoints_time.txt");
    for (const auto& p : searchLocalPoints_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/SLP_frameMapPointsItr_time.txt");
    for (const auto& p : SLP_frameMapPointsItr_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/SLP_localMapPointsItr_time.txt");
    for (const auto& p : SLP_localMapPointsItr_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/SLP_searchByProjection_time.txt");
    for (const auto& p : SLP_searchByProjection_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/TLM_poseOptimization_time.txt");
    for (const auto& p : TLM_poseOptimization_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/num_local_mappoints.txt");
    for (const auto& p : num_local_mappoints) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/kernel_initialization_time.txt");
    myfile << "total_initialization_time: " << orbExtraction_init_time + stereoMatch_init_time + searchLocalPoints_init_time + poseEstimation_init_time << std::endl;
    myfile << "orbExtraction_initialization_time: " << orbExtraction_init_time << std::endl;
    myfile << "stereoMatch_initialization_time: " << stereoMatch_init_time << std::endl;
    myfile << "searchLocalPoints_initialization_time: " << searchLocalPoints_init_time << std::endl;
    myfile << "poseEstimation_initialization_time: " << poseEstimation_init_time << std::endl;
    myfile.close();

    myfile.open(data_path + "/num_frames_lost.txt");
    myfile << num_frames_lost << std::endl;
    myfile.close(); 
#endif
}