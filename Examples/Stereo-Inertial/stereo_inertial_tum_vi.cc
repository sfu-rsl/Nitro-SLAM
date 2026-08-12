/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>
#include <cstring>

#include<opencv2/core/core.hpp>

#include<System.h>
#include<Stats/TrackingStats.h>
#include<Stats/LocalMappingStats.h>
#include<Stats/LoopClosingStats.h>
#include "Kernels/TrackingKernelController.h"
#include "Kernels/MappingKernelController.h"
#include "Kernels/LoopClosingKernelController.h"
#include "ImuTypes.h"

using namespace std;

void LoadImagesTUMVI(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

double ttrack_tot = 0;
int main(int argc, char **argv)
{   
    int min_num_argc = 7 + 7;
    if(argc < min_num_argc) {
        cerr << endl << "Usage: ./stereo_inertial_tum_vi path_to_vocabulary path_to_settings path_to_image_folder_1 path_to_image_folder_2 path_to_times_file path_to_imu_data (trajectory_file_name)" 
                    << "strStatsFile <FastTrack[0|1]> <TurboMap[0|1]> <FastLoop[0|1]> kernel_status_FT kernel_status_TM kernel_status_FL"  << endl;
        return 1;
    }

    bool run_FastTrack = (strcmp(argv[argc-6], "1") == 0);
    bool run_TurboMap = (strcmp(argv[argc-5], "1") == 0);
    bool run_FastLoop = (strcmp(argv[argc-4], "1") == 0);
    string strStatsFile = argv[argc-7];
    
    if (!run_FastTrack && !run_TurboMap && !run_FastLoop)
        cout << "Running the original ORB-SLAM3 code...\n";
    else {
        cout << "Running FastTrack->" << run_FastTrack << ", TurboMap->" << run_TurboMap << ", FastLoop->" << run_FastLoop << endl;
    }

    bool FT_orbExtractionEnabled = (argv[argc-3][0] == '1');
    bool FT_stereoMatchEnabled = (argv[argc-3][1] == '1');
    bool FT_searchLocalPointsEnabled = (argv[argc-3][2] == '1');
    bool FT_poseEstimationEnabled = (argv[argc-3][3] == '1');
    bool FT_poseOptimizationEnabled = (argv[argc-3][4] == '1');
    
    bool TM_searchForTriangulationEnabled = (argv[argc-2][0] == '1');
    bool TM_fuseEnabled = (argv[argc-2][1] == '1');
    bool TM_keyframeCullingEnabled = (argv[argc-2][2] == '1');
    bool TM_LBAEnabled = (argv[argc-2][3] == '1');
    // Optional 5th bit selects the rewritten triangulation search. Absent (4-char
    // status) keeps the original SearchForTriangulationKernel, so existing scripts are
    // unaffected.
    bool TM_newTriangulation = (strlen(argv[argc-2]) > 4 && argv[argc-2][4] == '1');
    // 6th bit selects CreateNewMapPointsGPU2 (search + geometry both on GPU).
    bool TM_gpu2 = (strlen(argv[argc-2]) > 5 && argv[argc-2][5] == '1');

    bool FL_mergedSearchByProjectionEnabled = (argv[argc-1][0] == '1');
    bool FL_merged3SearchByProjectionEnabled = (argv[argc-1][1] == '1');
    bool FL_searchAndFuseEnabled = (argv[argc-1][2] == '1');
    bool FL_singleSearchByProjectionEnabled = (argv[argc-1][3] == '1');
    bool FL_graphOptimizationEnabled = (argv[argc-1][4] == '1');
    bool FL_globalBAEnabled = (argv[argc-1][5] == '1');

    if (run_FastTrack) {
        TrackingKernelController::activate();
        TrackingKernelController::setGPURunMode(
            FT_orbExtractionEnabled, FT_stereoMatchEnabled, FT_searchLocalPointsEnabled, FT_poseEstimationEnabled, FT_poseOptimizationEnabled
        );

        cout << "Activated FastTrack Kernels are: (";
        if (FT_orbExtractionEnabled)
            cout << "OrbExtraction ";
        if (FT_stereoMatchEnabled)
            cout << "StereoMatch ";
        if (FT_searchLocalPointsEnabled)
            cout << "SearchLocalPoints ";
        if (FT_poseEstimationEnabled)
            cout << "PoseEstimation ";
        if (!FT_poseOptimizationEnabled)
            cout << "PoseOptimization_off";
        cout << ")\n";
    }
    
    if (run_TurboMap) {
        MappingKernelController::activate();
        MappingKernelController::setGPURunMode(TM_searchForTriangulationEnabled, TM_fuseEnabled, TM_keyframeCullingEnabled, TM_LBAEnabled);
        MappingKernelController::useNewTriangulation = TM_newTriangulation;
        MappingKernelController::useGPU2Pipeline = TM_gpu2;
        if (TM_gpu2)
            cout << "Using CreateNewMapPointsGPU2 (batched search + batched geometry)" << endl;
        else if (TM_newTriangulation)
            cout << "Using rewritten triangulation search (TriangulationMatchKernel)" << endl;

        cout << "Activated TurboMap Kernels are: (";
        if (TM_searchForTriangulationEnabled)
            cout << "SearchForTriangulation ";
        if (TM_fuseEnabled)
            cout << "Fuse ";
        if (TM_keyframeCullingEnabled)
            cout << "KeyframeCulling ";
        if (TM_LBAEnabled)
            cout << "LBA";
        cout << ")\n";
    }

    if (run_FastLoop) {
        LoopClosingKernelController::activate();
        LoopClosingKernelController::setGPURunMode(FL_mergedSearchByProjectionEnabled, FL_merged3SearchByProjectionEnabled, FL_searchAndFuseEnabled, FL_singleSearchByProjectionEnabled, FL_graphOptimizationEnabled, FL_globalBAEnabled);

        cout << "Activated FastLoop Kernels are: (";
        if (FL_mergedSearchByProjectionEnabled)
            cout << "MergedSearchByProjection ";
        if (FL_merged3SearchByProjectionEnabled)
            cout << "Merged3SearchByProjection ";
        if (FL_searchAndFuseEnabled)
            cout << "SearchAndFuse ";
        if (FL_singleSearchByProjectionEnabled)
            cout << "SingleSearchByProjection ";
        if (FL_graphOptimizationEnabled)
            cout << "GraphOptimization ";
        if (FL_globalBAEnabled)
            cout << "GlobalBA ";
        cout << ")\n";
    }
    
    argc-=7;
    
    const int num_seq = (argc-3)/4;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 4) == 1);
    string file_name;
    if (bFileName)
        file_name = string(argv[argc-1]);


    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageLeftFilenames;
    vector< vector<string> > vstrImageRightFilenames;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeftFilenames.resize(num_seq);
    vstrImageRightFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        LoadImagesTUMVI(string(argv[4*(seq+1)-1]), string(argv[4*(seq+1)]), string(argv[4*(seq+1)+1]), vstrImageLeftFilenames[seq], vstrImageRightFilenames[seq], vTimestampsCam[seq]);
        cout << "Total images: " << vstrImageLeftFilenames[seq].size() << endl;
        cout << "Total cam ts: " << vTimestampsCam[seq].size() << endl;
        cout << "first cam ts: " << vTimestampsCam[seq][0] << endl;

        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(string(argv[4*(seq+1)+2]), vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "Total IMU meas: " << vTimestampsImu[seq].size() << endl;
        cout << "first IMU ts: " << vTimestampsImu[seq][0] << endl;
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageLeftFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered

    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    /*cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "IMU data in the sequence: " << nImu << endl << endl;*/

    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, false, 0, file_name);
    
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    int proccIm = 0;

    size_t num_tracked = 0;
    double track_total = 0.0;

    for (seq = 0; seq<num_seq; seq++)
    {

        // Main loop
        cv::Mat imLeft, imRight;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {

            // Read image from file
            imLeft = cv::imread(vstrImageLeftFilenames[seq][ni],cv::IMREAD_GRAYSCALE);
            imRight = cv::imread(vstrImageRightFilenames[seq][ni],cv::IMREAD_GRAYSCALE);

            if(imageScale != 1.f)
            {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
                int width = imLeft.cols * imageScale;
                int height = imLeft.rows * imageScale;
                cv::resize(imLeft, imLeft, cv::Size(width, height));
                cv::resize(imRight, imRight, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
                t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }

            // clahe
            clahe->apply(imLeft,imLeft);
            clahe->apply(imRight,imRight);

            double tframe = vTimestampsCam[seq][ni];

            if(imLeft.empty() || imRight.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageLeftFilenames[seq][ni] << endl;
                return 1;
            }


            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0)
            {
                // cout << "t_cam " << tframe << endl;

                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni])
                {
                    // vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[first_imu],vGyro[first_imu],vTimestampsImu[first_imu]));
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    // cout << "t_imu = " << fixed << vImuMeas.back().t << endl;
                    first_imu[seq]++;
                }
            }

            /*cout << "first imu: " << first_imu[seq] << endl;
            cout << "first imu time: " << fixed << vTimestampsImu[seq][0] << endl;
            cout << "size vImu: " << vImuMeas.size() << endl;*/

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the image to the SLAM system
            SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);
            num_tracked++;

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

#ifdef REGISTER_TRACKING_STATS
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            TrackingStats::getInstance().tracking_time.emplace_back((long unsigned int)ni, t_track);
#endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ttrack_tot += ttrack;
            // std::cout << "ttrack: " << ttrack << std::endl;

            vTimesTrack[ni]=ttrack;
            track_total += ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
        }
        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }
    std::cout << "Tracked " << num_tracked << " images in " << track_total << " seconds." << std::endl;
    std::cout << "Frametime: " << (track_total / num_tracked) << std::endl;
    std::cout << "FPS: " << (num_tracked / track_total) << std::endl;


    // Stop all threads
    SLAM.Shutdown();
#ifdef REGISTER_TRACKING_STATS
    TrackingStats::getInstance().saveStats(strStatsFile);
#endif

#ifdef REGISTER_LOCAL_MAPPING_STATS
    LocalMappingStats::getInstance().saveStats(strStatsFile);
#endif

#ifdef REGISTER_LOOP_CLOSING_STATS
    LoopClosingStats::getInstance().saveStats(strStatsFile);
#endif

    // Save camera trajectory
    std::chrono::system_clock::time_point scNow = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(scNow);
    std::stringstream ss;
    ss << now;

    if (bFileName)
    {
        // const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        // const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        const string kf_file =  "kf_" + file_name + ".txt";
        const string f_file =  "f_" + file_name + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages[0]; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages[0]/2] << endl;
    cout << "mean tracking time: " << totaltime/proccIm << endl;

    return 0;
}

void LoadImagesTUMVI(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    cout << strPathLeft << endl;
    cout << strPathRight << endl;
    cout << strPathTimes << endl;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);

        if(!s.empty())
        {
            if (s[0] == '#')
                continue;

            int pos = s.find(' ');
            string item = s.substr(0, pos);

            vstrImageLeft.push_back(strPathLeft + "/" + item + ".png");
            vstrImageRight.push_back(strPathRight + "/" + item + ".png");

            double t = stod(item);
            vTimeStamps.push_back(t/1e9);
        }
    }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            vTimeStamps.push_back(data[0]/1e9);
            vAcc.push_back(cv::Point3f(data[4],data[5],data[6]));
            vGyro.push_back(cv::Point3f(data[1],data[2],data[3]));
        }
    }
}
