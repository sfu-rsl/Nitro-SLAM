#ifndef CUDA_MAPPOINT_H
#define CUDA_MAPPOINT_H

#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#define MAX_NUM_OBSERVATIONS 200

namespace TRACKING_DATA_WRAPPER {

class CudaMapPoint {
    public:
        CudaMapPoint();
        CudaMapPoint(ORB_SLAM3::MapPoint* mp);
    
    public:
        // For creating empty mapPoints instead of using null ptr
        bool isEmpty;

    // For searchByProjection in TrackLocalMap
    public:
        long unsigned int mnId;
        bool mbBad;
        bool mbTrackInView;
        bool mbTrackInViewR;
        float mTrackDepth;
        int mnTrackScaleLevel;
        float mTrackViewCos;
        float mTrackProjX;
        float mTrackProjY;
        int mnTrackScaleLevelR;
        float mTrackViewCosR;
        float mTrackProjXR;
        float mTrackProjYR;
        long unsigned int mnLastFrameSeen;
        int nObs;
        uint8_t mDescriptor[32];
        
    // For searchByProjection in PoseEstimation
    public:
        Eigen::Vector3f mWorldPos;
    };
}

namespace MAPPING_DATA_WRAPPER {

class CudaKeyFrame;

class CudaMapPoint {
    public:
        CudaMapPoint();
        CudaMapPoint(ORB_SLAM3::MapPoint* mp);
        void initialize();
        void setMemory(ORB_SLAM3::MapPoint* mp);
        void setObservations(int nObs, map<ORB_SLAM3::KeyFrame*, tuple<int,int>> observations);
        void setDescriptor(cv::Mat descriptor);
        void setMinDistance(float mfMinDistance);
        void setMaxDistance(float mfMaxDistance);
        void setNormalVector(Eigen::Vector3f mNormalVector);
        void setWorldPos(Eigen::Vector3f pos); 
        void setAsEmpty() { isEmpty = true; };
        void freeMemory();

    public:
        // For creating empty mapPoints instead of using null ptr
        bool isEmpty;

    public:
        long unsigned int mnId;
        bool mbBad;
        Eigen::Vector3f mWorldPos;
        float mfMaxDistance;
        float mfMinDistance;
        Eigen::Vector3f mNormalVector;
        uint8_t mDescriptor[32];
    };
}

#endif // CUDA_MAPPOINT_H