#include "Kernels/CudaWrappers/CudaMapPoint.h"
#include "Kernels/CudaKeyFrameStorage.h"
#include <iostream>
#include <map>
#include <tuple>

#ifdef TIME_MEASURMENT
#define TIMESTAMP_PRINT(msg) std::cout << "TimeStamp [CudaFrame]: " << msg << std::endl
#else
#define TIMESTAMP_PRINT(msg) do {} while (0)
#endif

namespace TRACKING_DATA_WRAPPER
{
    CudaMapPoint::CudaMapPoint() {
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;
        mbTrackInView = mp->mbTrackInView;
        mbTrackInViewR = mp->mbTrackInViewR;
        mTrackDepth = mp->mTrackDepth;
        mnTrackScaleLevel = mp->mnTrackScaleLevel;
        mTrackViewCos = mp->mTrackViewCos;
        mTrackProjX = mp->mTrackProjX;
        mTrackProjY = mp->mTrackProjY;
        mnTrackScaleLevelR = mp->mnTrackScaleLevelR;
        mTrackViewCosR = mp->mTrackViewCosR;
        mTrackProjXR = mp->mTrackProjXR;
        mTrackProjYR = mp->mTrackProjYR;
        mnLastFrameSeen = mp->mnLastFrameSeen;
        nObs = mp->Observations();
        const cv::Mat& descriptor = mp->GetDescriptor();
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mWorldPos = mp->GetWorldPos();
    }
}

namespace MAPPING_DATA_WRAPPER
{
    void CudaMapPoint::initialize() {}

    CudaMapPoint::CudaMapPoint() {
        initialize();
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        if (mp)
            isEmpty = false;
        else {
            isEmpty = true;
            return;
        }
        mnId = mp->mnId;
        mWorldPos = mp->GetWorldPos();
        mfMaxDistance = mp->GetMaxDistance();
        mfMinDistance = mp->GetMinDistance();
        mNormalVector = mp->GetNormal();
        const cv::Mat& descriptor = mp->GetDescriptor();
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mbBad = mp->isBad();
    }

    void CudaMapPoint::setMemory(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;
        mWorldPos = mp->GetWorldPos();
        mfMaxDistance = mp->GetMaxDistance();
        mfMinDistance = mp->GetMinDistance();
        mNormalVector = mp->GetNormal();
        const cv::Mat& descriptor = mp->GetDescriptor();
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mbBad = mp->isBad();
    }

    void CudaMapPoint::setDescriptor(cv::Mat descriptor) {
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
    }

    void CudaMapPoint::setMinDistance(float _mfMinDistance) {
        mfMinDistance = _mfMinDistance;
    }

    void CudaMapPoint::setMaxDistance(float _mfMaxDistance) {
        mfMaxDistance = _mfMaxDistance;
    }

    void CudaMapPoint::setNormalVector(Eigen::Vector3f _mNormalVector) {
        mNormalVector = _mNormalVector;
    }

    void CudaMapPoint::setWorldPos(Eigen::Vector3f pos) {
        mWorldPos = pos;
    }

    void CudaMapPoint::freeMemory() {}
}