#include "../include/Kernel.h"

namespace optimizer {

void Kernel::ExpSO3Test(sycl::queue& queue, sycl::buffer<Mat3d, 1>& rotationBuffer, sycl::buffer<Vec3d,1>& translationBuffer, int numEdges){
    sycl::buffer<Mat3d, 1> resultBuffer(numEdges);
    queue.submit([&](sycl::handler& cgh){
        auto translationAcc = translationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto rotationAcc = rotationBuffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto resultAcc = resultBuffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class computeJacobians>(sycl::range<1>(numEdges), [=](sycl::id<1> id){
            int edgeIdx = id;
            resultAcc[edgeIdx] = ExpSO3(translationAcc[edgeIdx]);
        });
    });
}

void Kernel::LogSO3Test(sycl::queue& queue, sycl::buffer<Mat3d, 1>& rotationBuffer, int numEdges){
    sycl::buffer<Vec3d, 1> resultBuffer(numEdges);
    queue.submit([&](sycl::handler& cgh){
        auto rotationAcc = rotationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto resultAcc = resultBuffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class computeJacobians>(sycl::range<1>(numEdges), [=](sycl::id<1> id){
            int edgeIdx = id;
            resultAcc[edgeIdx] = LogSO3(rotationAcc[edgeIdx]);
        });
    });
}

std::tuple<Mat3d, Vec3d> Kernel::computeUpdates(const Mat3d& DR, const Vec3d& ut, 
                                        const Mat3d& Rwb0, const Vec3d& twb,
                                        const Mat3d& Rcb, const Vec3d& tcb) 
{

    Mat3d new_Rwb = DR * Rwb0;
    Vec3d new_twb = twb + ut;

    // Compute new values for Rcw and tcw for X
    Mat3d Rbw = new_Rwb.transpose();
    Vec3d tbw = -Rbw * new_twb;

    Mat3d Rcw = Rcb * Rbw;
    Vec3d tcw = Rcb * tbw + tcb;

    return std::make_tuple(Rcw, tcw);
}

/*

void Kernel::update(sycl::queue& queue, int dimension, bool isPositive, sycl::buffer<Mat3d, 1>& Rwb0Buffer, 
                          sycl::buffer<Vec3d, 1>& twbBuffer, sycl::buffer<Mat3d, 1>& RcwBuffer, 
                          sycl::buffer<Mat3d, 1>& RcbBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer, 
                          sycl::buffer<Vec3d, 1>& tcbBuffer, sycl::buffer<Mat3d, 1>& RcwUpdatePositiveBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdatePositiveBuffer, sycl::buffer<Mat3d, 1>& RcwUpdateNegativeBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdateNegativeBuffer, sycl::buffer<int, 1>& vertexXMapBuffer,
                          sycl::buffer<int, 1>& vertexYMapBuffer, int numEdges, int numVertices)
{

        const double delta = 1.0e-9;
        double add_vi[4] = {0.0};
        std::fill(add_vi, add_vi+4, 0.0);
        add_vi[dimension] = isPositive ? delta : -delta;
        Vec3d ur = Eigen::Vector3d(0, 0, add_vi[0]);
        Vec3d ut = Eigen::Vector3d(add_vi[1], add_vi[2], add_vi[3]);

        Mat3d DR = ExpSO3(ur) * Eigen::Matrix3d::Identity(); // DR = dr * DR

        queue.submit([&](sycl::handler& cgh){
        auto Rwb0Acc = Rwb0Buffer.get_access<sycl::access::mode::read>(cgh);
        auto twbAcc = twbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcbAcc = tcbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcbAcc = RcbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwUpdateAcc = RcwUpdatePositiveBuffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto tcwUpdateAcc = tcwUpdatePositiveBuffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class updateVertex>(sycl::range<1>(numVertices), [=](sycl::id<1> id){
            int vertexIdx = id[0];

            // Compute new values for Rwb and twb for X
            Mat3d new_Rwb = DR * Rwb0Acc[vertexIdx]; // Rwb = DR * Rwb0

            Vec3d new_twb = twbAcc[vertexIdx] + ut; // twb += ut

            // Compute new values for Rcw and tcw for X
            Mat3d Rbw = new_Rwb.transpose(); // Rbw = Rwb.transpose()

            Vec3d tbw = -Rbw * new_twb; // tbw = -Rbw * twb

            RcwUpdateAcc[vertexIdx] = RcbAcc[vertexIdx] * Rbw; // Rcw = Rcb * Rbw
            tcwUpdateAcc[vertexIdx] = RcbAcc[vertexIdx] * tbw + tcbAcc[vertexIdx]; // tcw = Rcb * tbw + tcb
        });

    });
}

void Kernel::computeError(sycl::queue& queue, int dimension, bool isPositive, 
                          sycl::buffer<Vec3d, 1>& translationBuffer,sycl::buffer<Mat3d, 1>& rotationBuffer,
                          sycl::buffer<Mat3d, 1>& RcwBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer,
                          sycl::buffer<Mat3d, 1>& RcwUpdateBuffer, sycl::buffer<Vec3d, 1>& tcwUpdateBuffer,
                          sycl::buffer<int, 1>& vertexXMapBuffer, sycl::buffer<int, 1>& vertexYMapBuffer,
                          sycl::buffer<double, 1>& jacobianBufferX, sycl::buffer<double, 1>& jacobianBufferY,
                          sycl::buffer<int, 1>& fixedBuffer, int numEdges)
{
    const double delta = 1.0e-9;
    const double scalar = 1.0 / (2*delta);

    queue.submit([&](sycl::handler& cgh){
        auto translationAcc = translationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto rotationAcc = rotationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwAcc = RcwBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcwAcc = tcwBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwUpdateAcc = RcwUpdateBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcwUpdateAcc = tcwUpdateBuffer.get_access<sycl::access::mode::read>(cgh);
        auto vertexXMapAcc = vertexXMapBuffer.get_access<sycl::access::mode::read>(cgh);
        auto vertexYMapAcc = vertexYMapBuffer.get_access<sycl::access::mode::read>(cgh);
        auto jacobianXAcc = jacobianBufferX.get_access<sycl::access::mode::read_write>(cgh);
        auto jacobianYAcc = jacobianBufferY.get_access<sycl::access::mode::read_write>(cgh);


        cgh.parallel_for<class computeJacobians>(sycl::range<1>(numEdges), [=](sycl::id<1> id){
            int edgeIdx = id[0];
            int vertexXIdx = vertexXMapAcc[edgeIdx];
            int vertexYIdx = vertexYMapAcc[edgeIdx];

            Vec3d RotationalErrorX = LogSO3(RcwUpdateAcc[vertexXIdx] * (RcwAcc[vertexYIdx]).transpose()* rotationAcc[edgeIdx].transpose());
            Vec3d TranslationErrorX = RcwUpdateAcc[vertexXIdx]*(-RcwAcc[vertexYIdx].transpose()*tcwAcc[vertexYIdx])+tcwUpdateAcc[vertexXIdx]-translationAcc[edgeIdx];

            Vec3d RotationalErrorY = LogSO3(RcwAcc[vertexXIdx] * (RcwUpdateAcc[vertexYIdx]).transpose()* rotationAcc[edgeIdx].transpose());
            Vec3d TranslationErrorY = RcwAcc[vertexXIdx]*(-RcwUpdateAcc[vertexYIdx].transpose()*tcwUpdateAcc[vertexYIdx])+tcwAcc[vertexXIdx]-translationAcc[edgeIdx];

            // initial offset
            int baseIdx = edgeIdx * 4 * 6 + dimension * 6;

            // Column major format
            for(int i = 0; i < 3; ++i){
                int indexRot = baseIdx + i; // For rotational part
                int indexTrans = baseIdx + i + 3; // For translation part
                if(isPositive){
                    jacobianXAcc[indexRot] = RotationalErrorX[i];
                    jacobianXAcc[indexTrans] = TranslationErrorX[i];
                    jacobianYAcc[indexRot] = RotationalErrorY[i];
                    jacobianYAcc[indexTrans] = TranslationErrorY[i];
                } else {
                    jacobianXAcc[indexRot] = (jacobianXAcc[indexRot] - RotationalErrorX[i]) * scalar;
                    jacobianXAcc[indexTrans] = (jacobianXAcc[indexTrans] - TranslationErrorX[i]) * scalar;
                    jacobianYAcc[indexRot] = (jacobianYAcc[indexRot] - RotationalErrorY[i]) * scalar;
                    jacobianYAcc[indexTrans] = (jacobianYAcc[indexTrans] - TranslationErrorY[i]) * scalar;
                }
            }
        });
    });
}

*/

void Kernel::update(sycl::queue& queue, int dimension, sycl::buffer<Mat3d, 1>& Rwb0Buffer, 
                          sycl::buffer<Vec3d, 1>& twbBuffer, sycl::buffer<Mat3d, 1>& RcwBuffer, 
                          sycl::buffer<Mat3d, 1>& RcbBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer, 
                          sycl::buffer<Vec3d, 1>& tcbBuffer, sycl::buffer<Mat3d, 1>& RcwUpdatePositiveBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdatePositiveBuffer, sycl::buffer<Mat3d, 1>& RcwUpdateNegativeBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdateNegativeBuffer, int numEdges, int numVertices)
{
    const double delta = 1.0e-9;
    Vec3d ur_pos = Eigen::Vector3d::Zero(); 
    Vec3d ut_pos = Eigen::Vector3d::Zero();
    Vec3d ur_neg = Eigen::Vector3d::Zero();
    Vec3d ut_neg = Eigen::Vector3d::Zero();

    if(dimension == 0){
        ur_pos.z() = delta;
        ur_neg.z() = -delta;
    } else {
        ut_pos[dimension-1] = delta;
        ut_neg[dimension-1] = -delta;
    }

    Mat3d DR_pos = ExpSO3(ur_pos) * Eigen::Matrix3d::Identity(); // DR = dr * DR
    Mat3d DR_neg = ExpSO3(ur_neg) * Eigen::Matrix3d::Identity(); // DR = dr * DR

    sycl::event event = queue.submit([&](sycl::handler& cgh){
        auto Rwb0Acc = Rwb0Buffer.get_access<sycl::access::mode::read>(cgh);
        auto twbAcc = twbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcbAcc = tcbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcbAcc = RcbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwUpdatePositiveAcc = RcwUpdatePositiveBuffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto tcwUpdatePositiveAcc = tcwUpdatePositiveBuffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto RcwUpdateNegativeAcc = RcwUpdateNegativeBuffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto tcwUpdateNegativeAcc = tcwUpdateNegativeBuffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class updateDelta>(sycl::range<1>(numVertices), [=](sycl::id<1> id){
            int vertexIdx = id[0];

            auto [Rcw_pos, tcw_pos] = computeUpdates(DR_pos, ut_pos, Rwb0Acc[vertexIdx], twbAcc[vertexIdx], RcbAcc[vertexIdx], tcbAcc[vertexIdx]);
            RcwUpdatePositiveAcc[vertexIdx] = Rcw_pos;
            tcwUpdatePositiveAcc[vertexIdx] = tcw_pos;

            auto [Rcw_neg, tcw_neg] = computeUpdates(DR_neg, ut_neg, Rwb0Acc[vertexIdx], twbAcc[vertexIdx], RcbAcc[vertexIdx], tcbAcc[vertexIdx]);
            RcwUpdateNegativeAcc[vertexIdx] = Rcw_neg;
            tcwUpdateNegativeAcc[vertexIdx] = tcw_neg;
        });

    });
    // queue.wait();
    // uint64_t submit_time = event.template get_profiling_info<
    //                 sycl::info::event_profiling::command_submit>();
    // uint64_t start_time = event.template get_profiling_info<
    //         sycl::info::event_profiling::command_start>();
    // uint64_t end_time = event.template get_profiling_info<
    //         sycl::info::event_profiling::command_end>();

    // std::cout << "Submit time: " << submit_time << std::endl;
    // std::cout << "Start time: " << start_time << std::endl;
    // std::cout << "End time: " << end_time << std::endl;

}

void Kernel::update(sycl::queue& queue, double* values, sycl::buffer<Mat3d, 1>& Rwb0Buffer, 
                          sycl::buffer<Vec3d, 1>& twbBuffer, sycl::buffer<Mat3d, 1>& RcwBuffer, 
                          sycl::buffer<Mat3d, 1>& RcbBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer, 
                          sycl::buffer<Vec3d, 1>& tcbBuffer, int numEdges, int numVertices)
{
    Vec3d ur = Eigen::Vector3d::Zero();
    Vec3d ut = Eigen::Vector3d::Zero();
    ur[2] = values[0];
    ut[0] = values[1];
    ut[1] = values[2];
    ut[2] = values[3];

    Mat3d DR = ExpSO3(ur) * Eigen::Matrix3d::Identity(); // DR = dr * DR

    queue.submit([&](sycl::handler& cgh){
        auto Rwb0Acc = Rwb0Buffer.get_access<sycl::access::mode::read>(cgh);
        auto twbAcc = twbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcbAcc = tcbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcbAcc = RcbBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwAcc = RcwBuffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto tcwAcc = tcwBuffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class updateValue>(sycl::range<1>(numVertices), [=](sycl::id<1> id){
            int vertexIdx = id[0];

            auto [Rcw, tcw] = computeUpdates(DR, ut, Rwb0Acc[vertexIdx], twbAcc[vertexIdx], RcbAcc[vertexIdx], tcbAcc[vertexIdx]);

            RcwAcc[vertexIdx] = Rcw;
            tcwAcc[vertexIdx] = tcw;
        });
    });
}

void Kernel::computeError(sycl::queue& queue, int dimension, 
                          sycl::buffer<Vec3d, 1>& translationBuffer,sycl::buffer<Mat3d, 1>& rotationBuffer,
                          sycl::buffer<Mat3d, 1>& RcwBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer,
                          sycl::buffer<Mat3d, 1>& RcwUpdatePositiveBuffer, sycl::buffer<Vec3d, 1>& tcwUpdatePositiveBuffer,
                          sycl::buffer<Mat3d, 1>& RcwUpdateNegativeBuffer, sycl::buffer<Vec3d, 1>& tcwUpdateNegativeBuffer,
                          sycl::buffer<int, 1>& vertexXMapBuffer, sycl::buffer<int, 1>& vertexYMapBuffer,
                          sycl::buffer<double, 1>& jacobianBufferX, sycl::buffer<double, 1>& jacobianBufferY,
                          sycl::buffer<int, 1>& fixedBuffer, int numEdges)
{
    const double delta = 1.0e-9;
    const double scalar = 1.0 / (2*delta);

    sycl::event event = queue.submit([&](sycl::handler& cgh){
        auto translationAcc = translationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto rotationAcc = rotationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwAcc = RcwBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcwAcc = tcwBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwUpdatePositiveAcc = RcwUpdatePositiveBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcwUpdatePositiveAcc = tcwUpdatePositiveBuffer.get_access<sycl::access::mode::read>(cgh);
        auto RcwUpdateNegativeAcc = RcwUpdateNegativeBuffer.get_access<sycl::access::mode::read>(cgh);
        auto tcwUpdateNegativeAcc = tcwUpdateNegativeBuffer.get_access<sycl::access::mode::read>(cgh);
        auto vertexXMapAcc = vertexXMapBuffer.get_access<sycl::access::mode::read>(cgh);
        auto vertexYMapAcc = vertexYMapBuffer.get_access<sycl::access::mode::read>(cgh);
        auto jacobianXAcc = jacobianBufferX.get_access<sycl::access::mode::read_write>(cgh);
        auto jacobianYAcc = jacobianBufferY.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class computeJacobians>(sycl::range<1>(numEdges), [=](sycl::id<1> id){
            int edgeIdx = id[0];
            int vertexXIdx = vertexXMapAcc[edgeIdx];
            int vertexYIdx = vertexYMapAcc[edgeIdx];

            Mat3d edgeRotationTranspose = rotationAcc[edgeIdx].transpose();

            Vec3d RotationalErrorPositiveX = LogSO3(RcwUpdatePositiveAcc[vertexXIdx] * (RcwAcc[vertexYIdx]).transpose()* edgeRotationTranspose);
            Vec3d TranslationErrorPositiveX = RcwUpdatePositiveAcc[vertexXIdx]*(-RcwAcc[vertexYIdx].transpose()*tcwAcc[vertexYIdx])+tcwUpdatePositiveAcc[vertexXIdx]-translationAcc[edgeIdx];

            Vec3d RotationalErrorNegativeX = LogSO3(RcwUpdateNegativeAcc[vertexXIdx] * (RcwAcc[vertexYIdx]).transpose()* edgeRotationTranspose);
            Vec3d TranslationErrorNegativeX = RcwUpdateNegativeAcc[vertexXIdx]*(-RcwAcc[vertexYIdx].transpose()*tcwAcc[vertexYIdx])+tcwUpdateNegativeAcc[vertexXIdx]-translationAcc[edgeIdx];

            Vec3d RotationalErrorPositiveY = LogSO3(RcwAcc[vertexXIdx] * (RcwUpdatePositiveAcc[vertexYIdx]).transpose()* edgeRotationTranspose);
            Vec3d TranslationErrorPositiveY = RcwAcc[vertexXIdx]*(-RcwUpdatePositiveAcc[vertexYIdx].transpose()*tcwUpdatePositiveAcc[vertexYIdx])+tcwAcc[vertexXIdx]-translationAcc[edgeIdx];

            Vec3d RotationalErrorNegativeY = LogSO3(RcwAcc[vertexXIdx] * (RcwUpdateNegativeAcc[vertexYIdx]).transpose()* edgeRotationTranspose);
            Vec3d TranslationErrorNegativeY = RcwAcc[vertexXIdx]*(-RcwUpdateNegativeAcc[vertexYIdx].transpose()*tcwUpdateNegativeAcc[vertexYIdx])+tcwAcc[vertexXIdx]-translationAcc[edgeIdx];

            // initial offset
            // int baseIdx = edgeIdx * 4 * 6 + dimension * 6;

            // Column major format
            // for(int i = 0; i < 3; ++i){
            //     int rotationIndex = baseIdx + i; // For rotational part
            //     int translationIndex = baseIdx + i + 3; // For translation part
            //     jacobianXAcc[rotationIndex] = (RotationalErrorPositiveX[i] - RotationalErrorNegativeX[i]) * scalar;
            //     jacobianXAcc[translationIndex] = (TranslationErrorPositiveX[i] - TranslationErrorNegativeX[i]) * scalar;
            //     jacobianYAcc[rotationIndex] = (RotationalErrorPositiveY[i] - RotationalErrorNegativeY[i]) * scalar;
            //     jacobianYAcc[translationIndex] = (TranslationErrorPositiveY[i] - TranslationErrorNegativeY[i]) * scalar;
            // }

            int baseIdx = edgeIdx*15;
            if(dimension > 0){
                baseIdx += 6 + (dimension - 1)*3;
            }
            for(int i = 0; i < 3; ++i){
                if(dimension == 0) {
                    int rotationIndex = baseIdx + i; // For rotational part
                    jacobianXAcc[rotationIndex] = (RotationalErrorPositiveX[i] - RotationalErrorNegativeX[i]) * scalar;
                    jacobianYAcc[rotationIndex] = (RotationalErrorPositiveY[i] - RotationalErrorNegativeY[i]) * scalar;
                }
                
                int translationIndex;
                if(dimension == 0) {
                    translationIndex = baseIdx + i + 3; // For translation part of the first dimension
                } else {
                    translationIndex = baseIdx + i; // For translation part of the next three dimensions
                }
                
                jacobianXAcc[translationIndex] = (TranslationErrorPositiveX[i] - TranslationErrorNegativeX[i]) * scalar;
                jacobianYAcc[translationIndex] = (TranslationErrorPositiveY[i] - TranslationErrorNegativeY[i]) * scalar;
            }
        });

    });

}

void Kernel::computeMultiplication(sycl::queue& queue, sycl::buffer<Mat3d, 1>& rotationBuffer, 
                                   sycl::buffer<Vec3d, 1>& translationBuffer, int numEdges)
{
    sycl::buffer<Vec3d, 1> resultBuffer(numEdges);

    queue.submit([&](sycl::handler& cgh) {
        auto rotationAcc = rotationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto translationAcc = translationBuffer.get_access<sycl::access::mode::read>(cgh);
        auto resultAcc = resultBuffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class matrix_vector_multiplication>(sycl::range<1>(numEdges), [=](sycl::item<1> item) {
            size_t idx = item.get_id()[0];

            resultAcc[idx] = rotationAcc[idx] * translationAcc[idx];
        });
    });

    std::vector<Vec3d> result(numEdges);

    queue.submit([&](sycl::handler& cgh) {
        auto resultAcc = resultBuffer.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(resultAcc, result.data());
    }).wait();

    for (size_t i = 0; i < numEdges; ++i) {
        std::cout << "Result for edge " << i << ": "
                << result[i * 3] << ", "
                << result[i * 3 + 1] << ", "
                << result[i * 3 + 2] << std::endl;
    }
}

Mat3d Kernel::NormalizeRotation(const Mat3d& R) {
    Vec3d x = R.col(0);
    Vec3d y = R.col(1);
    Vec3d z = R.col(2);

    x.normalize();
    y -= x * (x.transpose() * y);
    y.normalize();
    z = x.cross(y);

    Mat3d result;
    result.col(0) = x;
    result.col(1) = y;
    result.col(2) = z;
    return result;
}

Mat3d Kernel::ExpSO3(const Vec3d& w) {
    Mat3d omegaHat;
    omegaHat << 0, -w(2), w(1),
                w(2), 0, -w(0),
                -w(1), w(0), 0;
    
    auto d2 = w(0)*w(0) + w(1)*w(1) + w(2)*w(2);
    auto d = sycl::sqrt(d2);

    Mat3d result;

    if(d < 1e-5) {
        result = Mat3d::Identity() + omegaHat + 0.5 * omegaHat * omegaHat;
    } else {
        result = Mat3d::Identity() + sycl::sin(d)/d * omegaHat + (1 - sycl::cos(d))/d2 * omegaHat * omegaHat;
    }

    return NormalizeRotation(result);
}

Vec3d Kernel::LogSO3(const Mat3d& R) {
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = sycl::acos(costheta);
    const double s = sycl::sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}


} // namespace optimizer