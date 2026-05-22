#ifndef KERNEL_H
#define KERNEL_H

#include "types.hpp"
#include <sycl/sycl.hpp>

namespace optimizer {

class Kernel {
public:

    static void ExpSO3Test(sycl::queue& queue, sycl::buffer<Mat3d, 1>& rotationBuffer, sycl::buffer<Vec3d, 1>& translationBuffer , int numEdges);
    static void LogSO3Test(sycl::queue& queue, sycl::buffer<Mat3d, 1>& rotationBuffer, int numEdges);

    static std::tuple<Mat3d, Vec3d> computeUpdates(const Mat3d& DR, const Vec3d& ut, 
                                        const Mat3d& Rwb0, const Vec3d& twb,
                                        const Mat3d& Rcb, const Vec3d& tcb);

    /*
    static void update(sycl::queue& queue, int dimension, bool isPositive, sycl::buffer<Mat3d, 1>& Rwb0Buffer, 
                          sycl::buffer<Vec3d, 1>& twbBuffer, sycl::buffer<Mat3d, 1>& RcwBuffer, 
                          sycl::buffer<Mat3d, 1>& RcbBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer, 
                          sycl::buffer<Vec3d, 1>& tcbBuffer, sycl::buffer<Mat3d, 1>& RcwUpdatePositiveBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdatePositiveBuffer, sycl::buffer<Mat3d, 1>& RcwUpdateNegativeBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdateNegativeBuffer, sycl::buffer<int, 1>& vertexXMapBuffer,
                          sycl::buffer<int, 1>& vertexYMapBuffer, int numEdges, int numVertices);

    static void computeError(sycl::queue& queue, int dimension, bool isPositive, 
                          sycl::buffer<Vec3d, 1>& translationBuffer,sycl::buffer<Mat3d, 1>& rotationBuffer,
                          sycl::buffer<Mat3d, 1>& RcwBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer,
                          sycl::buffer<Mat3d, 1>& RcwUpdateBuffer, sycl::buffer<Vec3d, 1>& tcwUpdateBuffer,
                          sycl::buffer<int, 1>& vertexXMapBuffer, sycl::buffer<int, 1>& vertexYMapBuffer,
                          sycl::buffer<double, 1>& jacobianBufferX, sycl::buffer<double, 1>& jacobianBufferY, 
                          sycl::buffer<int, 1>& fixedBuffer, int numEdges);
    */

    static void update(sycl::queue& queue, int dimension, sycl::buffer<Mat3d, 1>& Rwb0Buffer, 
                          sycl::buffer<Vec3d, 1>& twbBuffer, sycl::buffer<Mat3d, 1>& RcwBuffer, 
                          sycl::buffer<Mat3d, 1>& RcbBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer, 
                          sycl::buffer<Vec3d, 1>& tcbBuffer, sycl::buffer<Mat3d, 1>& RcwUpdatePositiveBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdatePositiveBuffer, sycl::buffer<Mat3d, 1>& RcwUpdateNegativeBuffer, 
                          sycl::buffer<Vec3d, 1>& tcwUpdateNegativeBuffer, int numEdges, int numVertices);

    static void update(sycl::queue& queue, double* values, sycl::buffer<Mat3d, 1>& Rwb0Buffer, 
                          sycl::buffer<Vec3d, 1>& twbBuffer, sycl::buffer<Mat3d, 1>& RcwBuffer, 
                          sycl::buffer<Mat3d, 1>& RcbBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer, 
                          sycl::buffer<Vec3d, 1>& tcbBuffer, int numEdges, int numVertices);

    static void computeError(sycl::queue& queue, int dimension, 
                          sycl::buffer<Vec3d, 1>& translationBuffer,sycl::buffer<Mat3d, 1>& rotationBuffer,
                          sycl::buffer<Mat3d, 1>& RcwBuffer, sycl::buffer<Vec3d, 1>& tcwBuffer,
                          sycl::buffer<Mat3d, 1>& RcwUpdatePositiveBuffer, sycl::buffer<Vec3d, 1>& tcwUpdatePositiveBuffer,
                          sycl::buffer<Mat3d, 1>& RcwUpdateNegativeBuffer, sycl::buffer<Vec3d, 1>& tcwUpdateNegativeBuffer,
                          sycl::buffer<int, 1>& vertexXMapBuffer, sycl::buffer<int, 1>& vertexYMapBuffer,
                          sycl::buffer<double, 1>& jacobianBufferX, sycl::buffer<double, 1>& jacobianBufferY,
                          sycl::buffer<int, 1>& fixedBuffer, int numEdges);
    static Mat3d NormalizeRotation(const Mat3d& R);
    static Mat3d ExpSO3(const Vec3d& w);
    static Vec3d LogSO3(const Mat3d& R);

    static void computeMultiplication(sycl::queue& queue, sycl::buffer<Mat3d, 1>& rotationBuffer, 
                                      sycl::buffer<Vec3d, 1>& translationBuffer, int numEdges);
};
} // namespace optimizer

#endif  // KERNEL_H