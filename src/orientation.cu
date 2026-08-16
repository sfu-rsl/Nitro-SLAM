/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#include <cuda.h>
#include <iostream>

#include "ORBextractor.h"

#include "orientation.h"

// __device__ __constant__ int HALF_PATCH_SIZE_GPU;

// Matches descriptor.cu: enough threads for a normal frame's corners, with a
// stride loop for anything above that.
#define ORIENTATION_THREADS 128
#define ORIENTATION_BLOCKS_PER_LEVEL 16

__device__ inline float ic_angle_gpu(const uchar *image, int x, int y, int *u_max, int imageStep) {
    int m_01 = 0, m_10 = 0;

    int center_index = x + y * imageStep;

    const uchar* center = &(image[center_index]);

    // Treat the center line differently, v=0
    #pragma unroll
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    const int step = imageStep;
    #pragma unroll
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        const int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    float kp_dir = atan2f((float)m_01, (float)m_10);
    kp_dir += (kp_dir < 0) * (2.0f * CV_PI);
    kp_dir *= 180.0f / CV_PI;

    return kp_dir;
}

__global__ void compute_orientation_kernel(const uchar *images, const uchar *inputImage, ORB_SLAM3::GpuPoint *pointsTotal, const uint *sizes, int* umax, int inputImageStep, int maxLevel, const float *mvScaleFactor, int cols, int rows) {
    const int level = blockIdx.y;
    if (level >= maxLevel)
        return;

    const uint n = sizes[level];

    ORB_SLAM3::GpuPoint *points = &(pointsTotal[level*cols*rows]);

    const uchar* im[2] = {inputImage, &(images[level*cols*rows])};
    const int imIndex = (level == 0) * 0 + (level != 0) * 1;

    const float scale = mvScaleFactor[level];
    const int new_cols = round(cols * 1/scale);
    const int imageStep = (level == 0) * inputImageStep + (level != 0) * new_cols;

    const uchar *myImagePyrimid = im[imIndex];

    const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

    // umax is 16 ints read by every thread on every patch row; a copy in shared
    // memory keeps that off the L1 path.
    __shared__ int umaxShared[HALF_PATCH_SIZE + 1];
    for (int i = threadIdx.x; i <= HALF_PATCH_SIZE; i += blockDim.x)
        umaxShared[i] = umax[i];
    __syncthreads();

    const uint stride = gridDim.x * blockDim.x;
    for (uint index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
         index += stride) {
        const int x = points[index].x;
        const int y = points[index].y;
        const float angle = ic_angle_gpu(myImagePyrimid, x, y, umaxShared, imageStep);
        points[index].angle = angle;
        points[index].octave = level;
        points[index].size = scaledPatchSize;
    }
}

void compute_orientation(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, uint *sizes, int maxPointsLevel, int* umax, int inputImageStep, int maxLevel, int cols, int rows, float *mvScaleFactor, cudaStream_t cudaStream){
    // See compute_descriptor: `maxPointsLevel` is the buffer capacity, not the
    // corner count, so sizing the grid by it launched a thread per pixel per
    // level to do ~1.7k threads' work.
    dim3 dg(ORIENTATION_BLOCKS_PER_LEVEL, maxLevel);
    dim3 db(ORIENTATION_THREADS, 1);

    compute_orientation_kernel<<<dg, db, 0, cudaStream>>>(images, inputImage, points, sizes, umax, inputImageStep, maxLevel, mvScaleFactor, cols, rows);
}
