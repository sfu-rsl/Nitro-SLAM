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

#include "descriptor.h"

// __device__ __constant__ int HALF_PATCH_SIZE_GPU;

// A warp per keypoint, 4 warps per block: 128 keypoints per level in flight,
// which covers a 512x512 TUM-VI level (under 700 corners on the busiest one) in
// a handful of trips round the stride loop.
#define DESCRIPTOR_THREADS 128
#define DESCRIPTOR_BLOCKS_PER_LEVEL 32

// BORDER_REFLECT_101, the border cv::copyMakeBorder puts around each level of
// the CPU pyramid. The GPU pyramid has no border, and a rotated 31-pixel patch
// reaches about 21 pixels from its centre while keypoints sit as close as 19 to
// the edge, so without this the outermost keypoints sample the neighbouring row
// - or the next pyramid level - and get a descriptor built from unrelated
// pixels.
__device__ __forceinline__ int reflect101(int v, int len) {
    if (v < 0)
        return -v;
    if (v >= len)
        return 2 * (len - 1) - v;
    return v;
}

// One warp per keypoint, one descriptor byte per lane. The 512 pattern lookups
// of a descriptor are scattered single-byte reads, so a thread doing all of them
// spends the kernel waiting on memory with nothing else to issue; splitting them
// 32 ways puts 32x as many loads in flight for the same keypoint and turns the
// 32 byte-stores into one coalesced write.
__device__ inline void comp_descr_lane(const uchar *image, ORB_SLAM3::GpuPoint &pt, const cv::Point *pattern, int imageStep, int cols, int rows, int lane) {
        const float factorPI = (float)(CV_PI/180.f);
        const float angle = (float)pt.angle*factorPI;
        const float a = (float)cos(angle), b = (float)sin(angle);

        const int cx = (int)pt.x, cy = (int)pt.y;
        const int step = imageStep;

        // Lane i owns descriptor byte i, which is the pattern's i-th group of 16.
        pattern += lane * 16;

#define GET_VALUE(idx) \
        image[reflect101(cy + (int)round(pattern[idx].x*b + pattern[idx].y*a), rows)*step + \
              reflect101(cx + (int)round(pattern[idx].x*a - pattern[idx].y*b), cols)]

        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        pt.descriptor[lane] = (uchar)val;

#undef GET_VALUE
    }

__global__ void compute_descriptor_kernel(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *pointsTotal, const uint *sizes, cv::Point* pattern, int inputImageStep, int maxLevel, const float *mvScaleFactor, int cols, int rows) {
    const int level = blockIdx.y;
    if (level >= maxLevel)
        return;

    const uint n = sizes[level];

    ORB_SLAM3::GpuPoint *points = &(pointsTotal[level*cols*rows]);

    const uchar* im[2] = {inputImage, &(images[level*cols*rows])};
    const int imIndex = (level == 0) * 0 + (level != 0) * 1;

    const float scale = mvScaleFactor[level];
    const int new_cols = round(cols * 1/scale);
    const int new_rows = round(rows * 1/scale);
    const int imageStep = (level == 0) * inputImageStep + (level != 0) * new_cols;

    const uchar *myImagePyrimid = im[imIndex];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const uint warpsPerBlock = blockDim.x >> 5;
    const uint stride = gridDim.x * warpsPerBlock;
    for (uint index = blockIdx.x * warpsPerBlock + warp; index < n;
         index += stride)
        comp_descr_lane(myImagePyrimid, points[index], pattern, imageStep,
                        new_cols, new_rows, lane);
}

void compute_descriptor(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, uint *sizes, int maxPointsLevel, cv::Point* pattern, int inputImageStep, int maxLevel, int cols, int rows, float *mvScaleFactor, cudaStream_t cudaStream){
    // One block row per level, sized to a normal frame's corner count rather
    // than to the buffer: `maxPointsLevel` is the buffer capacity (cols*rows),
    // and using it as the grid bound launched 2.1M threads per frame so that
    // ~1.7k of them could find a corner and the rest could read sizes[] and
    // exit. That launch cost more than every descriptor in the frame. The
    // counts only exist on the device at this point, so the surplus is handled
    // by a stride loop instead of by the grid.
    dim3 dg(DESCRIPTOR_BLOCKS_PER_LEVEL, maxLevel);
    dim3 db(DESCRIPTOR_THREADS, 1);

    compute_descriptor_kernel<<<dg, db, 0, cudaStream>>>(images, inputImage, points, sizes, pattern, inputImageStep, maxLevel, mvScaleFactor, cols, rows);
}
