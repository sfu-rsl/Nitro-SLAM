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
#include <opencv2/core/hal/interface.h>

#include "gaussian_blur.h"
#include "ORBextractor.h"

// The 7x7 Gaussian is the outer product of a 7-tap 1-D one (that is what
// exp(-(h*h + w*w)/2s^2) factors into), so the blur runs as two passes of 7
// taps instead of one of 49. cv::GaussianBlur is separable for the same reason,
// which is what the CPU path this has to agree with does.
//
// BORDER_REFLECT_101 in both passes, matching cv::GaussianBlur. Clamping a
// linear index instead both failed to reflect and let horizontal offsets wrap
// into the neighbouring row.
__device__ __forceinline__ int reflect101_blur(int v, int len) {
    if (v < 0)
        return -v;
    if (v >= len)
        return 2 * (len - 1) - v;
    return v;
}

// Geometry of one pyramid level. Level 0 lives in the input image with the
// host's row step; the rest are packed at their own width.
struct BlurLevel {
    int cols, rows, srcStep;
    const uchar *src;
};

__device__ __forceinline__ BlurLevel blurLevel(int level, uint old_w, uint old_h,
                                               const float *scaleFactor,
                                               const uchar *original_img,
                                               const uchar *images,
                                               uint inputImageStep) {
    BlurLevel l;
    const float scale = scaleFactor[level];
    l.rows = round(old_h * 1 / scale);
    l.cols = round(old_w * 1 / scale);
    l.srcStep = level == 0 ? (int)inputImageStep : l.cols;
    l.src = level == 0 ? original_img : &images[level * old_w * old_h];
    return l;
}

// Pass 1: horizontal taps, kept in float so the two passes together produce the
// same value the single 49-tap pass did.
__global__ void gaussian_blur_h_kernel(uint old_h, uint old_w,
                                       const float *_scaleFactor,
                                       const uchar *original_img,
                                       const uchar *images, float *tmp,
                                       const float *kernel1d, uint maxLevel,
                                       uint inputImageStep) {
    const int level = blockIdx.z;
    if (level >= maxLevel)
        return;

    const BlurLevel l = blurLevel(level, old_w, old_h, _scaleFactor,
                                  original_img, images, inputImageStep);

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= l.cols || y >= l.rows)
        return;

    // Uniform across the block, so these are broadcast out of L1.
    float w[KW];
#pragma unroll
    for (int i = 0; i < KW; i++)
        w[i] = kernel1d[i];

    const uchar *row = l.src + y * l.srcStep;
    float acc = 0.f;
#pragma unroll
    for (int i = -KW / 2; i <= KW / 2; i++)
        acc += row[reflect101_blur(x + i, l.cols)] * w[i + KW / 2];

    tmp[level * old_w * old_h + y * l.cols + x] = acc;
}

// Pass 2: vertical taps over the horizontal result, rounded to the output plane.
__global__ void gaussian_blur_v_kernel(uint old_h, uint old_w,
                                       const float *_scaleFactor,
                                       const float *tmp,
                                       uchar *original_img_blurred,
                                       uchar *images_blurred,
                                       const float *kernel1d, uint maxLevel,
                                       uint inputImageStep) {
    const int level = blockIdx.z;
    if (level >= maxLevel)
        return;

    const float scale = _scaleFactor[level];
    const int rows = round(old_h * 1 / scale);
    const int cols = round(old_w * 1 / scale);

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;

    float w[KW];
#pragma unroll
    for (int i = 0; i < KH; i++)
        w[i] = kernel1d[i];

    const float *plane = &tmp[level * old_w * old_h];
    float acc = 0.f;
#pragma unroll
    for (int i = -KH / 2; i <= KH / 2; i++)
        acc += plane[reflect101_blur(y + i, rows) * cols + x] * w[i + KH / 2];

    const int dstStep = level == 0 ? (int)inputImageStep : cols;
    uchar *dst = level == 0 ? original_img_blurred
                            : &images_blurred[level * old_w * old_h];
    dst[y * dstStep + x] = (uchar)round(acc);
}

void gaussian_blur( uchar *images, uchar *inputImage, uchar *imagesBlured, uchar *inputImageBlured, const float *kernel1d, float *tmp, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream) {
    dim3 dg( ceil( (float)cols/64 ), ceil( (float)rows/8 ), maxLevel );
    dim3 db( 64, 8, 1 );

    gaussian_blur_h_kernel<<<dg, db, 0, cudaStream>>>(rows, cols, mvScaleFactor, inputImage, images, tmp, kernel1d, maxLevel, inputImageStep);
    gaussian_blur_v_kernel<<<dg, db, 0, cudaStream>>>(rows, cols, mvScaleFactor, tmp, inputImageBlured, imagesBlured, kernel1d, maxLevel, inputImageStep);
}
