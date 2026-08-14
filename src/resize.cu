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
#include <stdio.h>

#include "resize.h"

// One pyramid level, resampled from the level above with the same arithmetic
// cv::resize(..., INTER_LINEAR) uses: 11-bit fixed point weights, a 22-bit
// descale, and the tables built by build_resize_tables().
//
// Two things here were wrong before and both moved every keypoint above level 0:
// the level was sampled straight from level 0 rather than from level-1 (so a
// level-7 pixel came from one bilinear tap on the full-resolution image instead
// of seven successive halvings), and the sample point was x*scale instead of
// OpenCV's (x+0.5)*scale-0.5, a half-pixel shift that grows with the level.
__global__ void resize_level_kernel(const uchar *src, int srcStep, uchar *dst,
                                    int dstRows, int dstCols, int dstStep,
                                    const int *xofs, const short *xalpha,
                                    const int *yofs, const short *yalpha) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstCols || y >= dstRows)
        return;

    const int sx = xofs[x];
    const int ax0 = xalpha[2 * x], ax1 = xalpha[2 * x + 1];
    const int sy = yofs[y];
    const int ay0 = yalpha[2 * y], ay1 = yalpha[2 * y + 1];

    const uchar *row0 = src + sy * srcStep;
    const uchar *row1 = row0 + srcStep;

    // Horizontal pass, then OpenCV's VResizeLinear<uchar,...> vertical combine.
    // The vertical step is not a plain 22-bit descale: it drops 4 bits off each
    // row sum first, which rounds differently and is what cv::resize actually
    // produces.
    const int v0 = row0[sx] * ax0 + row0[sx + 1] * ax1;
    const int v1 = row1[sx] * ax0 + row1[sx + 1] * ax1;
    const int v = (((ay0 * (v0 >> 4)) >> 16) + ((ay1 * (v1 >> 4)) >> 16) + 2) >> 2;

    dst[y * dstStep + x] = (uchar)v;
}

// Straight copy of the input into level 0 of the pyramid buffer, which the
// higher levels and StereoMatchKernel both read from.
__global__ void copy_level0_kernel(const uchar *src, int srcStep, uchar *dst,
                                   int rows, int cols, int dstStep) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;
    dst[y * dstStep + x] = src[y * srcStep + x];
}

// cv::resize's INTER_LINEAR tables, computed once per level because the image
// size never changes during a run. Mirrors the coefficient setup in OpenCV's
// resize.cpp so that a GPU level is bit-identical to the CPU one.
void build_resize_tables(int srcLen, int dstLen, int *ofs, short *alpha) {
    const double scale = (double)srcLen / (double)dstLen;
    for (int d = 0; d < dstLen; d++) {
        float f = (float)((d + 0.5) * scale - 0.5);
        int s = (int)floorf(f);
        f -= s;

        if (s < 0) {
            s = 0;
            f = 0.f;
        }
        if (s >= srcLen - 1) {
            s = srcLen > 1 ? srcLen - 2 : 0;
            f = 1.f;
        }

        ofs[d] = s;
        // saturate_cast<short> rounds to nearest, and OpenCV rounds both
        // coefficients independently rather than deriving one from the other.
        alpha[2 * d] = (short)lrintf((1.f - f) * 2048.f);
        alpha[2 * d + 1] = (short)lrintf(f * 2048.f);
    }
}

void resize(const uchar *inputImage, int inputImageStep, uchar *pyramid,
            const int *levelCols, const int *levelRows, int maxLevel, int cols,
            int rows, const int *d_xofs, const short *d_xalpha,
            const int *d_yofs, const short *d_yalpha, cudaStream_t stream) {
    dim3 db(32, 8);

    {
        dim3 dg(ceil((float)levelCols[0] / db.x), ceil((float)levelRows[0] / db.y));
        copy_level0_kernel<<<dg, db, 0, stream>>>(inputImage, inputImageStep,
                                                  pyramid, levelRows[0],
                                                  levelCols[0], levelCols[0]);
    }

    // Sequential by construction: level L reads the level L-1 the previous
    // launch just wrote, exactly as ComputePyramid() does on the CPU.
    for (int level = 1; level < maxLevel; level++) {
        const uchar *src = pyramid + (level - 1) * rows * cols;
        uchar *dst = pyramid + level * rows * cols;
        dim3 dg(ceil((float)levelCols[level] / db.x),
                ceil((float)levelRows[level] / db.y));
        resize_level_kernel<<<dg, db, 0, stream>>>(
                src, levelCols[level - 1], dst, levelRows[level],
                levelCols[level], levelCols[level], d_xofs + level * cols,
                d_xalpha + level * cols * 2, d_yofs + level * rows,
                d_yalpha + level * rows * 2);
    }
}
