/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#ifndef RESIZE
#define RESIZE

#include <opencv2/core/hal/interface.h>

// Interpolation tables for one axis of one level, in cv::resize's fixed-point
// form: ofs[d] is the source index and alpha[2d], alpha[2d+1] the 11-bit
// weights of that sample and its successor.
void build_resize_tables(int srcLen, int dstLen, int *ofs, short *alpha);

void resize(const uchar *inputImage, int inputImageStep, uchar *pyramid,
            const int *levelCols, const int *levelRows, int maxLevel, int cols,
            int rows, const int *d_xofs, const short *d_xalpha,
            const int *d_yofs, const short *d_yalpha, cudaStream_t stream);

#endif
