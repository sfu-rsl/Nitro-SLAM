/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#ifndef GAUSSIAN_BLUR
#define GAUSSIAN_BLUR

#include <opencv2/core/hal/interface.h>
#include <cuda.h>

// `kernel1d` is the KW-tap 1-D Gaussian whose outer product with itself is the
// KWxKH kernel; `tmp` is a cols*rows*maxLevel float scratch plane for the
// horizontal pass.
void gaussian_blur( uchar *images, uchar *inputImage, uchar *imagesBlured, uchar *inputImageBlured, const float *kernel1d, float *tmp, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream);

#endif