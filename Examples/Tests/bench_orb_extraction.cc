/**
 * Times the GPU ORB extractor on real frames, one extractor, one thread.
 *
 * test_orb_extraction answers "is the GPU path correct"; this one answers "where
 * does the time go". It runs the same extractor over the same images repeatedly
 * so the numbers are steady, exits normally (so nsys/ncu can flush), and prints
 * a percentile summary rather than a mean, because the tail is what misses a
 * frame deadline.
 *
 * Set NITRO_ORB_PROFILE=1 to get the per-phase breakdown that ORBextractor
 * prints to stderr, e.g.
 *   NITRO_ORB_PROFILE=1 ./bench_orb_extraction <dir> 100 1 2>phases.csv
 *
 * Usage:
 *   bench_orb_extraction <image_dir> [num_images] [repeats] [cpu]
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ORBextractor.h"
#include "Kernels/TrackingKernelController.h"

using namespace std;
using namespace ORB_SLAM3;

namespace {

vector<string> listImages(const string &dir, size_t limit) {
    vector<string> all;
    DIR *d = opendir(dir.c_str());
    if (!d)
        return all;
    while (struct dirent *ent = readdir(d)) {
        const string name = ent->d_name;
        const size_t dot = name.rfind('.');
        if (dot == string::npos)
            continue;
        const string ext = name.substr(dot);
        if (ext == ".png" || ext == ".jpg" || ext == ".pgm")
            all.push_back(dir + "/" + name);
    }
    closedir(d);
    sort(all.begin(), all.end());
    if (all.size() > limit)
        all.resize(limit);
    return all;
}

double pct(vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    sort(v.begin(), v.end());
    const size_t i = min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5));
    return v[i];
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <image_dir> [num_images] [repeats] [cpu]\n", argv[0]);
        return 1;
    }
    const string imageDir = argv[1];
    const size_t numImages = argc > 2 ? stoul(argv[2]) : 50;
    const size_t repeats = argc > 3 ? stoul(argv[3]) : 4;
    const bool useCpu = argc > 4 && strcmp(argv[4], "cpu") == 0;

    const vector<string> paths = listImages(imageDir, numImages);
    if (paths.empty()) {
        fprintf(stderr, "No images found in %s\n", imageDir.c_str());
        return 1;
    }

    // Decode up front: image loading is not what is being measured.
    vector<cv::Mat> images;
    for (const string &p : paths) {
        cv::Mat im = cv::imread(p, cv::IMREAD_GRAYSCALE);
        if (!im.empty())
            images.push_back(im);
    }
    if (images.empty()) {
        fprintf(stderr, "No decodable images in %s\n", imageDir.c_str());
        return 1;
    }

    // TUM-VI ORB settings (Examples/Stereo-Inertial/TUM-VI_far.yaml).
    const int nFeatures = 1000, nLevels = 8, iniThFAST = 20, minThFAST = 7;
    const float scaleFactor = 1.2f;

    TrackingKernelController::orbExtractionKernelRunStatus = !useCpu;
    ORBextractor extractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST,
                           images[0].cols, images[0].rows);

    vector<int> lapping = {0, 0};
    vector<cv::KeyPoint> keys;
    cv::Mat desc;

    // One warm pass: the first call pays for lazy CUDA context work.
    extractor(images[0], cv::Mat(), keys, desc, lapping);

    vector<double> times;
    size_t keypointTotal = 0;
    const auto wall0 = chrono::steady_clock::now();
    for (size_t r = 0; r < repeats; r++) {
        for (const cv::Mat &im : images) {
            const auto t0 = chrono::steady_clock::now();
            extractor(im, cv::Mat(), keys, desc, lapping);
            times.push_back(
                    chrono::duration<double, milli>(
                            chrono::steady_clock::now() - t0).count());
            keypointTotal += keys.size();
        }
    }
    const double wall = chrono::duration<double>(
                                chrono::steady_clock::now() - wall0).count();

    printf("%s extractor, %zu images x %zu repeats = %zu calls\n",
           useCpu ? "CPU" : "GPU", images.size(), repeats, times.size());
    printf("keypoints/frame  %.0f\n", (double)keypointTotal / times.size());
    printf("per call ms      mean %.3f  p50 %.3f  p90 %.3f  p99 %.3f  max %.3f\n",
           accumulate(times.begin(), times.end(), 0.0) / times.size(),
           pct(times, 0.50), pct(times, 0.90), pct(times, 0.99),
           pct(times, 1.0));
    printf("throughput       %.1f extractions/s (%.1f stereo frames/s)\n",
           times.size() / wall, times.size() / wall / 2.0);
    return 0;
}
