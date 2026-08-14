/**
 * Compares the GPU ORB extractor against the stock CPU one on real frames.
 *
 * The ablation found GPU ORB extraction to be the one kernel that kills loop
 * closure on outdoors5 (0/3 detected), while tracking still limps along. That
 * is the signature of descriptors that are self-inconsistent rather than merely
 * different, so this harness measures three things that a full dataset run
 * cannot isolate:
 *
 *   1. determinism  -- extract the same image twice on the GPU and diff
 *   2. agreement    -- keypoint sets and descriptors against the CPU path
 *   3. recognition  -- DBoW2 score between frames, CPU vs GPU
 *
 * (3) is the one that matters: place recognition compares BoW vectors, so if
 * the GPU's scores between neighbouring frames collapse relative to the CPU's,
 * loop detection cannot work no matter how good tracking looks.
 *
 * Usage:
 *   test_orb_extraction <image_dir> [num_images] [stride] [vocabulary.txt]
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Converter.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Kernels/TrackingKernelController.h"

using namespace std;
using namespace ORB_SLAM3;

namespace {

struct Extraction {
    vector<cv::KeyPoint> keys;
    cv::Mat descriptors;
};

// Keypoints come back in image coordinates (level coordinates * scale), so undo
// the scale to recover the integer pixel the detector actually fired on.
struct KeyId {
    int octave;
    int x;
    int y;
    bool operator==(const KeyId &o) const {
        return octave == o.octave && x == o.x && y == o.y;
    }
};

struct KeyIdHash {
    size_t operator()(const KeyId &k) const {
        return (static_cast<size_t>(k.octave) << 40) ^
               (static_cast<size_t>(k.x) << 20) ^ static_cast<size_t>(k.y);
    }
};

KeyId idOf(const cv::KeyPoint &kp, const vector<float> &invScale) {
    const float inv = invScale[kp.octave];
    return KeyId{kp.octave, static_cast<int>(lroundf(kp.pt.x * inv)),
                 static_cast<int>(lroundf(kp.pt.y * inv))};
}

int hamming(const uchar *a, const uchar *b) {
    int dist = 0;
    for (int i = 0; i < 32; i++)
        dist += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
    return dist;
}

Extraction extract(ORBextractor &extractor, const cv::Mat &im, bool gpu,
                   double *elapsedMs = nullptr) {
    TrackingKernelController::orbExtractionKernelRunStatus = gpu;
    Extraction out;
    vector<int> lapping = {0, 0};
    const auto t0 = chrono::steady_clock::now();
    extractor(im, cv::Mat(), out.keys, out.descriptors, lapping);
    if (elapsedMs)
        *elapsedMs = chrono::duration<double, milli>(
                             chrono::steady_clock::now() - t0).count();
    return out;
}

// Fraction of `a`'s keypoints that also appear in `b`, and the descriptor
// distance over that intersection.
struct Agreement {
    size_t aCount = 0;      // keypoints in a
    size_t bCount = 0;      // keypoints in b
    size_t common = 0;      // keypoints of a that b also found
    size_t identical = 0;   // of those, with a bit-identical descriptor
    long long distSum = 0;  // Hamming distance summed over the intersection
    double meanHamming = 0.0;
    double meanAngleDiff = 0.0;
};

Agreement compare(const Extraction &a, const Extraction &b,
                  const vector<float> &invScale,
                  vector<Agreement> *perLevel = nullptr) {
    unordered_map<KeyId, int, KeyIdHash> index;
    index.reserve(b.keys.size() * 2);
    for (size_t i = 0; i < b.keys.size(); i++)
        index.emplace(idOf(b.keys[i], invScale), static_cast<int>(i));

    const size_t nLevels = perLevel ? perLevel->size() : 0;
    auto levelOf = [&](int lvl) -> Agreement * {
        if (!perLevel || lvl < 0 || lvl >= static_cast<int>(nLevels))
            return nullptr;
        return &(*perLevel)[lvl];
    };

    Agreement ag;
    ag.aCount = a.keys.size();
    ag.bCount = b.keys.size();
    double angleSum = 0.0;
    for (const cv::KeyPoint &kp : b.keys)
        if (Agreement *lv = levelOf(kp.octave))
            lv->bCount++;

    for (size_t i = 0; i < a.keys.size(); i++) {
        Agreement *lv = levelOf(a.keys[i].octave);
        if (lv)
            lv->aCount++;
        auto it = index.find(idOf(a.keys[i], invScale));
        if (it == index.end())
            continue;
        ag.common++;
        const int d = hamming(a.descriptors.ptr<uchar>(static_cast<int>(i)),
                              b.descriptors.ptr<uchar>(it->second));
        ag.distSum += d;
        if (d == 0)
            ag.identical++;
        if (lv) {
            lv->common++;
            lv->distSum += d;
            if (d == 0)
                lv->identical++;
        }
        float da = fabsf(a.keys[i].angle - b.keys[it->second].angle);
        if (da > 180.0f)
            da = 360.0f - da;
        angleSum += da;
    }
    if (ag.common) {
        ag.meanHamming = static_cast<double>(ag.distSum) / ag.common;
        ag.meanAngleDiff = angleSum / ag.common;
    }
    return ag;
}

vector<string> listImages(const string &dir, size_t limit, size_t stride) {
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

    vector<string> picked;
    for (size_t i = 0; i < all.size() && picked.size() < limit; i += stride)
        picked.push_back(all[i]);
    return picked;
}

double mean(const vector<double> &v) {
    if (v.empty())
        return 0.0;
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " <image_dir> [num_images] [stride] [vocabulary.txt]" << endl;
        return 1;
    }

    const string imageDir = argv[1];
    const size_t numImages = argc > 2 ? stoul(argv[2]) : 20;
    const size_t stride = argc > 3 ? stoul(argv[3]) : 1;
    const string vocPath = argc > 4 ? argv[4] : "";

    const vector<string> images = listImages(imageDir, numImages, stride);
    if (images.empty()) {
        cerr << "No images found in " << imageDir << endl;
        return 1;
    }

    cv::Mat first = cv::imread(images[0], cv::IMREAD_GRAYSCALE);
    if (first.empty()) {
        cerr << "Failed to read " << images[0] << endl;
        return 1;
    }

    // TUM-VI ORB settings (Examples/Stereo-Inertial/TUM-VI_far.yaml).
    const int nFeatures = 1000, nLevels = 8, iniThFAST = 20, minThFAST = 7;
    const float scaleFactor = 1.2f;

    // Deliberately leaked: the destructor frees device memory based on the
    // global flag, which this harness flips per call, so letting either one run
    // would free buffers the other owns (or buffers the CPU one never had).
    TrackingKernelController::orbExtractionKernelRunStatus = false;
    ORBextractor *cpuExtractorPtr = new ORBextractor(
            nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST, first.cols,
            first.rows);
    TrackingKernelController::orbExtractionKernelRunStatus = true;
    ORBextractor *gpuExtractorPtr = new ORBextractor(
            nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST, first.cols,
            first.rows);
    ORBextractor &cpuExtractor = *cpuExtractorPtr;
    ORBextractor &gpuExtractor = *gpuExtractorPtr;

    const vector<float> invScale = cpuExtractor.GetInverseScaleFactors();

    ORBVocabulary voc;
    bool haveVoc = false;
    if (!vocPath.empty()) {
        cout << "Loading vocabulary from " << vocPath << " ..." << flush;
        haveVoc = voc.loadFromTextFile(vocPath);
        cout << (haveVoc ? " ok" : " FAILED") << endl;
    }

    vector<DBoW2::BowVector> cpuBow, gpuBow;
    vector<double> selfScores;   // GPU run 1 vs GPU run 2, same image
    vector<double> crossScores;  // CPU vs GPU, same image
    vector<double> cpuCounts, gpuCounts;
    vector<double> commonFrac, meanHam, identFrac, angleDiff, gpuDeterm;
    vector<Agreement> perLevelCpu(nLevels), perLevelSelf(nLevels);
    vector<double> cpuTimes, gpuTimes;
    vector<double> pyrMaxDiff(nLevels, 0.0), pyrDiffPct(nLevels, 0.0);
    vector<int> pyrFrames(nLevels, 0), pyrSizeMismatch(nLevels, 0);

    printf("%-28s %7s %7s %8s %8s %8s %8s\n", "image", "nCPU", "nGPU",
           "common%", "meanHam", "ident%", "gpuDet%");

    for (const string &path : images) {
        cv::Mat im = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (im.empty())
            continue;

        double cpuMs = 0.0, gpuMs = 0.0;
        Extraction cpu = extract(cpuExtractor, im, false, &cpuMs);
        Extraction gpu1 = extract(gpuExtractor, im, true, &gpuMs);
        Extraction gpu2 = extract(gpuExtractor, im, true);
        cpuTimes.push_back(cpuMs);
        gpuTimes.push_back(gpuMs);

        const Agreement vsCpu = compare(cpu, gpu1, invScale, &perLevelCpu);
        const Agreement vsSelf = compare(gpu1, gpu2, invScale, &perLevelSelf);

        const double commonPct =
                cpu.keys.empty() ? 0.0 : 100.0 * vsCpu.common / cpu.keys.size();
        const double identPct =
                vsCpu.common ? 100.0 * vsCpu.identical / vsCpu.common : 0.0;
        const double determPct =
                gpu1.keys.empty() ? 0.0 : 100.0 * vsSelf.common / gpu1.keys.size();

        cpuCounts.push_back(cpu.keys.size());
        gpuCounts.push_back(gpu1.keys.size());
        commonFrac.push_back(commonPct);
        meanHam.push_back(vsCpu.meanHamming);
        identFrac.push_back(identPct);
        angleDiff.push_back(vsCpu.meanAngleDiff);
        gpuDeterm.push_back(determPct);

        printf("%-28s %7zu %7zu %8.1f %8.2f %8.1f %8.1f\n",
               path.substr(path.rfind('/') + 1).c_str(),
               cpu.keys.size(), gpu1.keys.size(), commonPct, vsCpu.meanHamming,
               identPct, determPct);

        // The pyramid the GPU built vs the one cv::resize built. Descriptors are
        // sampled from these, so any difference here lower-bounds the
        // descriptor difference at that level.
        for (int l = 1; l < nLevels; l++) {
            const cv::Mat &a = cpuExtractor.mvImagePyramid[l];
            const cv::Mat &b = gpuExtractor.mvImagePyramid[l];
            if (a.empty() || b.empty() || a.size() != b.size()) {
                pyrSizeMismatch[l]++;
                continue;
            }
            cv::Mat diff;
            cv::absdiff(a, b, diff);
            double maxDiff;
            cv::minMaxLoc(diff, nullptr, &maxDiff);
            pyrMaxDiff[l] = max(pyrMaxDiff[l], maxDiff);
            pyrDiffPct[l] += 100.0 * cv::countNonZero(diff) / diff.total();
            pyrFrames[l]++;
        }

        if (haveVoc) {
            DBoW2::BowVector bc, bg1, bg2;
            DBoW2::FeatureVector fv;
            voc.transform(Converter::toDescriptorVector(cpu.descriptors), bc, fv, 4);
            voc.transform(Converter::toDescriptorVector(gpu1.descriptors), bg1, fv, 4);
            voc.transform(Converter::toDescriptorVector(gpu2.descriptors), bg2, fv, 4);
            cpuBow.push_back(bc);
            gpuBow.push_back(bg1);
            selfScores.push_back(voc.score(bg1, bg2));
            crossScores.push_back(voc.score(bc, bg1));
        }
    }

    printf("\n=== summary over %zu images ===\n", cpuCounts.size());
    printf("keypoints           CPU %.0f   GPU %.0f\n", mean(cpuCounts),
           mean(gpuCounts));
    printf("CPU keypoints found by GPU   %.1f%%\n", mean(commonFrac));
    printf("descriptor mean Hamming      %.2f bits (0 = identical, 256 = max)\n",
           mean(meanHam));
    printf("descriptors bit-identical    %.1f%%\n", mean(identFrac));
    printf("orientation mean |dtheta|    %.2f deg\n", mean(angleDiff));
    printf("extraction time              CPU %.2f ms   GPU %.2f ms\n",
           mean(cpuTimes), mean(gpuTimes));
    printf("GPU repeatability (same img) %.1f%%  <- 100%% unless the GPU path races\n",
           mean(gpuDeterm));

    printf("\nper level (totals over all images):\n");
    printf("%5s %8s %8s %9s %9s %9s %10s %9s\n", "level", "nCPU", "nGPU",
           "common%", "meanHam", "gpuDet%", "pyrDiff%", "pyrMax");
    for (int l = 0; l < nLevels; l++) {
        const Agreement &c = perLevelCpu[l];
        const Agreement &d = perLevelSelf[l];
        printf("%5d %8zu %8zu %9.1f %9.2f %9.1f %10.2f %9.0f%s\n", l, c.aCount,
               c.bCount, c.aCount ? 100.0 * c.common / c.aCount : 0.0,
               c.common ? static_cast<double>(c.distSum) / c.common : 0.0,
               d.aCount ? 100.0 * d.common / d.aCount : 0.0,
               pyrFrames[l] ? pyrDiffPct[l] / pyrFrames[l] : 0.0, pyrMaxDiff[l],
               pyrSizeMismatch[l] ? "  SIZE MISMATCH" : "");
    }

    if (haveVoc) {
        // Place recognition compares BoW vectors of frames seen at different
        // times. Consecutive frames are a stand-in for a revisit: whatever score
        // the CPU path gets between them is the budget loop detection has.
        vector<double> cpuNeighbour, gpuNeighbour;
        for (size_t i = 0; i + 1 < cpuBow.size(); i++) {
            cpuNeighbour.push_back(voc.score(cpuBow[i], cpuBow[i + 1]));
            gpuNeighbour.push_back(voc.score(gpuBow[i], gpuBow[i + 1]));
        }
        printf("\nDBoW2 score, same image extracted twice on GPU   %.4f  (1.0 = deterministic)\n",
               mean(selfScores));
        printf("DBoW2 score, same image CPU vs GPU               %.4f  (1.0 = same words)\n",
               mean(crossScores));
        printf("DBoW2 score between neighbouring frames, CPU     %.4f\n",
               mean(cpuNeighbour));
        printf("DBoW2 score between neighbouring frames, GPU     %.4f\n",
               mean(gpuNeighbour));
    }

    fflush(stdout);
    // Both extractors are intentionally not destroyed; see above.
    _exit(0);
}
