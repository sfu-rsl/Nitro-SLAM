#include "PoseOptShadow.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>

#include "Frame.h"
#include "MapPoint.h"
#include "CameraModels/GeometricCamera.h"
#include "G2oTypes.h"

namespace ORB_SLAM3 {
namespace PoseOptShadow {
namespace {

std::mutex g_mutex;
std::ofstream g_out;
bool g_opened = false;

struct Row {
    int variant;
    double dpos, drot, dvel, dbias;
    int cpu_in, gpu_in, ndiff, nfac;
    // [kept-by-gpu][category] and [kept-by-cpu][category]; categories below.
    int keep[2][4];
};
// mono far, mono close, stereo, right camera
const char *kCat[4] = {"mono_far", "mono_close", "stereo", "right_cam"};
std::vector<Row> g_rows;
long g_chi2_cpu_lower[3] = {0, 0, 0};
long g_chi2_gpu_lower[3] = {0, 0, 0};
std::vector<double> g_chi2_ratio[3];

const char *env(const char *name) { return std::getenv(name); }

double RotDeg(const Sophus::SE3f &a, const Sophus::SE3f &b) {
    const Eigen::Matrix3d Ra = a.rotationMatrix().cast<double>();
    const Eigen::Matrix3d Rb = b.rotationMatrix().cast<double>();
    const double c = ((Ra.transpose() * Rb).trace() - 1.0) * 0.5;
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

double Pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = static_cast<size_t>(q * (v.size() - 1));
    return v[i];
}

const char *VariantName(int v) {
    return v == 0 ? "VISUAL" : (v == 1 ? "INERTIAL_F" : "INERTIAL_KF");
}

}  // namespace

bool Enabled() {
    static const bool on = env("NITRO_PO_SHADOW") != nullptr;
    return on;
}

bool KeepGpu() {
    static const bool keep_cpu =
        env("NITRO_PO_SHADOW_KEEP") && std::string(env("NITRO_PO_SHADOW_KEEP")) == "cpu";
    return !keep_cpu;
}

FrameState Capture(const Frame &F) {
    FrameState s;
    Frame &nc = const_cast<Frame &>(F);
    s.hasPose = nc.isSet();
    if (s.hasPose) s.Tcw = nc.GetPose();
    s.hasVelocity = nc.HasVelocity();
    if (s.hasVelocity) s.Vw = nc.GetVelocity();
    s.bias = nc.mImuBias;
    s.outlier = nc.mvbOutlier;
    s.cpi = nc.mpcpi;
    s.prev = nc.mpPrevFrame;
    if (s.prev && s.prev->mpcpi) {
        const ConstraintPoseImu &c = *s.prev->mpcpi;
        s.hasPrevPrior = true;
        s.pRwb = c.Rwb;
        s.ptwb = c.twb;
        s.pvwb = c.vwb;
        s.pbg = c.bg;
        s.pba = c.ba;
        s.pH = c.H;
    }
    return s;
}

double VisualChi2(const Frame &F, const Sophus::SE3f &Tcw,
                  const std::vector<bool> &mask) {
    Frame &nc = const_cast<Frame &>(F);
    const Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    const Eigen::Vector3f tcw = Tcw.translation();
    const bool rig = (nc.Nleft != -1);
    const Sophus::SE3f Trl = rig ? nc.GetRelativePoseTrl() : Sophus::SE3f();
    double total = 0.0;
    for (int i = 0; i < nc.N; i++) {
        MapPoint *mp = nc.mvpMapPoints[i];
        if (!mp) continue;
        if (i < static_cast<int>(mask.size()) && mask[i]) continue;   // outlier
        const Eigen::Vector3f Xc = Rcw * mp->GetWorldPos() + tcw;
        const bool right = rig && i >= nc.Nleft;
        const Eigen::Vector3f Xo = right ? (Trl * Xc).eval() : Xc;
        if (Xo.z() <= 0.f) continue;
        GeometricCamera *cam = right ? nc.mpCamera2 : nc.mpCamera;
        const cv::Point2f uv = cam->project(cv::Point3f(Xo.x(), Xo.y(), Xo.z()));
        const cv::KeyPoint &kp = right ? nc.mvKeysRight[i - nc.Nleft]
                                       : (rig ? nc.mvKeys[i] : nc.mvKeysUn[i]);
        const double du = uv.x - kp.pt.x, dv = uv.y - kp.pt.y;
        const Eigen::Matrix<double, 2, 1> o(kp.pt.x, kp.pt.y);
        const float unc2 = nc.mpCamera->uncertainty2(o);
        const double inv = nc.mvInvLevelSigma2[kp.octave] / unc2;
        total += (du * du + dv * dv) * inv;
    }
    return total;
}

void Restore(Frame &F, const FrameState &s) {
    // A solver that ran since the capture will have allocated a fresh prior; the
    // frame is going back to the one it held, so the new one is released rather
    // than leaked. DeleteConstraintPoseImu pairs with however it was allocated.
    if (F.mpcpi && F.mpcpi != s.cpi) {
        DeleteConstraintPoseImu(F.mpcpi);
    }
    F.mpcpi = s.cpi;
    if (s.hasPose) F.SetPose(s.Tcw);
    if (s.hasVelocity) F.SetVelocity(s.Vw);
    F.mImuBias = s.bias;
    F.mvbOutlier = s.outlier;

    // Reinstate the previous frame's prior if a solver consumed it, so the next
    // backend sees the same input this one did.
    if (s.hasPrevPrior && s.prev) {
        if (s.prev->mpcpi) DeleteConstraintPoseImu(s.prev->mpcpi);
        s.prev->mpcpi =
            NewConstraintPoseImu(s.pRwb, s.ptwb, s.pvwb, s.pbg, s.pba, s.pH);
    }
}

void Record(int variant, long frame_id, int n_factors, const FrameState &cpu,
            int cpu_inliers, const FrameState &gpu, int gpu_inliers, const Frame &F) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_opened) {
        const char *path = env("NITRO_PO_SHADOW_OUT");
        g_out.open(path ? path : "pose_opt_shadow.csv");
        g_out << "frame_id,variant,n_factors,dpos_m,drot_deg,dvel,dbias,"
                 "cpu_inliers,gpu_inliers,n_outlier_diff\n";
        g_opened = true;
    }

    const double dpos = (cpu.hasPose && gpu.hasPose)
                            ? (cpu.Tcw.translation() - gpu.Tcw.translation())
                                  .cast<double>()
                                  .norm()
                            : 0.0;
    const double drot =
        (cpu.hasPose && gpu.hasPose) ? RotDeg(cpu.Tcw, gpu.Tcw) : 0.0;
    const double dvel = (cpu.hasVelocity && gpu.hasVelocity)
                            ? (cpu.Vw - gpu.Vw).cast<double>().norm()
                            : 0.0;
    const Eigen::Vector3d dba(cpu.bias.bax - gpu.bias.bax, cpu.bias.bay - gpu.bias.bay,
                              cpu.bias.baz - gpu.bias.baz);
    const Eigen::Vector3d dbg(cpu.bias.bwx - gpu.bias.bwx, cpu.bias.bwy - gpu.bias.bwy,
                              cpu.bias.bwz - gpu.bias.bwz);
    const double dbias = std::sqrt(dba.squaredNorm() + dbg.squaredNorm());

    int ndiff = 0;
    int keep[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    const int N = static_cast<int>(std::min(cpu.outlier.size(), gpu.outlier.size()));
    for (int i = 0; i < N; i++) {
        if (cpu.outlier[i] == gpu.outlier[i]) continue;
        ndiff++;
        int cat;
        if (F.Nleft != -1 && i >= F.Nleft) {
            cat = 3;                                   // right-camera observation
        } else if (F.Nleft == -1 && i < static_cast<int>(F.mvuRight.size()) &&
                   F.mvuRight[i] >= 0) {
            cat = 2;                                   // rectified stereo
        } else {
            MapPoint *mp = (i < static_cast<int>(F.mvpMapPoints.size()))
                               ? F.mvpMapPoints[i] : nullptr;
            cat = (mp && mp->mTrackDepth < 10.f) ? 1 : 0;
        }
        // cpu says outlier, gpu says inlier -> the gpu kept it.
        keep[cpu.outlier[i] ? 0 : 1][cat]++;
    }

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%ld,%s,%d,%.9g,%.9g,%.9g,%.9g,%d,%d,%d\n",
                  frame_id, VariantName(variant), n_factors, dpos, drot, dvel, dbias,
                  cpu_inliers, gpu_inliers, ndiff);
    g_out << buf;
    const double chi2_cpu = VisualChi2(F, cpu.Tcw, cpu.outlier);
    const double chi2_gpu = VisualChi2(F, gpu.Tcw, cpu.outlier);
    g_chi2_cpu_lower[variant] += (chi2_cpu < chi2_gpu) ? 1 : 0;
    g_chi2_gpu_lower[variant] += (chi2_gpu < chi2_cpu) ? 1 : 0;
    if (chi2_cpu > 0.0) g_chi2_ratio[variant].push_back(chi2_gpu / chi2_cpu);

    Row row{variant, dpos, drot, dvel, dbias, cpu_inliers, gpu_inliers, ndiff,
            n_factors, {{0, 0, 0, 0}, {0, 0, 0, 0}}};
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 4; b++) row.keep[a][b] = keep[a][b];
    g_rows.push_back(row);
}

void Finish() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_opened) return;
    g_out.flush();

    std::printf("\n[PO_SHADOW] %zu calls compared (g2o vs fused, identical inputs)\n",
                g_rows.size());
    std::printf("[PO_SHADOW] %-12s %7s  %10s %10s %10s   %9s %9s %9s   %8s\n",
                "variant", "calls", "dpos p50", "dpos p95", "dpos max", "drot p95",
                "dvel p95", "dbias p95", "outl!=");
    for (int v = 0; v < 3; v++) {
        std::vector<double> p, r, vel, b;
        long ndiff = 0;
        int inlier_diff = 0;
        for (const Row &row : g_rows) {
            if (row.variant != v) continue;
            p.push_back(row.dpos);
            r.push_back(row.drot);
            vel.push_back(row.dvel);
            b.push_back(row.dbias);
            ndiff += row.ndiff;
            inlier_diff += (row.cpu_in != row.gpu_in) ? 1 : 0;
        }
        if (p.empty()) continue;
        std::printf("[PO_SHADOW] %-12s %7zu  %10.3e %10.3e %10.3e   %9.3e %9.3e "
                    "%9.3e   %8ld (%d calls w/ inlier mismatch)\n",
                    VariantName(v), p.size(), Pct(p, 0.50), Pct(p, 0.95),
                    *std::max_element(p.begin(), p.end()), Pct(r, 0.95),
                    Pct(vel, 0.95), Pct(b, 0.95), ndiff, inlier_diff);
    }
    // Where the disagreements live. "gpu kept" means g2o called it an outlier and
    // the fused solver did not, which is the direction of the surplus.
    std::printf("[PO_SHADOW] which solver fit better (same factor set, same residual)\n");
    std::printf("[PO_SHADOW] %-12s %12s %12s %14s\n", "variant", "g2o lower",
                "fused lower", "chi2 ratio p50");
    for (int v = 0; v < 3; v++) {
        if (!g_chi2_cpu_lower[v] && !g_chi2_gpu_lower[v]) continue;
        std::printf("[PO_SHADOW] %-12s %12ld %12ld %14.6f\n", VariantName(v),
                    g_chi2_cpu_lower[v], g_chi2_gpu_lower[v],
                    Pct(g_chi2_ratio[v], 0.50));
    }
    std::printf("[PO_SHADOW] disagreements by observation kind\n");
    std::printf("[PO_SHADOW] %-12s %-12s %10s %10s\n", "variant", "category",
                "gpu kept", "cpu kept");
    for (int v = 0; v < 3; v++) {
        long tot = 0;
        for (const Row &r : g_rows)
            if (r.variant == v)
                for (int b = 0; b < 4; b++) tot += r.keep[0][b] + r.keep[1][b];
        if (!tot) continue;
        for (int b = 0; b < 4; b++) {
            long g = 0, c = 0;
            for (const Row &r : g_rows) {
                if (r.variant != v) continue;
                g += r.keep[0][b];
                c += r.keep[1][b];
            }
            if (g || c)
                std::printf("[PO_SHADOW] %-12s %-12s %10ld %10ld\n", VariantName(v),
                            kCat[b], g, c);
        }
    }
    g_out.close();
    g_opened = false;
}

namespace {
// Same shape as PoseOptTiming's reporter: the summary is printed when the process
// tears down, so no call site has to remember to ask for it.
struct Reporter {
    ~Reporter() { Finish(); }
};
Reporter g_reporter;
}  // namespace

}  // namespace PoseOptShadow
}  // namespace ORB_SLAM3
