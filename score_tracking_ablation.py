#!/usr/bin/env python3
"""Score the outdoors3 tracking-loss ablation.

A run "loses tracking" if the atlas ends with more than one map, or if any frame
reports a failed trackLocalMap. Both are reported because they are not the same
thing: a run can recover from a burst of failures without resetting the map.

Usage: ./score_tracking_ablation.py [results_dir]
"""
import os, re, sys, glob, statistics

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Results"
# Discover config labels from the tree rather than hardcoding them: the <version>
# component is the ablation label, and it sits directly above <dataset>/<pass>.
ORDER = ["baseline","ft-only","tm-only","fl-only","ft-tm","ft-fl","tm-fl","full","full-nopacing",
         "tm-none","tm-all","tm-no-tri","tm-no-fuse","tm-no-cull","tm-no-lba",
         "tm-tri","tm-fuse","tm-cull","tm-lba"]
def discover(root):
    seen = set()
    for p in glob.glob(f"{root}/**/outdoors3/*/ostream.txt", recursive=True):
        seen.add(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))))
    known = [l for l in ORDER if l in seen]
    return known + sorted(seen - set(known))
LABELS = None

def load(d, rel):
    m = {}
    p = os.path.join(d, "Tracking", "data", rel + ".txt")
    if not os.path.isfile(p):
        return m
    for line in open(p):
        try:
            k, v = line.split(':'); k = int(k)
            if k < 10**7:            # skip the uninitialised-Frame::mnId rows
                m[k] = float(v)
        except ValueError:
            pass
    return m

def score(d):
    o = os.path.join(d, "ostream.txt")
    if not os.path.isfile(o):
        return None
    t = open(o, errors="replace").read()
    if "End of saving trajectory" not in t:
        # A run still in flight also lacks the trailer; recent mtime tells them apart.
        import time
        fresh = (time.time() - os.path.getmtime(o)) < 120
        return {"status": "RUNNING" if fresh else "CRASHED"}
    r = {"status": "ok"}
    m = re.search(r"There are (\d+) maps in the atlas", t); r["maps"] = int(m.group(1)) if m else None
    r["kfs"] = [int(x) for x in re.findall(r"has (\d+) KFs", t)]
    r["failTLM"] = t.count("Fail to track local map!")
    m = re.search(r"absolute_translational_error\.rmse\s+([0-9.eE+-]+)", t)
    r["rmse"] = float(m.group(1)) if m else None
    m = re.search(r"compared_pose_pairs\s+(\d+)", t); r["pairs"] = int(m.group(1)) if m else None
    # frames that fell off the motion model
    tw = set(load(d, "trackWithMotionModel_time")); tlm = set(load(d, "TLM_poseOptimization_time"))
    # Frames 0..~17 legitimately have no motion model (initialisation), so they are
    # not a tracking loss; ignore anything before the first 50 frames.
    fb = sorted(k for k in tlm if k not in tw and k >= 50)
    r["offmm"] = len(fb); r["first_fail"] = fb[0] if fb else None
    # the new diagnostics, sampled in the 300 frames before the first failure
    q = load(d, "localmapper_queue")
    r["queue_max"] = max(q.values()) if q else None
    r["queue_mean"] = statistics.mean(q.values()) if q else None
    if fb:
        f0 = fb[0]
        for name, rel in (("to_match","num_slp_to_match"),("matches","num_slp_matches"),("inliers","num_matches_inliers")):
            s = load(d, rel)
            pre = [s[k] for k in s if f0-300 <= k < f0]
            at  = [s[k] for k in s if f0 <= k <= f0+50]
            r[name+"_pre"] = statistics.mean(pre) if pre else None
            r[name+"_at"]  = statistics.mean(at) if at else None
        qq = [q[k] for k in q if f0-300 <= k < f0]
        r["queue_pre_fail"] = statistics.mean(qq) if qq else None
    return r

LABELS = discover(ROOT)
print(f"{'config':16}{'pass':5}{'maps':5}{'failTLM':9}{'offMM':7}{'1stFail':9}{'pairs':7}{'rmse':10}{'qmax':6}{'q@fail':8}{'match pre>at':14}{'inlier pre>at'}")
print("-"*130)
for lbl in LABELS:
    for d in sorted(glob.glob(f"{ROOT}/**/{lbl}/outdoors3/*", recursive=True)):
        p = os.path.basename(d)
        s = score(d)
        if not s: continue
        if s["status"] != "ok":
            print(f"{lbl:16}{p:5}{s['status']}"); continue
        def f(x, w=0, dec=0):
            return "-" if x is None else (f"{x:.{dec}f}" if dec else f"{x:.0f}")
        rmse_s = "-" if s["rmse"] is None else f"{s['rmse']:.3f}"
        mm = f"{f(s.get('matches_pre'))}>{f(s.get('matches_at'))}"
        ii = f"{f(s.get('inliers_pre'))}>{f(s.get('inliers_at'))}"
        print(f"{lbl:16}{p:5}{s['maps'] or 0:<5}{s['failTLM']:<9}{s['offmm']:<7}"
              f"{str(s['first_fail'] or '-'):9}{str(s['pairs'] or '-'):7}"
              f"{rmse_s:10}"
              f"{f(s['queue_max']):6}{f(s.get('queue_pre_fail'),dec=1):8}{mm:14}{ii}")
