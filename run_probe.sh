#!/bin/bash
# Fast diagnostic runner: invokes the SLAM binary directly and skips the trajectory
# evaluation and chart rendering that run_script.sh does afterwards. The rendering step
# alone takes ~9 minutes on a 16k-frame sequence, which dominates an unpaced run and
# makes the debug loop slower than the thing being debugged.
#
#   ./run_probe.sh <dataset> <FT> <TM> <FL> <kFT> <kTM> <kFL> <label>
#
# Environment passes straight through, so combine with:
#   NITRO_NO_PACING=1              run flat out instead of at sequence timestamps
#   NITRO_EXIT_ON_TRACKING_LOSS=1  stop the moment a new map is created (exit 42)
#   NITRO_VALIDATE_CULL=1          cross-check the culling histogram against the walk
#
# Writes Results/probe/<label>/{ostream,stderr}.txt
set -u
ds=$1; ft=$2; tm=$3; fl=$4; kft=$5; ktm=$6; kfl=$7; label=$8
data=$HOME/SLAM/Datasets/tumvi/dataset-${ds}_512_16/mav0
out=Results/probe/${label}
mkdir -p "$out"; out_abs=$(cd "$out" && pwd)

case "$ds" in *outdoors*) cfg=Stereo-Inertial/TUM-VI_far.yaml ;; *) cfg=Stereo-Inertial/TUM-VI.yaml ;; esac

cd Examples
start=$SECONDS
CUDA_MODULE_LOADING=LAZY ./Stereo-Inertial/stereo_inertial_tum_vi \
    ../Vocabulary/ORBvoc.txt "$cfg" \
    "$data/cam0/data" "$data/cam1/data" \
    "Stereo-Inertial/TUM_TimeStamps/dataset-${ds}_512.txt" \
    "Stereo-Inertial/TUM_IMU/dataset-${ds}_512.txt" \
    "dataset-${ds}_stereoi" "$out_abs" \
    "$ft" "$tm" "$fl" "$kft" "$ktm" "$kfl" \
    > "$out_abs/ostream.txt" 2> "$out_abs/stderr.txt"
rc=$?

# NITRO_PROBE_EVAL=1 adds the ATE evaluation (a few seconds) but still skips the chart
# rendering, which is the slow part of the normal pipeline.
if [ "${NITRO_PROBE_EVAL:-0}" = "1" ] && [ "$rc" = "0" ]; then
    python3 -W ignore ../evaluation/evaluate3.py \
        "$data/mocap0/data.csv" "f_dataset-${ds}_stereoi.txt" --verbose \
        >> "$out_abs/ostream.txt" 2>> "$out_abs/stderr.txt"
fi
cd ..
rm -f Examples/f_dataset-${ds}_stereoi.* Examples/kf_dataset-${ds}_stereoi.* 2>/dev/null

echo "[probe] $label  rc=$rc  $((SECONDS-start))s  maps_created=$(grep -c 'Creation of new map with id' "$out_abs/ostream.txt")  failTLM=$(grep -c 'Fail to track local map' "$out_abs/ostream.txt")  KFs=$(grep -oE 'has [0-9]+ KFs' "$out_abs/ostream.txt" | head -1 | grep -oE '[0-9]+')  pairs=$(grep -m1 compared_pose_pairs "$out_abs/ostream.txt" | awk '{print $2}')  rmse=$(grep -m1 'absolute_translational_error.rmse' "$out_abs/ostream.txt" | awk '{print $2}')"
[ "$rc" = "42" ] && echo "[probe] -> TRACKING LOST (early exit)"
exit $rc
