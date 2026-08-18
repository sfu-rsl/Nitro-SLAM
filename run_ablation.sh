#!/bin/bash
#
# Loop-closure ablation on outdoors5.
#
# Runs one GPU kernel at a time (with the other two optimizations off) and
# records whether the sequence still detects and closes its loop. Passes are
# interleaved -- pass 0 covers every config, then pass 1, then pass 2 -- so the
# table is always at equal depth across configs and the job can be stopped at
# any pass boundary.
#
# Usage: ./run_ablation.sh [num_passes] [start_pass] [label_filter]
#
# label_filter runs only configs whose label matches it, e.g.
#   ./run_ablation.sh 3 0 maximal

set -u

NUM_PASSES=${1:-3}
START_PASS=${2:-0}
FILTER=${3:-}
DATASET=outdoors5
VERSION=abl
TIMEOUT=45m
LOG=ablation_driver.log

# label            FT TM FL kernel_status
CONFIGS=(
    "baseline          0 0 0 -"

    "FT-none           1 0 0 00001"
    "TM-none           0 1 0 0000"
    "FL-none           0 0 1 000000"

    "FT-orb            1 0 0 10001"
    "FT-stereo         1 0 0 01001"
    "FT-searchlocal    1 0 0 00101"
    "FT-poseest        1 0 0 00011"
    "FT-noposeopt      1 0 0 00000"

    "TM-tri            0 1 0 1000"
    "TM-fuse           0 1 0 0100"
    "TM-cull           0 1 0 0010"
    "TM-lba            0 1 0 0001"

    "FL-mergedsbp      0 0 1 100000"
    "FL-merged3sbp     0 0 1 010000"
    "FL-searchandfuse  0 0 1 001000"
    "FL-singlesbp      0 0 1 000100"
    "FL-pgo            0 0 1 000010"
    "FL-gba            0 0 1 000001"

    # Pose estimation with the rest of the GPU tracking pipeline around it
    # (stereoMatch + searchLocalPoints on, orbExtraction off). Tests whether the
    # FT-poseest crash is an artefact of running that kernel in isolation.
    "FT-pose-in-pipeline 1 0 0 01111"

    # Every kernel that survived the single-kernel sweep, enabled together.
    # Comma-separated statuses map to FT,TM,FL in that order. Drops only
    # FT orbExtraction (0/3 detections) and FT poseEstimation (3/3 crashes).
    "maximal           1 1 1 01101,1111,111111"

    # The recommended set. poseEstimation only crashes in isolation, so it goes
    # back in: 13 of 14 kernels, everything but GPU ORB extraction. 3/3 closed.
    "maximal13         1 1 1 01111,1111,111111"
)

# Where run_script.sh will put this config's output. Mirrors its naming so we can
# tell a finished run from an interrupted one before spending 16 minutes on it.
stats_dir() {
    local ft=$1 tm=$2 fl=$3 ks=$4 version=$5 iteration=$6
    local system="" kernels=""
    [ "$ft" -eq 1 ] && system+="FastTrack"
    [ "$tm" -eq 1 ] && system+="${system:+&}TurboMap"
    [ "$fl" -eq 1 ] && system+="${system:+&}FastLoop"
    # All three on is the full system, which run_script.sh names Nitro-SLAM.
    [ "$ft" -eq 1 ] && [ "$tm" -eq 1 ] && [ "$fl" -eq 1 ] && system="Nitro-SLAM"
    [ -z "$system" ] && system="ORB-SLAM3"
    # run_script.sh joins per-subsystem statuses with '-' in the directory name.
    [ "$ks" != "-" ] && kernels="/${ks//,/-}"
    echo "Results/${system}${kernels}/${version}/${DATASET}/${iteration}"
}

echo "=== ablation start $(date -Is): passes ${START_PASS}..$((NUM_PASSES - 1)), ${#CONFIGS[@]} configs ===" >> "$LOG"

for pass in $(seq "$START_PASS" $((NUM_PASSES - 1))); do
    for cfg in "${CONFIGS[@]}"; do
        read -r label ft tm fl ks <<< "$cfg"
        version="$VERSION"

        if [ -n "$FILTER" ] && [ "$label" != "$FILTER" ]; then
            continue
        fi

        # Resume: keep finished runs, discard interrupted ones. A run that never
        # wrote its trajectory left only a truncated log, which would otherwise
        # score as a genuine crash.
        dir=$(stats_dir "$ft" "$tm" "$fl" "$ks" "$version" "$pass")
        if [ -f "$dir/ostream.txt" ]; then
            if grep -q "End of saving trajectory" "$dir/ostream.txt"; then
                echo "[$(date -Is)] pass $pass  $label  SKIP (already complete)" >> "$LOG"
                continue
            fi
            echo "[$(date -Is)] pass $pass  $label  discarding partial $dir" >> "$LOG"
            rm -rf "$dir"
        fi

        # The pass number is run_script.sh's iteration argument; kernel statuses
        # follow it, one per enabled subsystem, in FT,TM,FL order.
        args=("$DATASET" "$ft" "$tm" "$fl" 1 "$version" "$pass")
        if [ "$ks" != "-" ]; then
            IFS=',' read -r -a statuses <<< "$ks"
            args+=("${statuses[@]}")
        fi

        echo "[$(date -Is)] pass $pass  $label  (FT=$ft TM=$tm FL=$fl ks=$ks)" >> "$LOG"
        start=$SECONDS
        timeout --signal=KILL "$TIMEOUT" ./run_script.sh "${args[@]}"
        rc=$?
        # timeout only kills run_script.sh; the SLAM binary is a grandchild and
        # would otherwise hold the GPU for the rest of the queue. Nothing else in
        # this call chain carries the binary's name on its command line.
        if [ "$rc" -ne 0 ]; then
            pkill -KILL -f stereo_inertial_tum_vi 2>/dev/null
        fi
        echo "[$(date -Is)] pass $pass  $label  rc=$rc  $((SECONDS - start))s" >> "$LOG"
    done
done

echo "=== ablation done $(date -Is) ===" >> "$LOG"
