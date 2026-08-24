#!/bin/bash
#
# TurboMap bisect on outdoors3 — follow-up to run_ablation_tracking.sh.
#
# That sweep showed TurboMap alone reproduces the tracking loss (2 maps, 152 failed
# trackLocalMap, 1208 GT pairs) while FastTrack and FastLoop alone are clean. This
# narrows it to one of TurboMap's four kernels.
#
# Leave-one-out rather than one-at-a-time. The failure comes from map points
# accumulating subtle position/descriptor errors over ~11k frames, so a single kernel
# running alone may not degrade the map enough to reproduce it, while disabling one
# kernel out of four should show plainly which one's removal restores clean tracking.
# tm-all is the positive control: if it stops failing, the effect is not reproducible
# enough to bisect and the passes need deepening instead.
#
# kernel_status_TM bit order (see stereo_inertial_tum_vi.cc argv[argc-2]):
#   [0] searchForTriangulation  [1] fuse  [2] keyframeCulling  [3] localBA
#
# Usage: ./run_ablation_turbomap.sh [num_passes] [start_pass] [label_filter]

set -u

NUM_PASSES=${1:-3}
START_PASS=${2:-0}
FILTER=${3:-}
DATASET=outdoors3
TIMEOUT=45m
LOG=ablation_turbomap.log

# label       FT TM FL  TM-status
CONFIGS=(
    # Decisive config: TurboMap active, every compute kernel off. The GPU keyframe
    # mirror (CudaKeyFrameAllocator::create + addFeatureVector, LocalMapping.cc:455) is
    # gated on MappingKernelController::is_active, not on these bits, so it still runs
    # here. If this fails, the cause is the shared mirroring infrastructure rather than
    # any compute kernel -- which is also why the leave-one-out sweep came out flat.
    "tm-none     0 1 0 0000"
    "tm-all      0 1 0 1111"
    "tm-no-tri   0 1 0 0111"
    "tm-no-fuse  0 1 0 1011"
    "tm-no-cull  0 1 0 1101"
    "tm-no-lba   0 1 0 1110"

    # Single-kernel sweep. Leave-one-out showed no kernel is individually necessary;
    # this shows whether any is individually sufficient. All clean => the degradation is
    # cumulative across the four. One failing alone => that one dominates.
    "tm-tri      0 1 0 1000"
    "tm-fuse     0 1 0 0100"
    "tm-cull     0 1 0 0010"
    "tm-lba      0 1 0 0001"

    # Prediction test. The single-kernel sweep found fuse (0100) and keyframe culling
    # (0010) each sufficient on their own, while triangulation and LBA are clean. Every
    # leave-one-out config still contained at least one of fuse/cull, which is exactly
    # why none of them recovered. If that reading is right, disabling both -- leaving
    # triangulation + LBA on GPU -- should track cleanly.
    "tm-tri-lba  0 1 0 1001"
)

stats_dir() {
    local ft=$1 tm=$2 fl=$3 ks=$4 version=$5 iteration=$6
    local system="" kernels=""
    [ "$ft" -eq 1 ] && system+="FastTrack"
    [ "$tm" -eq 1 ] && system+="${system:+&}TurboMap"
    [ "$fl" -eq 1 ] && system+="${system:+&}FastLoop"
    [ "$ft" -eq 1 ] && [ "$tm" -eq 1 ] && [ "$fl" -eq 1 ] && system="Nitro-SLAM"
    [ -z "$system" ] && system="ORB-SLAM3"
    [ "$ks" != "-" ] && kernels="/${ks//,/-}"
    echo "Results/${system}${kernels}/${version}/${DATASET}/${iteration}"
}

echo "=== turbomap bisect start $(date -Is): passes ${START_PASS}..$((NUM_PASSES-1)), ${#CONFIGS[@]} configs ===" >> "$LOG"
for pass in $(seq "$START_PASS" $((NUM_PASSES - 1))); do
    for cfg in "${CONFIGS[@]}"; do
        read -r label ft tm fl ks <<< "$cfg"
        [ -n "$FILTER" ] && [ "$label" != "$FILTER" ] && continue
        version="$label"
        dir=$(stats_dir "$ft" "$tm" "$fl" "$ks" "$version" "$pass")
        if [ -f "$dir/ostream.txt" ]; then
            if grep -q "End of saving trajectory" "$dir/ostream.txt"; then
                echo "[$(date -Is)] pass $pass  $label  SKIP (already complete)" >> "$LOG"; continue
            fi
            echo "[$(date -Is)] pass $pass  $label  discarding partial $dir" >> "$LOG"; rm -rf "$dir"
        fi
        echo "[$(date -Is)] pass $pass  $label  (TM-status=$ks)" >> "$LOG"
        start=$SECONDS
        timeout --signal=KILL "$TIMEOUT" ./run_script.sh "$DATASET" "$ft" "$tm" "$fl" 1 "$version" "$pass" "$ks"
        rc=$?
        [ "$rc" -ne 0 ] && pkill -KILL -f stereo_inertial_tum_vi 2>/dev/null
        echo "[$(date -Is)] pass $pass  $label  rc=$rc  $((SECONDS-start))s" >> "$LOG"
    done
done
echo "=== turbomap bisect done $(date -Is) ===" >> "$LOG"
