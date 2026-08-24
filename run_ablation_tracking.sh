#!/bin/bash
#
# Tracking-loss ablation on outdoors3.
#
# Question: ORB-SLAM3 tracks outdoors3 cleanly (5/5 runs, single map, zero
# "Fail to track local map"). Full Nitro-SLAM loses tracking mid-sequence and
# resets the map in ~4/5 runs. Which subsystem causes it -- or is it only the
# combination, which would point at a timing/interaction effect rather than any
# single kernel?
#
# Subsystems are tested alone and in pairs so that "no single one breaks it but
# the combination does" is distinguishable from "one kernel breaks it".
#
# The last config re-runs the full system with playback pacing disabled
# (NITRO_NO_PACING). Paced playback sleeps to the sequence timestamps, so faster
# tracking normally just idles more; removing pacing lets tracking run flat out
# and maximally starves local mapping of relative headroom. If the failure is a
# tracking-outruns-mapping effect, this config should make it markedly worse.
#
# Each config uses its own <version> directory, so paced and unpaced full-system
# runs cannot collide. Passes are interleaved -- pass 0 covers every config, then
# pass 1 -- so the table is always at equal depth and the job can be stopped at
# any pass boundary.
#
# Usage: ./run_ablation_tracking.sh [num_passes] [start_pass] [label_filter]

set -u

NUM_PASSES=${1:-3}
START_PASS=${2:-0}
FILTER=${3:-}
DATASET=outdoors3
TIMEOUT=45m
LOG=ablation_tracking.log

# label         FT TM FL kernel_status(FT,TM,FL)   env
CONFIGS=(
    "baseline       0 0 0 -                      -"
    "ft-only        1 0 0 11111                  -"
    "tm-only        0 1 0 1111                   -"
    "fl-only        0 0 1 001111                 -"
    "ft-tm          1 1 0 11111,1111             -"
    "ft-fl          1 0 1 11111,001111           -"
    "tm-fl          0 1 1 1111,001111            -"
    "full           1 1 1 11111,1111,001111      -"
    "full-nopacing  1 1 1 11111,1111,001111      NITRO_NO_PACING=1"
)

# Mirror run_script.sh's naming so a finished run is recognisable before spending
# 16 minutes redoing it.
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

echo "=== tracking ablation start $(date -Is): passes ${START_PASS}..$((NUM_PASSES-1)), ${#CONFIGS[@]} configs ===" >> "$LOG"

for pass in $(seq "$START_PASS" $((NUM_PASSES - 1))); do
    for cfg in "${CONFIGS[@]}"; do
        read -r label ft tm fl ks env <<< "$cfg"
        [ -n "$FILTER" ] && [ "$label" != "$FILTER" ] && continue

        # version = label, so configs sharing a kernel status (paced vs unpaced)
        # still land in separate directories.
        version="$label"
        dir=$(stats_dir "$ft" "$tm" "$fl" "$ks" "$version" "$pass")
        if [ -f "$dir/ostream.txt" ]; then
            if grep -q "End of saving trajectory" "$dir/ostream.txt"; then
                echo "[$(date -Is)] pass $pass  $label  SKIP (already complete)" >> "$LOG"
                continue
            fi
            echo "[$(date -Is)] pass $pass  $label  discarding partial $dir" >> "$LOG"
            rm -rf "$dir"
        fi

        args=("$DATASET" "$ft" "$tm" "$fl" 1 "$version" "$pass")
        if [ "$ks" != "-" ]; then
            IFS=',' read -r -a statuses <<< "$ks"
            args+=("${statuses[@]}")
        fi

        echo "[$(date -Is)] pass $pass  $label  (FT=$ft TM=$tm FL=$fl ks=$ks env=$env)" >> "$LOG"
        start=$SECONDS
        if [ "$env" != "-" ]; then
            env "$env" timeout --signal=KILL "$TIMEOUT" ./run_script.sh "${args[@]}"
        else
            timeout --signal=KILL "$TIMEOUT" ./run_script.sh "${args[@]}"
        fi
        rc=$?
        # timeout only kills run_script.sh; the SLAM binary is a grandchild and would
        # otherwise hold the GPU for the rest of the queue.
        [ "$rc" -ne 0 ] && pkill -KILL -f stereo_inertial_tum_vi 2>/dev/null
        echo "[$(date -Is)] pass $pass  $label  rc=$rc  $((SECONDS-start))s" >> "$LOG"
    done
done
echo "=== tracking ablation done $(date -Is) ===" >> "$LOG"
