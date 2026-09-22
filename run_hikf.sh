#!/bin/bash
# High-keyframe-insertion stress test (TurboMap paper, Table II).
#   ./run_hikf.sh <label> [num_iterations] [seq_filter]
#
# Reproduces the TurboMap stress setup on every arm: frames are fed unpaced
# (NITRO_NO_PACING) and the c1a frame gap between keyframes is lowered
# (NITRO_KF_MAX_FRAMES), so Tracking inserts keyframes faster than Local Mapping can
# absorb them. Local Mapping then skips LBA + keyframe culling whenever another
# keyframe is already queued; LocalMapping.cc prints the count at shutdown as
# "[LBA skip] ... skipped_queue=N", which scripts/hikf_table.py tabulates with ATE.
#
# label names the machine/setting (e.g. desktop, jetson). It becomes part of the
# version directory, so runs from different machines never collide and their trees
# can be merged by copying:
#   Results-euroc-hikf/<system>[/<kernels>]/hikf-<label>-mf<N>/<seq>/<iter>
#   Results-tumvi-hikf/...
#
# Environment:
#   KF_MAX_FRAMES  c1a frame gap (default 4; "off" keeps the fps default, i.e.
#                  unpaced only -- the version dir then ends in -mfoff). 4 was
#                  calibrated on MH01 against the paper's ORB-SLAM3 and TurboMap
#                  numbers; 0-2 trip the IMU-init "not enough motion" reset and
#                  tracking loss, which punishes the faster mapper instead.
#   ARMS           "|"-separated "FT TM FL" triples (default: ORB-SLAM3, FastTrack,
#                  TurboMap, FastTrack+TurboMap, TurboMap+FastLoop, Nitro-SLAM)
#   RUN_TIMEOUT    per-run watchdog in seconds (default 900; raise on the Jetson)
#   FORCE=1        redo runs already marked done
#
# Passes are interleaved (pass 0 over every sequence and arm, then pass 1, ...), and
# completed runs are skipped, so resuming is just re-issuing the same command.

set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <label> [num_iterations] [seq_filter]"; exit 1
fi
label=$1
num_itr=${2:-5}
filter=${3:-}

datasets=("MH01" "V101" "room1" "corridor1")
if [ -n "$filter" ]; then
    keep=()
    for d in "${datasets[@]}"; do [[ "$d" =~ $filter ]] && keep+=("$d"); done
    [ ${#keep[@]} -eq 0 ] && { echo "filter '$filter' matched no sequence"; exit 1; }
    datasets=("${keep[@]}")
fi

KF_MAX_FRAMES=${KF_MAX_FRAMES:-4}
export NITRO_NO_PACING=1
if [ "$KF_MAX_FRAMES" = "off" ]; then
    unset NITRO_KF_MAX_FRAMES
else
    export NITRO_KF_MAX_FRAMES=$KF_MAX_FRAMES
fi
version="hikf-${label}-mf${KF_MAX_FRAMES}"

ARMS=${ARMS:-"0 0 0|1 0 0|0 1 0|1 1 0|0 1 1|1 1 1"}
IFS='|' read -r -a arm_list <<< "$ARMS"
RUN_TIMEOUT=${RUN_TIMEOUT:-900}

LOG=hikf_${label}.log
exec > >(tee -a "$LOG") 2>&1

# Mirror run_script.sh's default kernel statuses so the stats path can be predicted.
kFT=$(grep -E "^kernel_status_FT='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
kTM=$(grep -E "^kernel_status_TM='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
kFL=$(grep -E "^kernel_status_FL='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")

# Same naming as run_script.sh.
arm_dir() {
    local ft=$1 tm=$2 fl=$3 name="" kdir=""
    [ "$ft" -eq 1 ] && { name+="FastTrack"; kdir+="$kFT"; }
    [ "$tm" -eq 1 ] && { name+="${name:+&}TurboMap"; kdir+="${kdir:+-}$kTM"; }
    [ "$fl" -eq 1 ] && { name+="${name:+&}FastLoop"; kdir+="${kdir:+-}$kFL"; }
    [ "$ft" -eq 1 ] && [ "$tm" -eq 1 ] && [ "$fl" -eq 1 ] && name="Nitro-SLAM"
    [ -z "$name" ] && name="ORB-SLAM3"
    echo "${name}${kdir:+/$kdir}"
}

results_root() {
    case "$1" in
        MH*|V1*|V2*) echo Results-euroc-hikf ;;
        *)           echo Results-tumvi-hikf ;;
    esac
}

cleanup() { pkill -KILL -x stereo_inertial 2>/dev/null; sleep 2; }
trap 'echo "[hikf] interrupted"; cleanup; exit 130' INT TERM

echo "[hikf] $(date +%F\ %T) label=$label version=$version iters=$num_itr seqs=${datasets[*]}"
echo "[hikf] NITRO_NO_PACING=1 NITRO_KF_MAX_FRAMES=${NITRO_KF_MAX_FRAMES:-<unset>} arms: $ARMS"

for i in $(seq 0 $((num_itr - 1))); do
    echo "[hikf] ===== pass $i / $((num_itr - 1)) ====="
    for dataset in "${datasets[@]}"; do
        export RESULTS_ROOT=$(results_root "$dataset")   # read by run_script.sh
        for cfg in "${arm_list[@]}"; do
            set -- $cfg
            statsDir="${RESULTS_ROOT}/$(arm_dir "$1" "$2" "$3")/${version}/${dataset}/${i}"
            if [ -f "${statsDir}/.batch_done" ] && [ "${FORCE:-0}" -eq 0 ]; then
                echo "[hikf] skip (done) iter=$i dataset=$dataset cfg='$cfg'"
                continue
            fi
            echo "[hikf] $(date +%T) iter=$i dataset=$dataset FT=$1 TM=$2 FL=$3"
            start=$SECONDS
            timeout "$RUN_TIMEOUT" ./run_script.sh "$dataset" "$1" "$2" "$3" 1 "$version" "$i"
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "[hikf] WARNING rc=$rc for $dataset iter=$i cfg='$cfg'"
            else
                touch "${statsDir}/.batch_done"
                grep -h "^\[LBA skip\]\|absolute_translational_error.rmse" "${statsDir}/ostream.txt" \
                    | sed 's/^/[hikf]   /'
            fi
            echo "[hikf]   $((SECONDS - start))s"
            cleanup
        done
    done
done
echo "[hikf] done $(date +%F\ %T) label=$label"
