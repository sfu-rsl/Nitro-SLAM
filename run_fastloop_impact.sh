#!/bin/bash
# FastLoop paper (arXiv:2603.17201v2), Tables II and III: the impact of a loop
# correction on the rest of the pipeline.
#   ./run_fastloop_impact.sh <label> [num_iterations] [seq_filter]
#
# Table II  -- Local Mapping stop time and keyframes refused by Tracking while it is
#              stopped, per sequence, FastLoop vs. ORB-SLAM3.
# Table III -- for the largest loop closure event: its duration, how many tracking
#              frames it overlapped, their latency, and the latency outside closure.
#              The paper reports Magistrale 1 only; this runs every Table II sequence.
#
# Both tables are per-occurrence measurements of a rare event, so unlike the batch
# timing runs these are PACED: frames are fed at the sequence's real frame rate
# (no NITRO_NO_PACING) and the keyframe frame gap is left at the camera default.
# Feeding unpaced would change how many frames land inside a stop window, which is
# precisely what Table II counts.
#
# Requires a build with REGISTER_STATS=ON: the numbers come from
#   <stats>/LoopClosing/data/{loopCorrection,lmStopRequested,lmStopped}_window.txt
#   <stats>/Tracking/data/{frame_window,kfRefusedLMStop,tracking_time}.txt
# which scripts/fastloop_impact.py turns into the two tables.
#
#   Results-euroc-fastloop/<system>[/<kernels>]/<version>/<seq>/<iter>
#   Results-tumvi-fastloop/...
#
# Environment:
#   ARMS         "|"-separated "FT TM FL" triples (default: ORB-SLAM3 and FastLoop).
#                "1 1 1" is Nitro-SLAM -- FastLoop plus accelerated tracking
#                (FastTrack) and mapping (TurboMap), which lowers Table III's normal
#                latency as well as the loop correction cost. Its results land under
#                Nitro-SLAM/<FT>-<TM>-<FL>/ rather than FastLoop/<FL>/.
#   FL_KERNELS   FastLoop bitmask (default 001110): loop fusion, the low-level Sim3
#                projection-search kernel and GPU PGO -- every parallelization on the
#                loop *correction* side, which is what both tables measure.
#                Digits 1-2 (the merged and triple projection searches, which the
#                harness default 001111 also disables) are off because enabling them
#                faults: V203 dies with an illegal memory access in the CUDA context
#                immediately after "*Loop detected". They act on region detection, not
#                on correction, so Table II is unaffected; the caveat for Table III is
#                that they can shift *when* a loop is detected, and so which event is
#                the largest.
#                Digit 6 (GPU global BA) is off because it also lifts ORB-SLAM3's
#                KeyFramesInMap()<200 guard on running GBA at all, which would put a
#                background optimizer in the FastLoop arm only -- a confound for the
#                normal-latency column on the ~3000-keyframe TUM-VI maps.
#   RUN_TIMEOUT  per-run watchdog in seconds (default 2400; outdoors5 is ~15 min of
#                paced playback before evaluation).
#   FORCE=1      redo runs already marked done
#
# Passes are interleaved and completed runs are skipped, so resuming is just
# re-issuing the same command.

set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <label> [num_iterations] [seq_filter]"; exit 1
fi
label=$1
num_itr=${2:-1}
filter=${3:-}

# The Table II sequences: three TUM-VI loops of increasing length plus the one EuRoC
# sequence, which gives the short-loop end of the range.
datasets=("magistrale1" "magistrale2" "outdoors5" "V203")
if [ -n "$filter" ]; then
    keep=()
    for d in "${datasets[@]}"; do [[ "$d" =~ $filter ]] && keep+=("$d"); done
    [ ${#keep[@]} -eq 0 ] && { echo "filter '$filter' matched no sequence"; exit 1; }
    datasets=("${keep[@]}")
fi

version="fl-${label}"
ARMS=${ARMS:-"0 0 0|0 0 1"}
IFS='|' read -r -a arm_list <<< "$ARMS"
RUN_TIMEOUT=${RUN_TIMEOUT:-2400}
FL_KERNELS=${FL_KERNELS:-001110}

LOG=fastloop_impact_${label}.log
exec > >(tee -a "$LOG") 2>&1

# run_script.sh's defaults for the modules we never enable here; FastLoop's bitmask is
# passed explicitly so the stats path stays predictable.
kFT=$(grep -E "^kernel_status_FT='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
kTM=$(grep -E "^kernel_status_TM='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")

# run_script.sh reads the kernel bitmasks from positional slots whose meaning depends
# on which modules are enabled: with FastTrack and TurboMap both on, FastLoop's bitmask
# is the THIRD optional argument, while with both off it is the first. Emitting only the
# enabled modules' masks, in module order, lands each one in the slot run_script.sh will
# read it from. Passing FastLoop's mask blindly as the first argument silently
# reprograms FastTrack instead, which is a corrupted run that still looks healthy.
kernel_args() {
    local ft=$1 tm=$2 fl=$3
    local args=()
    [ "$ft" -eq 1 ] && args+=("$kFT")
    [ "$tm" -eq 1 ] && args+=("$kTM")
    [ "$fl" -eq 1 ] && args+=("$FL_KERNELS")
    echo "${args[@]}"
}

arm_dir() {
    local ft=$1 tm=$2 fl=$3 name="" kdir=""
    [ "$ft" -eq 1 ] && { name+="FastTrack"; kdir+="$kFT"; }
    [ "$tm" -eq 1 ] && { name+="${name:+&}TurboMap"; kdir+="${kdir:+-}$kTM"; }
    [ "$fl" -eq 1 ] && { name+="${name:+&}FastLoop"; kdir+="${kdir:+-}$FL_KERNELS"; }
    [ "$ft" -eq 1 ] && [ "$tm" -eq 1 ] && [ "$fl" -eq 1 ] && name="Nitro-SLAM"
    [ -z "$name" ] && name="ORB-SLAM3"
    echo "${name}${kdir:+/$kdir}"
}

results_root() {
    case "$1" in
        MH*|V1*|V2*) echo Results-euroc-fastloop ;;
        *)           echo Results-tumvi-fastloop ;;
    esac
}

cleanup() {
    pkill -KILL -x stereo_inertial_euroc 2>/dev/null
    pkill -KILL -x stereo_inertial_tum_vi 2>/dev/null
    sleep 2
}
trap 'echo "[fl] interrupted"; cleanup; exit 130' INT TERM

echo "[fl] $(date +%F\ %T) label=$label version=$version iters=$num_itr seqs=${datasets[*]}"
echo "[fl] paced playback, FastLoop kernels=$FL_KERNELS arms: $ARMS"

for i in $(seq 0 $((num_itr - 1))); do
    echo "[fl] ===== pass $i / $((num_itr - 1)) ====="
    for dataset in "${datasets[@]}"; do
        export RESULTS_ROOT=$(results_root "$dataset")   # read by run_script.sh
        for cfg in "${arm_list[@]}"; do
            set -- $cfg
            statsDir="${RESULTS_ROOT}/$(arm_dir "$1" "$2" "$3")/${version}/${dataset}/${i}"
            if [ -f "${statsDir}/.batch_done" ] && [ "${FORCE:-0}" -eq 0 ]; then
                echo "[fl] skip (done) iter=$i dataset=$dataset cfg='$cfg'"
                continue
            fi
            echo "[fl] $(date +%T) iter=$i dataset=$dataset FT=$1 TM=$2 FL=$3"
            start=$SECONDS
            # shellcheck disable=SC2046  -- deliberate word splitting, 0-3 masks
            timeout "$RUN_TIMEOUT" ./run_script.sh "$dataset" "$1" "$2" "$3" 1 \
                "$version" "$i" $(kernel_args "$1" "$2" "$3")
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "[fl] WARNING rc=$rc for $dataset iter=$i cfg='$cfg'"
            else
                touch "${statsDir}/.batch_done"
                grep -h "closure(s) recorded\|absolute_translational_error.rmse" \
                    "${statsDir}/ostream.txt" | sed 's/^/[fl]   /'
            fi
            echo "[fl]   $((SECONDS - start))s"
            cleanup
        done
    done
done
echo "[fl] done $(date +%F\ %T) label=$label"
