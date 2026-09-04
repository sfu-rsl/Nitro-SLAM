#!/bin/bash
# Batch driver for the thread-breakdown timing study.
#   ./run_timing_batch.sh [group] [version] [num_iterations] [seq_filter]
#
# Runs one dataset group under both ORB-SLAM3 (0 0 0) and Nitro-SLAM (1 1 1), and
# samples GPU/CPU memory alongside each run into <statsDir>/memory.csv.
#
# Groups (each writes into its own results root, so a group can be run, analysed
# and archived without waiting on the others):
#   euroc     11 EuRoC sequences                    -> Results-euroc
#   tumvi     TUM-VI minus outdoors (20 sequences)  -> Results-tumvi
#   outdoors  the 8 outdoors sequences              -> Results-tumvi
#   all       everything                            -> Results
#
# outdoors shares the TUM-VI root on purpose: running it later merges into the
# same tree the tumvi group built, so the analysis scripts see one TUM-VI set.
#
# Passes are interleaved -- pass 0 covers every sequence, then pass 1, and so on
# -- so an interrupted job leaves equal iteration depth across sequences instead
# of a complete set for the first few and nothing for the rest. Completed runs
# are marked and skipped on a rerun, so resuming after an interrupt is just
# re-issuing the same command (set FORCE=1 to redo them anyway).
#
# seq_filter restricts the group to sequences matching it, e.g.
#   ./run_timing_batch.sh euroc desktop 1 MH01     # one sequence, one pass

set -u

group=${1:-euroc}
version=${2:-desktop}
num_itr=${3:-5}
filter=${4:-}

# Keep in sync with the whitelists in run_script.sh.
euroc_datasets=("MH01" "MH02" "MH03" "MH04" "MH05" \
                "V101" "V102" "V103" "V201" "V202" "V203")
tumvi_datasets=("corridor1" "corridor2" "corridor3" "corridor4" "corridor5" \
                "magistrale1" "magistrale2" "magistrale3" "magistrale4" "magistrale5" "magistrale6" \
                "room1" "room2" "room3" "room4" "room5" "room6" \
                "slides1" "slides2" "slides3")
outdoors_datasets=("outdoors1" "outdoors2" "outdoors3" "outdoors4" \
                   "outdoors5" "outdoors6" "outdoors7" "outdoors8")

case "$group" in
    euroc)    datasets=("${euroc_datasets[@]}");    RESULTS_ROOT=Results-euroc ;;
    tumvi)    datasets=("${tumvi_datasets[@]}");    RESULTS_ROOT=Results-tumvi ;;
    outdoors) datasets=("${outdoors_datasets[@]}"); RESULTS_ROOT=Results-tumvi ;;
    all)      datasets=("${euroc_datasets[@]}" "${tumvi_datasets[@]}" "${outdoors_datasets[@]}")
              RESULTS_ROOT=Results ;;
    *)        echo "unknown group '$group' (expected euroc, tumvi, outdoors or all)"; exit 1 ;;
esac
if [ -n "$filter" ]; then
    keep=()
    for d in "${datasets[@]}"; do [[ "$d" =~ $filter ]] && keep+=("$d"); done
    if [ ${#keep[@]} -eq 0 ]; then
        echo "filter '$filter' matched no sequence in group '$group'"; exit 1
    fi
    datasets=("${keep[@]}")
fi

export RESULTS_ROOT   # read by run_script.sh
interval=0.05
LOG=timing_batch_${group}.log

# Keep a durable record of a job that is expected to be interrupted and resumed.
exec > >(tee -a "$LOG") 2>&1

# Mirror run_script.sh's default kernel statuses so the stats path can be predicted
# (the monitor writes into the same directory the run does). Read them out of
# run_script.sh rather than duplicating them, so changing the defaults there is enough.
kFT=$(grep -E "^kernel_status_FT='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
kTM=$(grep -E "^kernel_status_TM='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
kFL=$(grep -E "^kernel_status_FL='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
nitro_kdir="${kFT}-${kTM}-${kFL}"
echo "[batch] $(date +%F\ %T) group=$group root=$RESULTS_ROOT version=$version iters=$num_itr${filter:+ filter=$filter}"
echo "[batch] ${#datasets[@]} sequences, Nitro-SLAM kernel dir: ${nitro_kdir}"

# Exact comm match: Linux truncates comm to 15 chars, so the process is
# "stereo_inertial". Matching on the full command line would also match this
# script's own shell and kill the driver.
cleanup() { pkill -KILL -x stereo_inertial 2>/dev/null; sleep 2; }
trap 'echo "[batch] interrupted"; cleanup; exit 130' INT TERM

# The long TUM-VI sequences (magistrale/outdoors/slides run 9k-29k frames vs
# ~2-6k for EuRoC and the room/corridor sets) do not finish inside the 900s that
# was enough for MH01, so scale the watchdog with sequence length instead.
run_timeout() {
    case "$1" in
        magistrale*|outdoors*|slides*) echo 2700 ;;
        *)                             echo 900  ;;
    esac
}

for i in $(seq 0 $((num_itr - 1))); do
    echo "[batch] ===== pass $i / $((num_itr - 1)) ====="
    for dataset in "${datasets[@]}"; do
        for cfg in "0 0 0" "1 1 1"; do
            set -- $cfg
            if [ "$1" -eq 1 ]; then
                statsDir="${RESULTS_ROOT}/Nitro-SLAM/${nitro_kdir}/${version}/${dataset}/${i}"
            else
                statsDir="${RESULTS_ROOT}/ORB-SLAM3/${version}/${dataset}/${i}"
            fi

            if [ -f "${statsDir}/.batch_done" ] && [ "${FORCE:-0}" -eq 0 ]; then
                echo "[batch] skip (done) iter=$i dataset=$dataset cfg='$cfg'"
                continue
            fi
            echo "[batch] $(date +%T) iter=$i dataset=$dataset FT=$1 TM=$2 FL=$3"

            # run_script.sh creates this itself, but the sampler starts first and
            # redirects into it, so it has to exist now.
            mkdir -p "$statsDir"

            # Start the sampler first so it captures a GPU baseline before the
            # process creates its CUDA context.
            ./scripts/monitor_memory.py --out "${statsDir}/memory.csv" \
                                --summary "${statsDir}/memory_summary.txt" \
                                --interval "$interval" --wait 120 \
                                > "${statsDir}/monitor.log" 2>&1 &
            mon_pid=$!

            timeout "$(run_timeout "$dataset")" ./run_script.sh "$dataset" "$1" "$2" "$3" 1 "$version" "$i"
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "[batch] WARNING rc=$rc for $dataset iter=$i cfg='$cfg'"
            else
                touch "${statsDir}/.batch_done"
            fi

            wait "$mon_pid" 2>/dev/null
            [ -f "${statsDir}/memory_summary.txt" ] \
                && sed -n 's/^gpu_peak_\(process\|delta\)_mib/[batch]   gpu_peak_\1/p;s/^cpu_hwm_rss_mib/[batch]   cpu_peak/p' \
                       "${statsDir}/memory_summary.txt"
            cleanup
        done
    done
done
echo "[batch] done $(date +%F\ %T) group=$group"
