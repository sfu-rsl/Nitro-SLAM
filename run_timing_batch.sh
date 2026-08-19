#!/bin/bash
# Batch driver for the thread-breakdown timing study.
#   ./run_timing_batch.sh <version> [num_iterations]
# Runs every dataset under both ORB-SLAM3 (0 0 0) and Nitro-SLAM (1 1 1), and
# samples GPU/CPU memory alongside each run into <statsDir>/memory.csv.

set -u

version=${1:-timing}
num_itr=${2:-5}

# Every EuRoC and TUM-VI sequence available under $HOME/SLAM/Datasets.
# Keep in sync with the whitelists in run_script.sh.
euroc_datasets=("MH01" "MH02" "MH03" "MH04" "MH05" \
                "V101" "V102" "V103" "V201" "V202" "V203")
tumvi_datasets=("corridor1" "corridor2" "corridor3" "corridor4" "corridor5" \
                "magistrale1" "magistrale2" "magistrale3" "magistrale4" "magistrale5" "magistrale6" \
                "outdoors1" "outdoors2" "outdoors3" "outdoors4" "outdoors5" "outdoors6" "outdoors7" "outdoors8" \
                "room1" "room2" "room3" "room4" "room5" "room6" \
                "slides1" "slides2" "slides3")
datasets=("${euroc_datasets[@]}" "${tumvi_datasets[@]}")
interval=0.05

# Mirror run_script.sh's default kernel statuses so the stats path can be predicted
# (the monitor writes into the same directory the run does). Read them out of
# run_script.sh rather than duplicating them, so changing the defaults there is enough.
kFT=$(grep -E "^kernel_status_FT='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
kTM=$(grep -E "^kernel_status_TM='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
kFL=$(grep -E "^kernel_status_FL='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
nitro_kdir="${kFT}-${kTM}-${kFL}"
echo "[batch] Nitro-SLAM kernel dir: ${nitro_kdir}"

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
    for dataset in "${datasets[@]}"; do
        for cfg in "0 0 0" "1 1 1"; do
            set -- $cfg
            if [ "$1" -eq 1 ]; then
                statsDir="Results/Nitro-SLAM/${nitro_kdir}/${version}/${dataset}/${i}"
            else
                statsDir="Results/ORB-SLAM3/${version}/${dataset}/${i}"
            fi
            echo "[batch] $(date +%T) iter=$i dataset=$dataset FT=$1 TM=$2 FL=$3"

            # run_script.sh creates this itself, but the sampler starts first and
            # redirects into it, so it has to exist now.
            mkdir -p "$statsDir"

            # Start the sampler first so it captures a GPU baseline before the
            # process creates its CUDA context.
            ./monitor_memory.py --out "${statsDir}/memory.csv" \
                                --summary "${statsDir}/memory_summary.txt" \
                                --interval "$interval" --wait 120 \
                                > "${statsDir}/monitor.log" 2>&1 &
            mon_pid=$!

            timeout "$(run_timeout "$dataset")" ./run_script.sh "$dataset" "$1" "$2" "$3" 1 "$version" "$i"
            rc=$?
            [ $rc -ne 0 ] && echo "[batch] WARNING rc=$rc for $dataset iter=$i cfg='$cfg'"

            wait "$mon_pid" 2>/dev/null
            [ -f "${statsDir}/memory_summary.txt" ] \
                && sed -n 's/^gpu_peak_\(process\|delta\)_mib/[batch]   gpu_peak_\1/p;s/^cpu_hwm_rss_mib/[batch]   cpu_peak/p' \
                       "${statsDir}/memory_summary.txt"
            cleanup
        done
    done
done
echo "[batch] done $(date +%T)"
