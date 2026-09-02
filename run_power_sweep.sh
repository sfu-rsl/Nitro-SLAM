#!/bin/bash
# Power / CPU / GPU sweep: one run of every sequence in a group, under both systems.
#
#   ./run_power_sweep.sh [version] [group ...]      groups: euroc tumvi outdoors all
#
# Companion to run_timing_batch.sh, but one iteration instead of five and sampling
# monitor_power.py instead of monitor_memory.py. The point is not statistics per run,
# it is coverage: enough sequences to characterise power and utilisation per sequence
# *type* (MH, vicon1, vicon2, corridor, magistrale, room, slides, outdoors).
#
# A cell whose power_summary.txt already exists is skipped, so sequences captured by
# run_power_probe.sh are reused and an interrupted sweep resumes by re-issuing the
# same command. FORCE=1 redoes them.
set -u
cd "$(dirname "$0")"

version=${1:-jetson-power}; shift || true
groups=("$@"); [ ${#groups[@]} -eq 0 ] && groups=(euroc tumvi outdoors)
interval=0.1
kFT=11111; kTM=1111; kFL=001111
nitro_kdir="${kFT}-${kTM}-${kFL}"

LOG=power_sweep.log
exec > >(tee -a "$LOG") 2>&1

euroc=(MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203)
tumvi=(corridor1 corridor2 corridor3 corridor4 corridor5 \
       magistrale1 magistrale2 magistrale3 magistrale4 magistrale5 magistrale6 \
       room1 room2 room3 room4 room5 room6 slides1 slides2 slides3)
outdoors=(outdoors1 outdoors2 outdoors3 outdoors4 outdoors5 outdoors6 outdoors7 outdoors8)

JOBS=()
for g in "${groups[@]}"; do
    case "$g" in
        euroc)    for s in "${euroc[@]}";    do JOBS+=("$s:Results-euroc:900");  done ;;
        tumvi)    for s in "${tumvi[@]}";    do JOBS+=("$s:Results-tumvi:2700"); done ;;
        outdoors) for s in "${outdoors[@]}"; do JOBS+=("$s:Results-tumvi:2700"); done ;;
        *) echo "[sweep] unknown group '$g' (euroc|tumvi|outdoors)"; exit 1 ;;
    esac
done

cleanup() { pkill -KILL -x stereo_inertial 2>/dev/null; sleep 2; }
trap 'echo "[sweep] interrupted"; cleanup; exit 130' INT TERM

echo "[sweep] $(date +%F\ %T) version=$version groups=${groups[*]} jobs=${#JOBS[@]} (x2 configs)"

for job in "${JOBS[@]}"; do
    IFS=: read -r ds root wd <<< "$job"
    export RESULTS_ROOT="$root"
    for cfg in "0 0 0" "1 1 1"; do
        set -- $cfg
        if [ "$1" -eq 1 ]; then
            statsDir="${root}/Nitro-SLAM/${nitro_kdir}/${version}/${ds}/0"; label=Nitro-SLAM
        else
            statsDir="${root}/ORB-SLAM3/${version}/${ds}/0"; label=ORB-SLAM3
        fi
        if [ -s "${statsDir}/power_summary.txt" ] && [ "${FORCE:-0}" -eq 0 ]; then
            echo "[sweep] skip (done) $ds $label"; continue
        fi
        echo "[sweep] $(date +%T) START $ds $label"
        mkdir -p "$statsDir"
        ./monitor_power.py --out "${statsDir}/power.csv" \
                           --summary "${statsDir}/power_summary.txt" \
                           --events "${statsDir}/events.csv" \
                           --ostream "${statsDir}/ostream.txt" \
                           --interval "$interval" --wait 240 \
                           > "${statsDir}/power_monitor.log" 2>&1 &
        mon=$!
        timeout "$wd" ./run_script.sh "$ds" "$1" "$2" "$3" 1 "$version" 0 "$kFT" "$kTM" "$kFL"
        rc=$?
        wait "$mon" 2>/dev/null
        if grep -q 'absolute_translational_error.rmse' "${statsDir}/ostream.txt" 2>/dev/null; then
            echo "[sweep] $(date +%T) OK    $ds $label rc=$rc"
        else
            echo "[sweep] $(date +%T) FAIL  $ds $label rc=$rc"
            tail -3 "${statsDir}/stderr.txt" 2>/dev/null | sed 's/^/[sweep] | /'
        fi
        cleanup
    done
done
echo "[sweep] done $(date +%F\ %T)"
