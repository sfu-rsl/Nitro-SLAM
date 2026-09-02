#!/bin/bash
# Power / CPU / GPU time series for a small set of representative runs.
#
#   ./run_power_probe.sh [version] [seq:results_root:watchdog ...]
#
# One iteration of each sequence under both ORB-SLAM3 (0 0 0) and Nitro-SLAM (1 1 1),
# with monitor_power.py sampling the INA3221 rails and tegra utilisation alongside.
# Unlike run_timing_batch.sh this is not a statistics run -- it is one trace per cell,
# chosen so the interesting events are in them:
#
#   outdoors7  TUM-VI  loop closure with a 5.6 s global BA (the largest in the study)
#   MH05       EuRoC   machine hall, loop closure under Nitro-SLAM
#   room3      TUM-VI  small room, loop closure under both systems
#
# monitor_memory.py is deliberately not run alongside: it is a second jtop client for
# the same daemon, and monitor_power.py already records the run's RSS and system RAM.
set -u
cd "$(dirname "$0")"

version=${1:-jetson-power}; shift || true
interval=0.1
LOG=power_probe.log
exec > >(tee -a "$LOG") 2>&1

kFT=11111; kTM=1111; kFL=001111
nitro_kdir="${kFT}-${kTM}-${kFL}"

# seq:results_root:watchdog
JOBS=(
  "outdoors7:Results-tumvi:2700"
  "MH05:Results-euroc:900"
  "room3:Results-tumvi:900"
)
# Loop closure is stochastic: measured over the 5-run sweep, outdoors7 closed on only
# 2/5 Nitro-SLAM iterations, while outdoors5 closed 5/5 and magistrale2 5/5. Pass those
# explicitly when the trace needs a closure in it.
[ $# -gt 0 ] && JOBS=("$@")

cleanup() { pkill -KILL -x stereo_inertial 2>/dev/null; sleep 2; }
trap 'echo "[power] interrupted"; cleanup; exit 130' INT TERM

echo "[power] $(date +%F\ %T) version=$version interval=${interval}s"

for job in "${JOBS[@]}"; do
    IFS=: read -r ds root wd <<< "$job"
    export RESULTS_ROOT="$root"
    for cfg in "0 0 0" "1 1 1"; do
        set -- $cfg
        if [ "$1" -eq 1 ]; then
            statsDir="${root}/Nitro-SLAM/${nitro_kdir}/${version}/${ds}/0"
            label="Nitro-SLAM"
        else
            statsDir="${root}/ORB-SLAM3/${version}/${ds}/0"
            label="ORB-SLAM3"
        fi
        echo "[power] $(date +%T) START $ds $label"
        mkdir -p "$statsDir"

        # Sampler first, so it captures a pre-run baseline. It follows the run's own
        # ostream.txt, which run_script.sh creates on redirect, to timestamp markers.
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
            echo "[power] $(date +%T) OK    $ds $label rc=$rc rmse=$(grep -m1 'absolute_translational_error.rmse' "${statsDir}/ostream.txt" | awk '{print $2}')"
        else
            echo "[power] $(date +%T) FAIL  $ds $label rc=$rc"
            tail -5 "${statsDir}/stderr.txt" 2>/dev/null | sed 's/^/[power] | /'
        fi
        sed -n 's/^\(power_mean_mw\|power_peak_mw\|energy_j\|gpu_load_mean_pct\|cpu_total_mean_pct\|events\) */[power]   &/p' \
            "${statsDir}/power_summary.txt" 2>/dev/null
        cleanup
    done
done
echo "[power] done $(date +%F\ %T)"
