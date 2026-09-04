#!/bin/bash
# Re-run individual (system, dataset, iteration) runs.
#
#   ./run_jobs.sh <version> <system>:<dataset>:<iter>[:<kFT>:<kTM>:<kFL>] ...
#
# run_timing_batch.sh sweeps every dataset under both systems, which is the right tool
# for a full batch but a poor one for filling in a handful of runs that crashed, or for
# probing a single sequence with non-default kernel statuses. This runs exactly the jobs
# named and nothing else, while mirroring run_timing_batch.sh's setup -- same memory
# sampler, same interval, same length-scaled watchdog, same cleanup -- so the results are
# directly comparable to the runs they replace.
#
#   <system>  orb (0 0 0) | nitro (1 1 1)
#   <kFT/kTM/kFL>  optional kernel-status overrides; default to run_script.sh's values.
#                  Non-default statuses land in their own Results/ tree, since the kernel
#                  status is part of the path, so they cannot collide with a stock batch.
#
# Examples:
#   # fill in three crashed iterations
#   ./run_jobs.sh desktop orb:room5:0 orb:room5:1 orb:room5:2
#
#   # same flags but pose-graph optimisation on the CPU (FL bit 5 clear)
#   ./run_jobs.sh cpupgo nitro:magistrale1:0:11111:1111:001101
#
# Must be invoked with cwd = repo root. Prints a per-run OK/FAIL verdict as it goes, and
# dumps the tail of stderr.txt on failure so an abort message is visible in the log.
set -u

if [ $# -lt 2 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

version=$1; shift
interval=0.05

# Read the defaults out of run_script.sh rather than duplicating them, so changing the
# defaults there is enough (same approach as run_timing_batch.sh).
dFT=$(grep -E "^kernel_status_FT='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
dTM=$(grep -E "^kernel_status_TM='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
dFL=$(grep -E "^kernel_status_FL='" run_script.sh | head -1 | sed "s/.*'\([01]*\)'.*/\1/")
echo "[jobs] defaults FT=$dFT TM=$dTM FL=$dFL"
echo "[jobs] ${#} job(s): $*"

# Exact comm match: Linux truncates comm to 15 chars, so the process is "stereo_inertial".
# Matching the full command line would also match this script's own shell.
cleanup() { pkill -KILL -x stereo_inertial 2>/dev/null; sleep 2; }
trap 'echo "[jobs] interrupted"; cleanup; exit 130' INT TERM

# The long TUM-VI sequences do not finish inside the 900s that suits the room/corridor
# sets, so scale the watchdog with sequence length.
run_timeout() {
    case "$1" in
        magistrale*|outdoors*|slides*) echo 2700 ;;
        *)                             echo 900  ;;
    esac
}

fail=0
for job in "$@"; do
    IFS=: read -r sysname dataset i jFT jTM jFL <<< "$job"
    kFT=${jFT:-$dFT}; kTM=${jTM:-$dTM}; kFL=${jFL:-$dFL}
    case "$sysname" in
        orb)   FT=0; TM=0; FL=0; statsDir="Results/ORB-SLAM3/${version}/${dataset}/${i}" ;;
        nitro) FT=1; TM=1; FL=1; statsDir="Results/Nitro-SLAM/${kFT}-${kTM}-${kFL}/${version}/${dataset}/${i}" ;;
        *) echo "[jobs] bad system '$sysname' in job '$job' (want orb|nitro)"; fail=1; continue ;;
    esac

    echo "[jobs] $(date +%T) START $sysname $dataset iter=$i  (FT=$FT TM=$TM FL=$FL  kFL=$kFL)"

    # run_script.sh creates this itself, but the sampler starts first and redirects into
    # it, so it has to exist now.
    mkdir -p "$statsDir"

    # Start the sampler first so it captures a GPU baseline before the process creates
    # its CUDA context.
    ./scripts/monitor_memory.py --out "${statsDir}/memory.csv" \
                        --summary "${statsDir}/memory_summary.txt" \
                        --interval "$interval" --wait 120 \
                        > "${statsDir}/monitor.log" 2>&1 &
    mon_pid=$!

    timeout "$(run_timeout "$dataset")" ./run_script.sh "$dataset" "$FT" "$TM" "$FL" 1 "$version" "$i" "$kFT" "$kTM" "$kFL"
    rc=$?
    [ $rc -ne 0 ] && { echo "[jobs] WARNING rc=$rc for $sysname $dataset iter=$i"; fail=1; }
    wait "$mon_pid" 2>/dev/null

    # A run only counts if the binary reached the save path and the evaluator produced an
    # ATE; a crash leaves ostream.txt truncated and no trajectory behind.
    dur=$(grep -m1 duration_s "${statsDir}/memory_summary.txt" 2>/dev/null | awk '{print $2}')
    if grep -q 'absolute_translational_error.rmse' "${statsDir}/ostream.txt" 2>/dev/null; then
        rmse=$(grep -m1 'absolute_translational_error.rmse' "${statsDir}/ostream.txt" | awk '{print $2}')
        echo "[jobs] $(date +%T) OK    $sysname $dataset iter=$i  rmse=$rmse  dur=${dur}s"
    else
        echo "[jobs] $(date +%T) FAIL  $sysname $dataset iter=$i  dur=${dur}s"
        if [ -s "${statsDir}/stderr.txt" ]; then
            echo "[jobs] ---- stderr (last 15 lines) ----"
            tail -15 "${statsDir}/stderr.txt" | sed 's/^/[jobs] | /'
            echo "[jobs] --------------------------------"
        else
            echo "[jobs] (stderr.txt empty or absent)"
        fi
        fail=1
    fi
    cleanup
done
echo "[jobs] done $(date +%T) fail=$fail"
