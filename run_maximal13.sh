#!/bin/bash
#
# Waits for any in-flight ablation driver to finish, then runs the 13-kernel
# maximal set: everything except GPU ORB extraction.
#
#   FastTrack 01111  stereoMatch + searchLocalPoints + poseEstimation (+ poseOpt on)
#   TurboMap  1111   all four
#   FastLoop  111111 all six
#
# Kept separate from run_ablation.sh so that script can stay untouched while it
# is still executing -- bash reads a script incrementally as it runs.

set -u

DATASET=outdoors5
TIMEOUT=45m
LOG=ablation_driver.log
LABEL=maximal13
KS_FT=01111
KS_TM=1111
KS_FL=111111

# Wait out the running driver, if any. Bracket the pattern so pgrep cannot
# match this script's own command line.
while pgrep -f '[r]un_ablation.sh' > /dev/null; do
    sleep 30
done
# And any straggler binary from it.
while pgrep -f '[s]tereo_inertial_tum_vi' > /dev/null; do
    sleep 30
done

echo "=== $LABEL start $(date -Is) ===" >> "$LOG"

for pass in 0 1 2; do
    version="abl.${pass}"
    dir="Results/FastTrack&TurboMap&FastLoop/${KS_FT}-${KS_TM}-${KS_FL}/${DATASET}/${version}"

    if [ -f "$dir/ostream.txt" ] && grep -q "End of saving trajectory" "$dir/ostream.txt"; then
        echo "[$(date -Is)] pass $pass  $LABEL  SKIP (already complete)" >> "$LOG"
        continue
    fi
    [ -d "$dir" ] && rm -rf "$dir"

    echo "[$(date -Is)] pass $pass  $LABEL  (FT=1 TM=1 FL=1 ks=$KS_FT,$KS_TM,$KS_FL)" >> "$LOG"
    start=$SECONDS
    timeout --signal=KILL "$TIMEOUT" ./run_script.sh "$DATASET" 1 1 1 1 "$version" \
        "$KS_FT" "$KS_TM" "$KS_FL"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        pkill -KILL -f stereo_inertial_tum_vi 2>/dev/null
    fi
    echo "[$(date -Is)] pass $pass  $LABEL  rc=$rc  $((SECONDS - start))s" >> "$LOG"
done

echo "=== $LABEL done $(date -Is) ===" >> "$LOG"
