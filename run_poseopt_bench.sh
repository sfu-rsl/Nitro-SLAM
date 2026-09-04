#!/bin/bash
# Pose-optimization A/B driver.
#
#   ./run_poseopt_bench.sh "<arms>" "<sequences>" <iterations>
#
# Arms are "cpu" and "gpu", which differ only in the sixth FastTrack bit --
# the one that picks between g2o and the fused single-kernel GPU solver. Every
# other kernel stays enabled in both, so the comparison isolates the pose
# optimizer and the mode lands in the results path via the kernel directory.
#
# Results go to their own root so this study never mixes with the timing
# batches in Results-euroc / Results-tumvi. Defaults to the six TUM-VI room
# sequences: they are short, so a full A/B pass is minutes rather than hours.
set -u

arms=${1:-"cpu gpu"}
seqs=${2:-"room1 room2 room3 room4 room5 room6"}
num_itr=${3:-1}

export RESULTS_ROOT=Results-poseopt
LOG=poseopt_bench.log
exec > >(tee -a "$LOG") 2>&1

echo "[po-bench] $(date +%F\ %T) arms='${arms}' seqs='${seqs}' iters=${num_itr} root=${RESULTS_ROOT}"

for i in $(seq 0 $((num_itr - 1))); do
    for seq in $seqs; do
        for arm in $arms; do
            case "$arm" in
                cpu) ft_bits='111110' ;;   # g2o
                gpu) ft_bits='111111' ;;   # fused GPU kernel
                *)   echo "[po-bench] unknown arm '${arm}'"; exit 1 ;;
            esac
            echo "[po-bench] ${seq} arm=${arm} pass=${i}"
            ./run_script.sh "$seq" 1 1 1 1 "$arm" "$i" "$ft_bits"
        done
    done
done

echo "[po-bench] done $(date +%F\ %T)"
