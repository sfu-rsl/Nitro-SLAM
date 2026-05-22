#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <FastTrack[0|1]> <TurboMap[0|1]> <FastLoop[0|1]> <version> <num_iterations>"
    exit 1
fi

FastTrack_on=$1
TurboMap_on=$2
FastLoop_on=$3
version=$4
num_itr=$5

datasets=("room3" "room4" "corridor1" "magistrale1")
# datasets=("MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "room1" "room2" "room3" "room4" "room5" "room6" "corridor1" "corridor2" "corridor3" "magistrale1")

for i in $(seq 0 $(expr $num_itr - 1)); do
    for dataset in "${datasets[@]}"; do
        echo -e "[bash:] -> ./run_script.sh $dataset $FastTrack_on $TurboMap_on $FastLoop_on 1 $version.$i"
        ./run_script.sh $dataset $FastTrack_on $TurboMap_on $FastLoop_on 1 $version.$i
    done
done

# for i in $(seq 0 $(expr $num_itr - 1)); do
#     for dataset in "${datasets[@]}"; do
#         echo -e "[bash:] -> ./run_script.sh $dataset 0 0 0 1 $version.$i"
#         ./run_script.sh $dataset 0 0 0 1 $version.$i
#     done
# done