#!/bin/bash

if [ $# -lt 6 ]; then
    echo "Usage: $0 <dataset_name> <FastTrack[0|1]> <TurboMap[0|1]> <FastLoop[0|1]> <output[0=stdout|1=file]> <version> [kernel_status1] [kernel_status2] [kernel_status3]"
    exit 1
fi

dataset_name=$1
FastTrack_on=$2
TurboMap_on=$3
FastLoop_on=$4
save_ostream=$5
version=$6
iteration=$7

if [ "$FastTrack_on" -eq 1 ]; then
    system_name+="FastTrack"
fi

if [ "$TurboMap_on" -eq 1 ]; then
    system_name+="${system_name:+&}TurboMap"
fi

if [ "$FastLoop_on" -eq 1 ]; then
    system_name+="${system_name:+&}FastLoop"
fi

# If all optimizations enabled → Nitro-SLAM
if [ "$FastTrack_on" -eq 1 ] && [ "$TurboMap_on" -eq 1 ] && [ "$FastLoop_on" -eq 1 ]; then
    system_name="Nitro-SLAM"
fi

# If no optimizations enabled → ORB-SLAM3
if [ -z "$system_name" ]; then
    system_name="ORB-SLAM3"
fi

# Defaults

# FastTrack
kernel_status_FT='11111' # all - don't skip pose optimization (last flag)
# kernel_status_FT='11110' # Skip pose optimization

# TurboMap
kernel_status_TM='1111' # all

# FastLoop
# kernel_status_FL='111111'
kernel_status_FL='001111' # disable merged projection searches


# Optional arguments
kernel_status1=${8:-}
kernel_status2=${9:-}
kernel_status3=${10:-}

# FastTrack kernel status
if [ "$FastTrack_on" -eq 1 ] && [ -n "$kernel_status1" ]; then
    kernel_status_FT="$kernel_status1"
fi

# TurboMap kernel status
if [ "$TurboMap_on" -eq 1 ]; then
    if [ "$FastTrack_on" -eq 1 ] && [ -n "$kernel_status2" ]; then
        kernel_status_TM="$kernel_status2"
    elif [ "$FastTrack_on" -eq 0 ] && [ -n "$kernel_status1" ]; then
        kernel_status_TM="$kernel_status1"
    fi
fi

# FastLoop kernel status
if [ "$FastLoop_on" -eq 1 ]; then
    if [ "$FastTrack_on" -eq 1 ] && [ "$TurboMap_on" -eq 1 ] && [ -n "$kernel_status3" ]; then
        kernel_status_FL="$kernel_status3"
    elif [ "$FastTrack_on" -eq 0 ] && [ "$TurboMap_on" -eq 0 ] && [ -n "$kernel_status1" ]; then
        kernel_status_FL="$kernel_status1"
    elif { [ "$FastTrack_on" -eq 1 ] || [ "$TurboMap_on" -eq 1 ]; } && [ -n "$kernel_status2" ]; then
        kernel_status_FL="$kernel_status2"
    fi
fi

statsDir="./Results/${system_name}"

# Append kernel statuses only for enabled optimizations
kernel_dir=""
[ "$FastTrack_on" -eq 1 ] && kernel_dir+="${kernel_status_FT}"
[ "$TurboMap_on" -eq 1 ] && kernel_dir+="${kernel_dir:+-}${kernel_status_TM}"
[ "$FastLoop_on" -eq 1 ] && kernel_dir+="${kernel_dir:+-}${kernel_status_FL}"

[ -n "$kernel_dir" ] && statsDir+="/${kernel_dir}"

statsDir+="/${version}/${dataset_name}/${iteration}"

if [ ! -d "$statsDir" ]; then
    mkdir -p "$statsDir"
fi

tumvi_datasets=("corridor1" "corridor2" "corridor3" "corridor4" "corridor5" \
                "magistrale1" "magistrale2" "magistrale3" "magistrale4" "magistrale5" "magistrale6" \
                "outdoors1" "outdoors2" "outdoors3" "outdoors4" "outdoors5" "outdoors6" "outdoors7" "outdoors8" \
                "room1" "room2" "room3" "room4" "room5" "room6" \
                "slides1" "slides2" "slides3")
euroc_datasets=("MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "V201" "V202" "V203")

found_in_tumvi=false
for dataset in "${tumvi_datasets[@]}"; do
    if [[ "$dataset" == "$dataset_name" ]]; then
        found_in_tumvi=true
        break
    fi
done

found_in_euroc=false
for dataset in "${euroc_datasets[@]}"; do
    if [[ "$dataset" == "$dataset_name" ]]; then
        found_in_euroc=true
        break
    fi
done

if [ "$save_ostream" -eq 0 ]; then
    if $found_in_euroc; then
        cd Examples/
        ./euroc_eval_examples.sh "$FastTrack_on" "$TurboMap_on" "$FastLoop_on" "$kernel_status_FT" "$kernel_status_TM" "$kernel_status_FL" "$dataset_name" "../$statsDir" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$FastTrack_on" "$TurboMap_on" "$FastLoop_on" "$kernel_status_FT" "$kernel_status_TM" "$kernel_status_FL" "$dataset_name" "../$statsDir" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
else
    if $found_in_euroc; then
        cd Examples/
        ./euroc_eval_examples.sh "$FastTrack_on" "$TurboMap_on" "$FastLoop_on" "$kernel_status_FT" "$kernel_status_TM" "$kernel_status_FL" "$dataset_name" "../$statsDir" > "../${statsDir}/ostream.txt" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$FastTrack_on" "$TurboMap_on" "$FastLoop_on" "$kernel_status_FT" "$kernel_status_TM" "$kernel_status_FL" "$dataset_name" "../$statsDir" > "../${statsDir}/ostream.txt" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
fi