#!/bin/bash
# pathDatasetEuroc='../EuRoC-Dataset' #Example, it is necesary to change it by the dataset path
pathDatasetEuroc=$HOME/SLAM/Datasets/EuRoc

FastTrack_on=$1
TurboMap_on=$2
FastLoop_on=$3
kernel_status_FT=$4
kernel_status_TM=$5
kernel_status_FL=$6
dataset_name=$7
statsDir=$8

file_name="dataset-${dataset_name}_stereoi"

echo "Launching $dataset_name with Stereo-Inertial sensor"

#with gdb
# EXECUTABLE=./Stereo-Inertial/stereo_inertial_euroc
# ARGS="../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml "${pathDatasetEuroc}"/"${dataset_name}" ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt "${file_name}" "${statsDir}" ${FastTrack_on} ${TurboMap_on} ${FastLoop_on} ${kernel_status_FT} ${kernel_status_TM}" ${kernel_status_FL}
# gdb -ex "set print thread-events off" -ex "set args $ARGS" -ex "run" $EXECUTABLE

#without gdb
CUDA_MODULE_LOADING=EAGER ./Stereo-Inertial/stereo_inertial_euroc ../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml "${pathDatasetEuroc}"/"${dataset_name}" ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt ${file_name} ${statsDir} ${FastTrack_on} ${TurboMap_on} ${FastLoop_on} ${kernel_status_FT} ${kernel_status_TM} ${kernel_status_FL}
# compute-sanitizer --tool memcheck --report-api-errors all --show-backtrace no ./Stereo-Inertial/stereo_inertial_euroc ../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml "${pathDatasetEuroc}"/"${dataset_name}" ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt "${file_name}" "${statsDir}" ${FastTrack_on} ${TurboMap_on} ${FastLoop_on} ${kernel_status_FT} ${kernel_status_TM} ${kernel_status_FL}

echo "------------------------------------"

echo "Evaluation of ${dataset_name} trajectory with Stereo-Inertial sensor"
python3 -W ignore ../evaluation/evaluate3.py ${pathDatasetEuroc}/${dataset_name}/mav0/state_groundtruth_estimate0/data.csv f_${file_name}.txt --plot ${dataset_name}_stereoi.pdf --verbose
echo "Plotting data"
python3 ../plot.py "${statsDir}"

files=("f_dataset-${dataset_name}_stereoi.csv"
"f_dataset-${dataset_name}_stereoi.txt"
"f_dataset-${dataset_name}_stereoi.png"
"kf_dataset-${dataset_name}_stereoi.txt"
)
destination_directory="${statsDir}/trajectory"
mkdir -p $destination_directory
mv "${files[@]}" "$destination_directory"