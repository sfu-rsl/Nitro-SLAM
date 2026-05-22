#!/bin/bash

# https://www.reddit.com/r/linux_gaming/comments/u0vdc3/disable_adaptive_clocking_on_2080_super_in_linux/

# sudo nvidia-smi -lgc min_clock,max_clock
# reset with -rgc
# 19002 (memory)
# 2130 Mhz (graphics)
# 9501 Mhz (memory)

echo Setting CPU profiles
#system76-power profile performance
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -d 4900MHz

# If you want to disable E cores
#echo "0" | sudo tee /sys/devices/system/cpu/cpu16/online
#echo "0" | sudo tee /sys/devices/system/cpu/cpu17/online
#echo "0" | sudo tee /sys/devices/system/cpu/cpu18/online
#echo "0" | sudo tee /sys/devices/system/cpu/cpu19/online

echo Setting PowerMizer mode
nvidia-settings -a "[gpu:0]/GpuPowerMizerMode=1"

echo Setting GPU Clocks
sudo nvidia-smi -pm 1
#sudo nvidia-smi -lgc unlimited
#sudo nvidia-smi -lmc unlimited

sudo nvidia-smi -lgc 2100
sudo nvidia-smi -lmc 9751
