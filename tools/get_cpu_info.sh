#!/bin/bash

if [ "`uname -s`" != "Linux" ]; then
  echo "Current scenario only support in Linux yet!"
  exit 0
fi

echo "********** Hardware Information **********"
sockets=`grep 'physical id' /proc/cpuinfo | sort -u | wc -l`
cores_per_socket=`grep 'core id' /proc/cpuinfo | sort -u | wc -l`
ht=`lscpu |grep "per core" |awk -F':' '{print $2}'|xargs`
physical_cores=$((sockets * cores_per_socket))
virtual_cores=`grep 'processor' /proc/cpuinfo | sort -u | wc -l`
numa_nodes=`lscpu |grep "NUMA node(s)"|awk -F':' '{print $2}'|xargs`
echo "CPU Name               : `cat /proc/cpuinfo |grep -i "model name" |uniq |awk -F ':' '{print $2}'|xargs`"
echo "CPU Family             : `lscpu |grep \"CPU family\" |awk -F':' '{print $2}'|xargs`"
echo "Socket Number          : $sockets"
echo "Cores Per Socket       : $cores_per_socket"
echo "Total Physical Cores   : $physical_cores"
echo "Total Virtual Cores    : $virtual_cores"
if [ $ht -eq 1 ]; then
  echo "Hyper Threading        : OFF"
  if [ $physical_cores -ne $virtual_cores ]; then
    echo "Error: HT logical error"
  fi
else
  echo "Hyper Threading        : ON"
  if [ $physical_cores -ge $virtual_cores ]; then
    echo "Error: HT logical error"
  fi
fi
echo "NUMA Nodes             : $numa_nodes"
if [ $numa_nodes -lt $sockets ]; then
  echo "Warning: NUMA node is not enough for the best performance,\
 at least $sockets"
fi

echo "********** Software Information **********"
echo "OS Version             : `uname -o`"
echo "Kernel Release Version : `uname -r`"
echo "Kernel Patch Version   : `uname -v`"
echo "GCC Version            :`gcc --version | head -n 1|awk -F '\\\(GCC\\\)' '{print $2}'`"
if command -v cmake >/dev/null 2>&1; then 
  cmake_ver=`cmake --version | head -n 1 | awk -F 'version' '{print $2}'`
else
  cmake_ver=" Not installed"
fi
echo "CMake Version          :$cmake_ver"
echo "******************************************"
