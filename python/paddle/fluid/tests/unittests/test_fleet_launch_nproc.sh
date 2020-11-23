#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
export FLAGS_START_PORT=35789

local_ip=`ip route get 1 | awk '{print $NF;exit}'`
echo "local ip:${local_ip}"
file_0="fleet_nproc_0.check_0.log"

function test_nproc_0(){
    gpus=$1
    rm -f ${file_0}
    distributed_args="--log_dir=testlog --nproc_per_node=1"
    # nproc_per_node=1, each with 2 gpus
    python -m paddle.distributed.fleet.launch ${distributed_args} nproc_process.py  fleet_nproc_0

    str0="selected_gpus:${gpus} worker_endpoints:127.0.0.1:35789 trainers_num:1 current_endpoint:127.0.0.1:35789 trainer_id:0"
    if grep -q "$str0" "$file_0"; then
        echo "find trainer 0"
    else
        echo "not find trainer 0"
        exit -1
    fi
}

# gpu
export CUDA_VISIBLE_DEVICES=0,1
test_nproc_0 "0,1"

# cpu
export CUDA_VISIBLE_DEVICES=""
test_nproc_0 ""

# nproc_per_node=2, each with 1 gpus
export CUDA_VISIBLE_DEVICES=0,1
file_0="fleet_nproc_1.check_0.log"
file_1="fleet_nproc_1.check_1.log"
rm -f ${file_0} ${file_1}
distributed_args="--log_dir=testlog --nproc_per_node=2"
python -m paddle.distributed.fleet.launch ${distributed_args} nproc_process.py  fleet_nproc_1

str0="selected_gpus:0 worker_endpoints:${local_ip}:35789,${local_ip}:35790 trainers_num:2 current_endpoint:127.0.0.1:35789 trainer_id:0"
if grep -q "$str0" "$file_0"; then
    echo "find trainer 0"
else
    echo "not find trainer 0"
    exit -1
fi

str1="selected_gpus:0 worker_endpoints:${local_ip}:35789,${local_ip}:35790 trainers_num:2 current_endpoint:${local_ip}:35790 trainer_id:1"
if grep -q "$str1" "$file_1"; then
    echo "find trainer 1"
else
    echo "not find trainer 1"
    exit -1
fi
