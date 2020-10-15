#!/bin/bash
set -e


function test_launch_ps(){
    fleetrun --server_num=2 --worker_num=2 fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi

    fleetrun --servers="127.0.0.1:6780,127.0.0.1:6781" --workers="127.0.0.1:6782,127.0.0.1:6783" fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi

    fleetrun --servers="127.0.0.1:6780,127.0.0.1:6781" --workers="127.0.0.1,127.0.0.1" fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

function test_launch_ps_heter(){
    fleetrun --server_num=2 --worker_num=2 --heter_worker_num=2 fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test heter pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

if [[ ${WITH_GPU} == "OFF" ]]; then
    echo "in cpu test mode"
    test_launch_ps
    exit 0
fi

echo "No.1 unittest"
test_launch_ps
test_launch_ps_heter
# use default values
echo "No.2 unittest"
fleetrun multi_process.py fleetrun

# use paddlecloud
echo "begin test use paddlecloud"
cluster_node_ips="127.0.0.1,127.0.0.2"
export PADDLE_TRAINERS_NUM=2
export POD_IP=127.0.0.1
export PADDLE_TRAINERS=127.0.0.1,127.0.0.2
export PADDLE_TRAINER_ID=0

export PADDLE_PORT=35789
export TRAINER_PORTS_NUM=2

echo "No.3 unittest"
distributed_args="--ips=${cluster_node_ips} --gpus=0,1 --log_dir=testlog"
CUDA_VISIBLE_DEVICES=0,1 fleetrun ${distributed_args} multi_process.py fleetrun

str1="selected_gpus:0 worker_endpoints:127.0.0.1:35789,127.0.0.1:35790,127.0.0.2:35789,127.0.0.2:35790 trainers_num:4 current_endpoint:127.0.0.1:35789 trainer_id:0"
str2="selected_gpus:1 worker_endpoints:127.0.0.1:35789,127.0.0.1:35790,127.0.0.2:35789,127.0.0.2:35790 trainers_num:4 current_endpoint:127.0.0.1:35790 trainer_id:1"
file_0="multi_process_fleetrun.check_0.log"
file_1="multi_process_fleetrun.check_1.log"

echo "paddlecloud params test"
if grep -q "$str1" "$file_0"; then
    echo "find trainer 0"
else
    echo "not find trainer 0"
    exit -1
fi

if grep -q "$str2" "$file_1"; then
    echo "find trainer 1"
else
    echo "not find trainer 1"
    exit -1
fi

# test async poll process
if [ -f $file_0 ]; then
    rm $file_0
fi
if [ -f $file_1 ]; then
    rm $file_1
fi

# test use DISTRIBUTED_TRAINER_ENDPOINTS env in paddlecloud
unset PADDLE_PORT
export DISTRIBUTED_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171

echo "No.4 unittest"
echo "paddle.distributed.launch async poll process test"
if ! CUDA_VISIBLE_DEVICES=0,1 fleetrun ${distributed_args} multi_process.py fleetrun abort; then
    echo "train abort as planned"
fi

abort_str1="abort>>> selected_gpus:0 worker_endpoints:127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171 trainers_num:4 current_endpoint:127.0.0.1:6170 trainer_id:0"

if grep -q "$abort_str1" "$file_0"; then
    echo "trainer 0 abort as planned"
else
    echo "trainer 0 not abort as planned"
    exit -1
fi

if [ ! -f $file_1 ]; then
    echo "trainer 1 terminate as planned"
else
    echo "trainer 1 not terminate as planned"
    exit -1
fi

#test for random ports
file_0_0="test_launch_filelock_0_0.log"
file_1_0="test_launch_filelock_1_0.log"
rm -rf $file_0_0 $file_0_1

distributed_args="--gpus=0,1 --log_dir=testlog"
export PADDLE_LAUNCH_LOG="test_launch_filelock_0"
echo "No.5 unittest"
CUDA_VISIBLE_DEVICES=0,1 fleetrun ${distributed_args} find_ports.py
str_0="worker_endpoints:127.0.0.1:6070,127.0.0.1:6071"
