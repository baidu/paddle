#!/bin/bash

clear

gpu_id=$1

echo "gpu_id = ${gpu_id}"

export ascend_include_path="/usr/local/Ascend/ascend-toolkit/5.0.2.alpha002/arm64-linux/fwkacllib/include"
export mpi_inc_path="/home/fuhaohan/src/third_party/openmpi/include"
export mpi_ld_path="/home/fuhaohan/src/third_party/openmpi/lib"
export ascend_ld_path="/usr/local/Ascend/ascend-toolkit/5.0.2.alpha002/fwkacllib/lib64"

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${mpi_ld_path}
export HCCL_ALLOWLIST_FILE=/home/fuhaohan/src/run/out/config/hccl_aLlow_list_27.txt
export HCCL_WHITELIST_DISABLE=1
export HCCL_SECURITY_MODE=1
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/5.0.2.alpha002/opp

export FLAGS_allocator_strategy=naive_best_fit
export PADDLE_DIR=/home/gongwb/go/src/github.com/Paddle
export RANK_ID=${gpu_id}
export PADDLE_TRAINER_ID=${gpu_id}
export DEVICE_ID=${gpu_id}
export FLAGS_selected_npus=${gpu_id}
export PROJECT="build_ubuntu_pipopt_release_ascend_y_none_3.7.5"
export GLOG_v=10
export ASCEND_GLOBAL_LOG_LEVEL=0
BIN_PATH=${PADDLE_DIR}/build/${PROJECT}/paddle/fluid/operators/collective
#ls -l ${BIN_PATH}
stdbuf -oL ${BIN_PATH}/c_allreduce_sum_op_npu_test

