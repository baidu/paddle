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

set +ex

[ -z "$PADDLE_ROOT" ] && PADDLE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

# PR modify op source files
CHANGE_CC_OP_FILES=()
CHANGE_CU_OP_FILES=()

# ops that will run benchmark test
declare -A CHANGE_CC_OP_MAP
declare -A CHANGE_CU_OP_MAP

# ops that benchmark repo has
declare -A BENCHMARK_CC_OP_MAP
declare -A BENCHMARK_CU_OP_MAP

# searched header files
declare -A INCLUDE_SEARCH_MAP

function LOG {
  echo "[$0:${BASH_LINENO[0]}] $*" >&2
}

# Limit file directory
function match_file_directory {
  local sub_dir file_dir
  file_dir=$(dirname ${1})
  for sub_dir in "" "/elementwise" "/reduce_ops"
  do
    [ "${file_dir}" == "paddle/fluid/operators${sub_dir}" ] && return 0
  done
  return 1
}

# Load op files by header file
function load_CHANGE_OP_FILES_by_header_file {
  local change_file
  for change_file in $(grep -rl "${1}" paddle/fluid/operators)
  do
    if [[ "$change_file" =~ "_op.cu" ]]
    then
      # match file directory limit
      match_file_directory $change_file || continue
      LOG "[INFO] Found \"${1}\" include by \"${change_file}\"."
      CHANGE_CU_OP_FILES[${#CHANGE_CU_OP_FILES[@]}]="$change_file"
    elif [[ "$change_file" =~ "_op.cc" ]]
    then
      # match file directory limit
      match_file_directory $change_file || continue
      LOG "[INFO] Found \"${1}\" include by \"${change_file}\"."
      CHANGE_CC_OP_FILES[${#CHANGE_CC_OP_FILES[@]}]="$change_file"
    elif [[ "$change_file" =~ ".h" ]]
    then
      [ -n "${INCLUDE_SEARCH_MAP[$change_file]}" ] && continue
      LOG "[INFO] Found \"${1}\" include by \"${change_file}\", keep searching."
      INCLUDE_SEARCH_MAP[$change_file]="searched"
      load_CHANGE_OP_FILES_by_header_file $change_file
    fi
  done
}

# Load op files that PR changes
function load_CHANGE_OP_FILES {
  local sub_dir change_file
  # TODO(Avin0323): Need to filter the files added by the new OP.
  for change_file in $(git diff --name-only origin/develop)
  do
    # match directory limit
    [[ "$change_file" =~ "paddle/fluid/operators/" ]] || continue
    # match file name limit
    if [[ "$change_file" =~ "_op.cu" ]]
    then
      # match cu file directory limit
      match_file_directory $change_file || continue
      LOG "[INFO] Found \"${change_file}\" changed."
      CHANGE_CU_OP_FILES[${#CHANGE_CU_OP_FILES[@]}]="$change_file"
    elif [[ "$change_file" =~ "_op.cc" ]]
    then
      # match cc file directory limit
      match_file_directory $change_file || continue
      LOG "[INFO] Found \"${change_file}\" changed."
      CHANGE_CC_OP_FILES[${#CHANGE_CC_OP_FILES[@]}]="$change_file"
    elif [[ "$change_file" =~ ".h" ]]
    then
      LOG "[INFO] Found \"${change_file}\" changed, keep searching."
      INCLUDE_SEARCH_MAP[${change_file}]="searched"
      load_CHANGE_OP_FILES_by_header_file $change_file
    fi
  done
  if [ ${#CHANGE_CU_OP_FILES[@]} -eq 0 -a ${#CHANGE_CU_OP_FILES[@]} -eq 0 ]
  then
    LOG "[INFO] No op to test, skip this ci." && exit 0
  fi
}

# Clone benchmark repo
function prepare_benchmark_environment {
  LOG "[INFO] Clone benchmark repo ..."
  git clone https://github.com/PaddlePaddle/benchmark.git
  [ $? -ne 0 ] && LOG "[FATAL] Clone benchmark repo fail." && exit -1
  LOG "[INFO] Collect api info ..."
  python benchmark/api/deploy/collect_api_info.py \
      --test_module_name tests_v2                 \
      --info_file api_info.txt >& 2
  [ $? -ne 0 ] && LOG "[FATAL] Collect api info fail." && exit -1
  if [ ! -f benchmark/ci/scripts/op_benchmark.config ]
  then
    LOG "[FATAL] Missing op_benchmark.config!" && exit -1
  fi
}

# Load unique op name from CHANGE_OP_FILES
function load_CHANGE_OP_MAP {
  local op_name change_file change_file_name
  source benchmark/ci/scripts/op_benchmark.config
  for change_file in ${CHANGE_CU_OP_FILES[@]}
  do
    change_file_name=${change_file#*paddle/fluid/operators/}
    if [ -n "${PADDLE_FILENAME_OP_MAP[$change_file_name]}" ]
    then
      for op_name in ${PADDLE_FILENAME_OP_MAP[$change_file_name]}
      do
        LOG "[INFO] Load op: \"${op_name}\"."
        CHANGE_CU_OP_MAP[${op_name}]="$change_file"
      done
    else
      op_name=${change_file_name##*/}
      op_name=${op_name%_cudnn_op*}
      op_name=${op_name%_op*}
      [ -n "${SKIP_OP_MAP[$op_name]}" ] && continue
      LOG "[INFO] Load op: \"${op_name}\"."
      CHANGE_CU_OP_MAP[${op_name}]="$change_file"
    fi
  done
  for change_file in ${CHANGE_CC_OP_FILES[@]}
  do
    change_file_name=${change_file#*paddle/fluid/operators/}
    if [ -n "${PADDLE_FILENAME_OP_MAP[$change_file_name]}" ]
    then
      for op_name in ${PADDLE_FILENAME_OP_MAP[$change_file_name]}
      do
        LOG "[INFO] Load op: \"${op_name}\"."
        CHANGE_CC_OP_MAP[${op_name}]="$change_file"
      done
    else
      op_name=${change_file_name##*/}
      op_name=${op_name%_op*}
      [ -n "${SKIP_OP_MAP[$op_name]}" ] && continue
      LOG "[INFO] Load op: \"${op_name}\"."
      CHANGE_CC_OP_MAP[${op_name}]="$change_file"
    fi
  done
}

# Load ops that will run benchmark test
function load_BENCHMARK_OP_MAP {
  local line op_name api_name
  source benchmark/ci/scripts/op_benchmark.config
  for line in $(cat api_info.txt)
  do
    api_name=${line%%,*}
    if [ -n "${BENCHMARK_APINAME_OP_MAP[$api_name]}" ]
    then
      op_name=${BENCHMARK_APINAME_OP_MAP[$api_name]}
    else
      op_name=$api_name
    fi
    if [ -n "${CHANGE_CU_OP_MAP[$op_name]}" ]
    then
      LOG "[INFO] Load benchmark settings with op \"${op_name}\"."
      BENCHMARK_CU_OP_MAP[$op_name]=$line
    fi
    if [ -n "${CHANGE_CC_OP_MAP[$op_name]}" ]
    then
      LOG "[INFO] Load benchmark settings with op \"${op_name}\"."
      BENCHMARK_CC_OP_MAP[$op_name]=$line
    fi
  done
}

# compile and install paddlepaddle
function compile_install_paddlepaddle {
  LOG "[INFO] Compiling install package ..."
  export WITH_GPU=ON
  export WITH_AVX=ON
  export WITH_MKL=ON
  export RUN_TEST=OFF
  export WITH_PYTHON=ON
  export WITH_TESTING=OFF
  export BUILD_TYPE=Release
  export CUDA_ARCH_NAME=Auto
  export WITH_DISTRIBUTE=OFF
  export CMAKE_BUILD_TYPE=Release
  [ -d build ] && rm -rf build
  bash paddle/scripts/paddle_build.sh build $(nproc)
  [ $? -ne 0 ] && LOG "[FATAL] compile fail." && exit 7
  LOG "[INFO] Uninstall Paddle ..."
  pip uninstall -y paddlepaddle paddlepaddle_gpu
  LOG "[INFO] Install Paddle ..."
  pip install build/python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
}

# run op benchmark test
function run_op_benchmark_test {
  [ ${#BENCHMARK_CC_OP_MAP[*]} -eq 0 -a ${#BENCHMARK_CU_OP_MAP[*]} -eq 0 ] && return
  local task device logs_dir op_name branch_name api_info_file cu_api_info_file cc_api_info_file
  [ -z "$VISIBLE_DEVICES" ] && export VISIBLE_DEVICES=0
  [ "$BENCHMARK_PRINT_FAIL_LOG" != "1" ] && export BENCHMARK_PRINT_FAIL_LOG=1
  cu_api_info_file="$(pwd)/cu_api_info.txt"
  cc_api_info_file="$(pwd)/cc_api_info.txt"
  [ -f "$cu_api_info_file" ] && rm -f $cu_api_info_file
  [ -f "$cc_api_info_file" ] && rm -f $cc_api_info_file
  for api_info in ${BENCHMARK_CU_OP_MAP[*]}
  do
    echo "$api_info" >> $cu_api_info_file
  done
  for api_info in ${BENCHMARK_CC_OP_MAP[*]}
  do
    echo "$api_info" >> $cc_api_info_file
  done
  # install tensorflow for testing accuary
  pip install tensorflow==2.3.0 tensorflow-probability
  for branch_name in "develop" "test"
  do
    git checkout $branch_name
    [ $? -ne 0 ] && LOG "[FATAL] Missing branch ${branch_name}." && exit 7
    LOG "[INFO] Now branch name is ${branch_name}."
    compile_install_paddlepaddle
    logs_dir="$(pwd)/logs-${branch_name}"
    [ -d $logs_dir ] && rm -rf $logs_dir/* || mkdir -p $logs_dir
    pushd benchmark/api > /dev/null
    for device in "gpu" "cpu"
    do
      if [ "${device}" == "gpu" ]
      then
        task="both" && api_info_file=$cu_api_info_file
      else
        task="accuracy" && api_info_file=$cc_api_info_file
      fi
      bash deploy/main_control.sh tests_v2 \
                                  tests_v2/configs \
                                  $logs_dir \
                                  $VISIBLE_DEVICES \
                                  ${device} \
                                  ${task} \
                                  $api_info_file \
                                  "paddle"
    done
    popd > /dev/null
  done
}

# check benchmark result
function check_op_benchmark_result {
  local logs_dir api_info_file check_status_code
  # default 3 times
  [ -z "${RETRY_TIMES}" ] && RETRY_TIMES=3
  logs_dir=$(pwd)/logs-test_pr
  api_info_file=$(pwd)/cu_api_info.txt
  for retry_time in $(seq 0 ${RETRY_TIMES})
  do
    if [ $retry_time -gt 0 ]; then
      # run op benchmark speed test
      # there is no need to recompile and install paddle
      LOG "[INFO] retry ${retry_time} times ..."
      pushd benchmark/api > /dev/null
      bash deploy/main_control.sh tests_v2 \
                                  tests_v2/configs \
                                  ${logs_dir} \
                                  $VISIBLE_DEVICES \
                                  "gpu" \
                                  "speed" \
                                  ${api_info_file} \
                                  "paddle"
      popd > /dev/null
    fi
    # check current result and update the file to benchmark test
    python ${PADDLE_ROOT}/tools/check_op_benchmark_result.py \
        --develop_logs_dir $(pwd)/logs-develop \
        --pr_logs_dir $(pwd)/logs-test_pr \
        --api_info_file ${api_info_file}
    check_status_code=$?
    # TODO(Avin0323): retry only if the performance check fails
    [ $check_status_code -eq 0 ] && break
  done
  return $check_status_code
}

# diff benchmakr result and miss op
function summary_problems {
  local op_name exit_code
  exit_code=0
  if [ ${#BENCHMARK_CC_OP_MAP[*]} -ne 0 -o ${#BENCHMARK_CU_OP_MAP[*]} -ne 0 ]
  then
    check_op_benchmark_result
    exit_code=$?
  fi
  for op_name in ${!CHANGE_CC_OP_MAP[@]}
  do
    if [ -z "${BENCHMARK_CC_OP_MAP[$op_name]}" ]
    then
      exit_code=8
      LOG "[ERROR] Missing test script of \"${op_name}\"(${CHANGE_CC_OP_MAP[$op_name]}) in benchmark."
    fi
  done
  for op_name in ${!CHANGE_CU_OP_MAP[@]}
  do
    if [ -z "${BENCHMARK_CU_OP_MAP[$op_name]}" ]
    then
      exit_code=8
      LOG "[ERROR] Missing test script of \"${op_name}\"(${CHANGE_CU_OP_MAP[$op_name]}) in benchmark."
    fi
  done
  if [ $exit_code -ne 0 ]; then
    LOG "[INFO] See https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-OP-benchmark-Manual for details."
    LOG "[INFO] Or you can apply for one RD (Avin0323(Recommend), Xreki, luotao1) approval to pass this PR."
    exit $exit_code
  fi
}

function main {
  LOG "[INFO] Start run op benchmark test ..."
  load_CHANGE_OP_FILES
  prepare_benchmark_environment
  load_CHANGE_OP_MAP
  load_BENCHMARK_OP_MAP
  run_op_benchmark_test
  summary_problems
  LOG "[INFO] Op benchmark run success and no error!"
  exit 0
}

main
