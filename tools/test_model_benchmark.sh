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


function check_whl {
    bash -x paddle/scripts/paddle_build.sh build
    [ $? -ne 0 ] && echo "build paddle failed." && exit 1
    pip uninstall -y paddlepaddle_gpu
    pip install build/python/dist/*.whl
    [ $? -ne 0 ] && echo "install paddle failed." && exit 1

    mkdir -p /tmp/pr && mkdir -p /tmp/develop
    unzip -q build/python/dist/*.whl -d /tmp/pr
    rm -f build/python/dist/*.whl && rm -f build/python/build/.timestamp

    git checkout .
    git checkout -b develop_base_pr upstream/$BRANCH
    bash -x paddle/scripts/paddle_build.sh build
    [ $? -ne 0 ] && echo "install paddle failed." && exit 1
    cd build
    unzip -q python/dist/*.whl -d /tmp/develop

    sed -i '/version.py/d' /tmp/pr/*/RECORD
    sed -i '/version.py/d' /tmp/develop/*/RECORD
    diff_whl=`diff /tmp/pr/*/RECORD /tmp/develop/*/RECORD|wc -l`
    if [ ${diff_whl} -eq 0 ];then
        echo "paddle whl does not diff in PR-CI-Model-benchmark, so skip this ci"
        echo "ipipe_log_param_isSkipTest_model_benchmark: 1" 
        exit 0
    else
        echo "ipipe_log_param_isSkipTest_model_benchmark: 0"
    fi
}

function compile_install_paddle {
    export CUDA_ARCH_NAME=Auto
    export PY_VERSION=3.7
    export WITH_DISTRIBUTE=OFF
    export WITH_GPU=ON
    export WITH_TENSORRT=OFF
    export WITH_TESTING=OFF
    export WITH_UNITY_BUILD=ON
    check_whl
}

function prepare_data {
    cd ${cache_dir}
    if [ -d "benchmark_data" ];then 
        echo -e "benchmark_data exist!"
    else
        mkdir benchmark_data
        cd benchmark_data
        mkdir dataset
        cd dataset
        wget --no-proxy -q https://paddle-qa.bj.bcebos.com/benchmark_data/Bert.zip 
        unzip Bert.zip
        wget --no-proxy -q https://paddle-qa.bj.bcebos.com/benchmark_data/imagenet100_data.zip
        unzip imagenet100_data.zip
    fi
}

function run_model_benchmark {
    cd ${cache_dir}/benchmark_data
    if [ -d "benchmark" ];then rm -rf benchmark
    fi
    git clone --recurse-submodules=PaddleClas --recurse-submodules=PaddleNLP https://github.com/paddlepaddle/benchmark.git
    export data_path=${cache_dir}/benchmark_data/dataset
    export BENCHMARK_ROOT=${cache_dir}/benchmark_data/benchmark
    cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
    bash model_ci.sh
}

case $1 in
  whl_check)
    compile_install_paddle
  ;;
  run_benchmark)
    prepare_data
    run_model_benchmark
  ;;
  *)
    echo "Error: Incorrect or no parameters"
    exit 1
  ;;
esac
